# ClearML on the Tailscale Mesh

This README covers the ClearML orchestration layer that wraps the translation
eval harness in this folder: how to bring up the server and agents, manage
model weights, run and extend the pipeline, and fix common failures.

For the underlying harness (vLLM, batch_translate.py, scoring) without ClearML,
see [README_orchestration.md](README_orchestration.md).

---

## Quick-start

**You are setting up the host (first time, one machine):**

```bash
git clone https://github.com/o6-webwork/clearml-eval-harness.git ~/Desktop/clearml-eval-harness
cd ~/Desktop/clearml-eval-harness
cp clearml-server/.env.example clearml-server/.env 2>/dev/null; true
cp .env.example .env && $EDITOR .env                  # set HF_TOKEN
cp conf/clearml.conf.template ~/clearml.conf
# Bring up the server, open the UI, mint credentials, then:
$EDITOR ~/clearml.conf                                # set CLEARML_HOST, ACCESS_KEY, SECRET_KEY
bash clearml_agent_setup.sh --start
python clearml_register_datasets.py
```

→ Full walkthrough in [§1](#1-host-setup).

> **On Windows?** Run everything inside WSL2. Follow the
> [Windows setup (WSL2)](#windows-setup-wsl2) section first, then return here.

**You are joining as a worker:**

```bash
git clone https://github.com/o6-webwork/clearml-eval-harness.git ~/Desktop/clearml-eval-harness
cd ~/Desktop/clearml-eval-harness
cp .env.example .env && $EDITOR .env                  # set HF_TOKEN
docker compose -f docker-compose-scorer.yml build
ssh-keygen -t ed25519 && ssh-copy-id <user>@<CANONICAL_HOST_IP>
CANONICAL_HOST=<user>@<CANONICAL_HOST_IP> bash clearml_sync_data.sh
cp conf/clearml.conf.template ~/clearml.conf
$EDITOR ~/clearml.conf                                # same values as host clearml.conf
bash clearml_agent_setup.sh --mode 1x --start         # or --mode 2x / --mode 4x
```

→ Full walkthrough in [§4](#4-worker-setup).

---

## Windows setup (WSL2)

All commands in this README run inside a Linux environment. On Windows, use
WSL2 — it gives you a full Ubuntu shell, runs the Docker containers natively,
and passes through the NVIDIA GPU. Skip this section if your machine runs Linux.

### W1. Install WSL2 and Ubuntu

From PowerShell (run as Administrator):

```powershell
wsl --install -d Ubuntu
```

Reboot when prompted. Ubuntu finishes setup on first launch — create a UNIX
username and password when asked. Open subsequent sessions via the **Ubuntu**
app or `wsl` in any terminal.

### W2. Install Docker Desktop

Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
During and after install:

- Backend: select **WSL2** (not Hyper-V)
- After install: **Settings → Resources → WSL Integration → enable Ubuntu**

Verify from inside WSL2:

```bash
docker run --rm hello-world
```

### W3. NVIDIA GPU passthrough (GPU machines)

The NVIDIA driver lives on the **Windows side** — do not install it inside
WSL2. If you already have a recent GeForce or Quadro driver on Windows, WSL2
inherits it automatically.

Install the NVIDIA Container Toolkit inside WSL2:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```

Restart Docker Desktop, then verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

### W4. Clone inside the WSL2 filesystem

Always clone into the WSL2 home directory — not under `/mnt/c/`:

```bash
# Correct — WSL2 filesystem, fast I/O, correct Docker volume permissions
git clone https://github.com/o6-webwork/clearml-eval-harness.git ~/Desktop/clearml-eval-harness

# Wrong — Windows filesystem mount, slow I/O, Docker bind-mount permission issues
# git clone ... /mnt/c/Users/you/Desktop/clearml-eval-harness
```

### W5. SSH server setup (Windows hosts only)

Workers pull model weights from the host over SSH. On Windows the cleanest
approach is to enable the built-in **Windows OpenSSH Server** and configure it
to drop incoming connections straight into WSL2, so workers see no difference
from a Linux host.

**1. Enable the Windows OpenSSH Server** (PowerShell as Administrator):

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
```

**2. Set WSL2 as the default shell** for incoming SSH sessions:

```powershell
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" `
  -Name DefaultShell `
  -Value "C:\Windows\System32\wsl.exe" `
  -PropertyType String -Force
```

**3. Check the firewall rule** (usually created automatically):

```powershell
Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP"
# If the above returns nothing, create it:
New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH Server (sshd)" `
  -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

Workers then use the machine's **Tailscale IP** (shown in the Tailscale system
tray icon on the Windows side) as `CANONICAL_HOST`. Their `ssh-copy-id` and
rsync calls connect to the Windows OpenSSH Server, which forwards the session
into WSL2 transparently.

Verify from another machine on the mesh:

```bash
ssh <WINDOWS_TAILSCALE_IP> exit   # should open a bash shell (WSL2)
```

---

Once the above is done, continue with §1 (host) or §4 (worker). All remaining
commands run inside the **WSL2 Ubuntu terminal**.

---

## 1. Host setup

Do this once, on one machine. This machine runs the ClearML server and is the
canonical source of truth for model weights.

### Prerequisites

- Tailscale up, reachable on the mesh
- Docker with the NVIDIA container runtime
- Python 3.10+
- ~50 GB free for model weights; a few GB for the server containers

### 1a. Clone the repo

```bash
git clone https://github.com/o6-webwork/clearml-eval-harness.git ~/Desktop/clearml-eval-harness
cd ~/Desktop/clearml-eval-harness
```

`~/Desktop/clearml-eval-harness` is a convention used across the mesh so machines can
refer to each other by a consistent path. The Python scripts anchor themselves
to their own location and work from any clone path. If a machine clones
elsewhere, pass `REMOTE_DATA_DIR=<actual-path>` when other nodes sync
from it (see §2d).

### 1b. Set your HuggingFace token

The token is required for gated models (Gemma, Unbabel/XCOMET-XL). Generate a
read-only token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
cp .env.example .env
$EDITOR .env    # replace hf_REPLACE_ME with your token
```

`.env` is git-ignored. Docker Compose auto-loads it, so `HF_TOKEN` is available
to vLLM and scorer containers without any extra export.

### 1c. Bring up the ClearML server

```bash
cd clearml-server
docker compose up -d
cd ..
```

The stack (apiserver / webserver / fileserver / mongo / redis / elasticsearch)
takes about 30 seconds to initialise. The UI is available at
`http://<THIS_MACHINE_TAILSCALE_IP>:8080`.

### 1d. Create your account and mint API credentials

1. Open `http://<THIS_MACHINE_TAILSCALE_IP>:8080`
2. Complete the first-run registration
3. Navigate to **Settings → Workspace → Create new credentials**
4. Save the `access_key` / `secret_key` pair — you need them in the next step

### 1e. Configure clearml.conf

```bash
cp conf/clearml.conf.template ~/clearml.conf
$EDITOR ~/clearml.conf
```

Replace the three placeholders: `CLEARML_HOST`, `ACCESS_KEY`, `SECRET_KEY`.

> **Two gotchas baked into the template:**
> The file must be at `~/clearml.conf` (no dot prefix) — `.clearml.conf` is
> silently ignored. URLs must stay wrapped in double quotes — HOCON treats `//`
> as a comment delimiter, so bare `http://host:port` silently becomes `http:`,
> producing `NameResolutionError: clearml_host` at runtime.

### 1f. Accept SSH connections from workers

Workers pull model weights from the host over SSH (via `clearml_sync_data.sh`).
Ubuntu desktop installs do not ship with `openssh-server` — you need to install
it explicitly:

```bash
sudo apt install -y openssh-server
sudo systemctl enable --now ssh

# Verify it's listening
ss -tlnp | grep :22
```

No further SSH configuration is needed — workers authenticate using ed25519
keys they generate themselves and push via `ssh-copy-id` (see §4c).

### 1h. Install the agent and create queues

```bash
bash clearml_agent_setup.sh --start           # single-GPU / legacy
bash clearml_agent_setup.sh --mode 1x --start # multi-GPU, one task per GPU
```

This installs `clearml-agent` into an isolated venv at `~/.clearml/agent-venv`,
creates the `default` and GPU queues server-side, and starts one daemon per
queue. All daemons should appear in the UI under **Workers & Queues** within
seconds.

The two-queue split is load-bearing: the pipeline controller runs on `default`;
GPU steps run on `gpu` (or `gpu-1x` / `gpu-2x` / `gpu-4x` in named-queue mode).
Routing both to the same queue on a single-GPU host deadlocks — the controller
holds the only agent slot while waiting for its own child step. See
[Appendix](#appendix-under-the-hood) for the full explanation.

### 1i. Register eval datasets

```bash
python clearml_register_datasets.py
```

Scans the repo root recursively for `*.jsonl` files and registers each as a ClearML
Dataset. Descriptions are read from `conf/datasets.yml` — edit that file to add
a description for any new eval file before running, otherwise the script will
prompt you to confirm a generic description (`Registered DD-MM-YYYY — filename`)
or skip that file.

Dataset metadata lives on the server; the files are stored in the fileserver
volume on this machine — nothing goes to GitHub or S3.

To register files from a different location:

```bash
python clearml_register_datasets.py --datasets-dir /path/to/jsonls
```

---

## 2. Local model repository

### 2a. Where the HF cache lives

Model weights live at `hf_cache/hub/` inside the repo. Both
`run_all_evals.py` and `clearml_pipeline.py` default to this location.
Override it with the `HF_HUB_DIR` env var if needed.

**The host is the single source of truth for model weights.** Workers never
download from HuggingFace directly — they sync from the host via
`clearml_sync_data.sh`. This keeps HF quota intact and means a slow or gated
download only happens once.

### 2b. Adding a new model to the host

Use `huggingface-cli download` — it is explicit, resumable, and gives clear
progress output:

```bash
cd ~/Desktop/clearml-eval-harness

# Public model
huggingface-cli download <org/model-id> \
  --cache-dir hf_cache

# Gated model (Gemma, etc.) — your HF account must have accepted the licence page
huggingface-cli download <org/model-id> \
  --cache-dir hf_cache \
  --token $HF_TOKEN
```

After downloading, verify with:
```bash
ls hf_cache/hub/ | grep models--
```

The model will be picked up automatically on the next pipeline run via
`discover_models()` (see §2c). No further registration is needed.

> **On vLLM auto-download:** the container downloads weights on first
> `docker compose up` if they are not already cached. It works, but progress is
> invisible, resumption is unreliable, and a missing `HF_TOKEN` causes a silent
> hang. Prefer `huggingface-cli download` for any model you're deliberately
> adding to the eval set.

### 2c. How models are discovered at run time

`discover_models()` scans `hf_cache/hub/` for directories named
`models--<org>--<name>`, converts them to HF IDs (`org/name`), and
automatically filters out known scoring/embedding models (Unbabel/XCOMET-XL,
cross-encoder/\*, etc.) so they don't get queued for vLLM.

Override discovery with the `MODELS` env var to target a specific subset:

```bash
# Auto-discover everything in the cache
python clearml_pipeline.py --local

# Run only these two models
MODELS=tencent/HY-MT1.5-1.8B-FP8,NiuTrans/LMT-60-4B \
  python clearml_pipeline.py --local

# Preview what would be discovered without running anything
python -c "from run_all_evals import discover_models; print('\n'.join(discover_models()))"
```

### 2d. Syncing model weights to workers

Workers pull the HF cache from the host before their first eval:

```bash
# Standard — host has models at hf_cache/ (post-consolidation layout)
CANONICAL_HOST=<user>@<HOST_TAILSCALE_IP> bash clearml_sync_data.sh

# Pre-consolidation host — models still at ~/Desktop/evals/hf_cache/
CANONICAL_HOST=<user>@<HOST_TAILSCALE_IP> \
  REMOTE_DATA_DIR=~/Desktop/evals \
  bash clearml_sync_data.sh
```

> **`CANONICAL_HOST` must include the username** — rsync uses `user@host` format,
> not a bare IP. Example: `CANONICAL_HOST=alice@100.64.0.10 bash clearml_sync_data.sh`

rsync is resumable — a dropped SSH connection won't restart a large download
from zero. Partial syncs when you only need one side:

```bash
CANONICAL_HOST=<user>@<IP> SYNC_DATA_ONLY=1    bash clearml_sync_data.sh  # JSONLs only
CANONICAL_HOST=<user>@<IP> SYNC_WEIGHTS_ONLY=1 bash clearml_sync_data.sh  # HF cache only
```

---

## 3. Running the pipeline

### 3a. TL;DR — the four most common invocations

```bash
cd ~/Desktop/clearml-eval-harness

# 1. Smoke test — 10 rows, local machine, tracked in UI
MODELS=tencent/HY-MT1.5-1.8B-FP8 \
  python clearml_pipeline.py --local --max-examples 10

# 2. Full local run — all cached models, all metrics
python clearml_pipeline.py --local

# 3. Distributed — fans out translate + scoring steps to GPU workers
python clearml_pipeline.py --queue gpu-1x

# 4. Score only — reuse translations from a prior run, skip re-translating
python clearml_pipeline.py --local \
  --translation-dataset <dataset-id> --metrics qe,ref
```

The pipeline appears in the UI at:
`http://<CLEARML_HOST>:8080` → **translation-evals/pipelines** → `translation-eval-run`

### 3b. Pipeline structure

```
┌────────────┐  ┌────────────┐       ┌────────────┐
│ translate   │  │ translate   │  ...  │ translate   │
│ model-1     │  │ model-2     │       │ model-N     │
└─────┬──────┘  └─────┬──────┘       └─────┬──────┘
      │               │                     │
      └───────────────┴──────────┬──────────┘
                   ┌─────────────┼─────────────┐
                   ▼             ▼              ▼
           score_ifeval  score_xcomet_qe  score_xcomet_ref
           (NLI clean %) (ref-free)       (ref-based)
```

In distributed mode translate steps run in parallel across workers. In
`--local` mode they are chained sequentially — one vLLM container at a time on
the single GPU. Scoring branches always run in parallel in distributed mode; in
`--local` mode they are also chained to avoid Docker network creation races.

Each translate step publishes its output CSVs as a ClearML Dataset
(`translations/<safe_model>/<run_id>`). Scoring steps can fetch this by Dataset
ID via `--translation-dataset`, allowing re-scoring without re-translating.

### 3c. Output layout

```
results/                                     ← --output-dir (default)
  <safe_model>_translations_<dataset>.csv    ← raw translations
  <safe_model>_run.log                       ← vLLM + batch_translate stdout
  ifeval/
    <safe_model>_<dataset>_scored.csv
  qe/
    <safe_model>_<dataset>_scored.csv
  ref/
    <safe_model>_<dataset>_scored.csv
```

`<safe_model>` is the HF model ID with `/` and `:` replaced by `_`.

### 3d. Full CLI reference

```
python clearml_pipeline.py [OPTIONS]
```

| Flag | Default | Example | Description |
|------|---------|---------|-------------|
| `--local` | off | `python clearml_pipeline.py --local` | Run on this machine (still tracked in ClearML UI). Steps are chained sequentially to avoid GPU contention. |
| `--queue` | `gpu` | `python clearml_pipeline.py --queue gpu-1x` | Fallback ClearML queue for models not in `conf/models.yml`. Ignored when `--local` is set. |
| `--output-dir` | `results/` | `python clearml_pipeline.py --output-dir results_overnight` | Directory for CSVs and logs |
| `--max-examples` | *(all rows)* | `python clearml_pipeline.py --local --max-examples 10` | Row cap per eval file. Useful for smoke tests. |
| `--metrics` | `ifeval,qe,ref` | `python clearml_pipeline.py --local --metrics qe,ref` | Comma-separated scoring branches to run. Valid values: `ifeval`, `qe`, `ref`. |
| `--vllm-extra-args` | `` | `python clearml_pipeline.py --vllm-extra-args "--quantization fp8"` | Extra arguments injected into vLLM. |
| `--translation-dataset` | `` | `python clearml_pipeline.py --local --translation-dataset abc123` | ClearML Dataset ID of prior translation outputs. Skips all translate steps and runs scoring only. |
| `--translations-dir` | `` | `python clearml_pipeline.py --local --translations-dir results_overnight --metrics qe,ref` | Local directory containing pre-translated `*_translations*.csv` files. Skips translate steps and scores from this directory. Must be used with `--local`. |
| `--compose-file` | `docker-compose-vllm.yml` | `python clearml_pipeline.py --compose-file docker-compose-vllm-e4b.yml` | vLLM compose file. Override for non-standard hardware configs. |

**Environment variables:**

| Variable | Example | Description |
|----------|---------|-------------|
| `MODELS` | `MODELS=tencent/HY-MT1.5-1.8B-FP8,NiuTrans/LMT-60-4B python clearml_pipeline.py --local` | Comma-separated HF model IDs. Overrides `hf_cache/hub/` auto-discovery. |
| `HF_TOKEN` | `HF_TOKEN=hf_abc123 huggingface-cli download google/gemma-4-E2B-it --cache-dir hf_cache` | HuggingFace auth token. Required for gated models. Loaded automatically from `.env` at runtime. |
| `HF_HUB_DIR` | `HF_HUB_DIR=/mnt/models/hf_cache/hub python clearml_pipeline.py --local` | Override the HF cache directory (default: `hf_cache/hub`). |
| `VLLM_EXTRA_ARGS` | `VLLM_EXTRA_ARGS="--tensor-parallel-size 4" python clearml_pipeline.py --queue gpu` | Same as `--vllm-extra-args` but as an env var. Useful when triggering from a script. |

### 3e. Full mesh reset (clean slate)

Use this after a pipeline refactor that changes the DAG shape, to clear
smoke-test pollution from the project view, or when the server and workers
have fallen out of sync. Run the host steps first, then the worker steps on
every other machine.

#### On the host

```bash
# 1. Stop all agent daemons — they hold credentials that become stale after
#    the wipe and will fail silently if left running.
pkill -f 'clearml-agent daemon'
pgrep -af 'clearml-agent daemon'   # verify — should return nothing

# 2. Wipe all server data (mongo / elasticsearch / redis / fileserver / logs)
cd ~/Desktop/clearml-eval-harness/clearml-server
docker compose down -v && docker compose up -d
cd ..

# 3. Open the UI → create user → mint new credentials
#    http://<CLEARML_HOST>:8080 → Settings → Workspace → "+ Create new credentials"

# 4. Update ONLY the credentials block in ~/clearml.conf — do NOT copy the
#    template over, it would overwrite your URLs with placeholders.
$EDITOR ~/clearml.conf

# 5. Clear the local per-task venv cache — the agent reuses cached task
#    environments keyed by dependency hash; stale caches survive a server wipe
#    and can cause ModuleNotFoundError or import mismatches on the next run.
rm -rf ~/.clearml/venvs-cache/*

# 6. Clear the local dataset/artifact cache — locally cached dataset copies
#    from the old server are now orphaned and will cause Dataset.get() to
#    return stale data instead of pulling fresh from the new server.
rm -rf ~/.clearml/cache/*

# 7. Re-create the queues and restart the daemons.
bash clearml_agent_setup.sh --start

# 8. Re-register datasets (conf/datasets.yml has the descriptions — no prompts).
python clearml_register_datasets.py
```

#### On every worker

```bash
# 1. Stop all agent daemons.
pkill -f 'clearml-agent daemon'
pgrep -af 'clearml-agent daemon'   # verify

# 2. Update credentials in ~/clearml.conf (same new keys as the host).
$EDITOR ~/clearml.conf

# 3. Clear venv and artifact caches.
rm -rf ~/.clearml/venvs-cache/*
rm -rf ~/.clearml/cache/*

# 4. Restart the agent.
bash clearml_agent_setup.sh --start
```

Workers should reappear in the UI under **Workers & Queues** within seconds.

### 3f. Per-model queue and tensor-parallel configuration

`conf/models.yml` maps HuggingFace model IDs to the queue they should run on
and how many GPUs vLLM should use (tensor-parallel size).

```yaml
# conf/models.yml
google/gemma-4-E2B-it:
  queue: gpu-1x
  tensor_parallel_size: 1

some-org/large-model-180B:
  queue: gpu-4x
  tensor_parallel_size: 4
```

- **`queue`** must match a queue registered on the ClearML server and consumed
  by a running agent. Use the `gpu-1x / gpu-2x / gpu-4x` naming convention from
  `clearml_agent_setup.sh --mode`.
- **`tensor_parallel_size`** must equal the number of GPUs assigned to the
  target queue's agents. For `gpu-2x` that is 2; for `gpu-4x` that is 4.
- Models **not listed** in this file fall back to the `--queue` CLI argument
  (default: `gpu`) with `tensor_parallel_size=1`.

The pipeline reads this file automatically at the start of each run and prints
the per-model routing before submitting:

```
Models (3):
  - tencent/HY-MT1.5-1.8B-FP8  (queue=gpu-1x, tp=1)
  - NiuTrans/LMT-60-4B          (queue=gpu-1x, tp=1)
  - some-org/large-model-180B   (queue=gpu-4x, tp=4)
```

The file lives at `conf/models.yml` and is checked into
the repo. Add an entry for any new model and commit before triggering a run
that targets it on a specific queue.

---

## 4. Worker setup

Do this on every machine that should pull tasks from the queue.

### Prerequisites

- Tailscale up, on the same tailnet as the host
- Docker with the NVIDIA container runtime
- Python 3.10+
- ~50 GB free for model weights
- Passwordless SSH to the canonical host (for the weight sync — set up in §4c)

### 4a. Clone the repo and build the scorer image

```bash
git clone https://github.com/o6-webwork/clearml-eval-harness.git ~/Desktop/clearml-eval-harness
cd ~/Desktop/clearml-eval-harness
docker compose -f docker-compose-scorer.yml build
```

### 4b. Set your HuggingFace token

```bash
cp .env.example .env
$EDITOR .env    # replace hf_REPLACE_ME with your token
```

### 4c. Set up passwordless SSH to the canonical host

The host must have `openssh-server` running before this will work — see §1f if
it hasn't been set up yet.

```bash
ssh-keygen -t ed25519           # press enter through all prompts
ssh-copy-id <CANONICAL_HOST_IP>
ssh <CANONICAL_HOST_IP> exit    # verify — should log in without a password
```

### 4d. Sync eval data and model weights

```bash
CANONICAL_HOST=<user>@<CANONICAL_HOST_IP> bash clearml_sync_data.sh
```

`CANONICAL_HOST` must include the SSH username — rsync requires `user@host`,
not a bare IP. This pulls both the eval JSONLs and the full HF model cache from
the host. See §2d for the `REMOTE_DATA_DIR` flag if the host is still on the
old `~/Desktop/evals` layout.

### 4e. Configure clearml.conf

```bash
cp conf/clearml.conf.template ~/clearml.conf
$EDITOR ~/clearml.conf    # same CLEARML_HOST / ACCESS_KEY / SECRET_KEY as the host
```

### 4f. Install the agent

Before starting agents, export `ORCH_DIR` so that step functions can locate
repo files regardless of where the agent clones the task:

```bash
export ORCH_DIR=~/Desktop/clearml-eval-harness
# Persist across sessions:
echo 'export ORCH_DIR=~/Desktop/clearml-eval-harness' >> ~/.bashrc
```

Then start the agents:

```bash
# Single-GPU or legacy setup
bash clearml_agent_setup.sh --start

# Multi-GPU — choose the mode that matches your hardware (see §4g)
bash clearml_agent_setup.sh --mode 1x --start   # one daemon per GPU
bash clearml_agent_setup.sh --mode 2x --start   # one daemon per GPU pair
bash clearml_agent_setup.sh --mode 4x --start   # one daemon for all GPUs
```

The worker should appear in the UI under **Workers & Queues** within seconds.

> **Before triggering a distributed run**: make sure the **host** machine is
> not running agent daemons that consume the GPU queues (`gpu`, `gpu-1x`, etc.).
> If the host has its own GPU-queue daemons running, it will pull tasks intended
> for the worker and run them locally. On the host:
> ```bash
> pkill -f 'clearml-agent daemon'
> pgrep -af 'clearml-agent daemon'        # verify — should return nothing
> clearml-agent daemon --queue default --detach   # restart controller only
> ```

### 4g. Multi-GPU workers

`clearml_agent_setup.sh` supports three mutually exclusive GPU allocation modes
via the `--mode` flag. Choose based on the largest model you need to run on this
machine.

| Mode | Queue | Daemons started | Use when |
|------|-------|-----------------|----------|
| *(none — default)* | `gpu` | 1, no GPU binding | Single-GPU hosts or legacy setups |
| `--mode 1x` | `gpu-1x` | 1 per GPU, each bound to its own GPU | All models fit in 1×48 GB |
| `--mode 2x` | `gpu-2x` | 1 per GPU pair (GPUs 0-1, 2-3, …) | Models need 2×48 GB |
| `--mode 4x` | `gpu-4x` | 1 across all GPUs (GPUs 0-3) | Models need 4×48 GB |

```bash
# Examples (from repo root, on the worker):

# 4-GPU machine — run 4 models in parallel, each on one GPU
bash orchestration/clearml_agent_setup.sh --mode 1x --start

# 4-GPU machine — run 2 models at a time, each using 2 GPUs (for 90 GB+ models)
bash orchestration/clearml_agent_setup.sh --mode 2x --start

# 4-GPU machine — run 1 model at a time across all 4 GPUs (for 192 GB+ models)
bash orchestration/clearml_agent_setup.sh --mode 4x --start
```

**How GPU binding works:** `clearml_agent_setup.sh --mode 1x` starts each
daemon with `CUDA_VISIBLE_DEVICES=<N>`, which is inherited by every task the
daemon runs. The vLLM compose file (`docker-compose-vllm.yml`) forwards
`CUDA_VISIBLE_DEVICES` into the container, so each task's vLLM process sees
only its assigned GPUs. The compose file exposes all host GPUs with
`count: all` — `CUDA_VISIBLE_DEVICES` is what limits which ones CUDA actually
uses.

**Queue-to-model routing** is configured in `conf/models.yml` (see §3f). The
pipeline reads this file automatically and dispatches each model to the correct
queue. Models not listed fall back to the `--queue` argument (default: `gpu`).

**Mixed machines:** a fleet can have workers on different modes simultaneously.
For example: two 4-GPU machines running `--mode 1x` (8 × `gpu-1x` slots total)
plus one 4-GPU machine running `--mode 4x` (1 × `gpu-4x` slot). Add a
`conf/models.yml` entry for any large model pointing to `gpu-4x` and it will
be automatically dispatched to the right worker while small models fan out
across the `gpu-1x` fleet.

### 4h. SSH tunnel when ClearML ports are blocked

Docker's iptables rules (`DOCKER-USER` chain) can block inbound connections on
the ClearML ports (8008/8080/8081) even when `ufw allow in on tailscale0` says
they are open. Symptoms: `curl http://<host-ip>:8008` from the worker returns
nothing or times out; `ss -tlnp` on the host shows the ports are listening.

The cleanest fix is to forward the ports over SSH instead of fighting iptables:

```bash
# On the worker — run once per session (forks to background)
ssh -fNL 8008:localhost:8008 \
       -L 8080:localhost:8080 \
       -L 8081:localhost:8081 \
       <user>@<HOST_TAILSCALE_IP>
```

Then update `~/clearml.conf` on the **worker** so all three URLs point to
`localhost` rather than the host's Tailscale IP:

```
api_server: "http://localhost:8008"
web_server: "http://localhost:8080"
files_server: "http://localhost:8081"
```

Verify connectivity before starting agents:

```bash
curl -s http://localhost:8008/health   # should return {}
```

**Tearing down the tunnel** when you are done with the session:

```bash
# Kill the SSH processes holding those ports
kill $(lsof -ti:8008 -ti:8080 -ti:8081)

# Or kill all matching SSH tunnel processes at once
pkill -f "ssh -fNL"

# Verify — should return nothing
lsof -ti:8008
```

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NameResolutionError: clearml_host` | HOCON ate the `//` in a URL | Wrap all URLs in double quotes in `~/clearml.conf` |
| `ConfigMissingException` | Config in wrong location | Must be `~/clearml.conf` — no dot prefix |
| `ModuleNotFoundError: clearml` (agent venv) | Partial install | `pip install clearml clearml-agent` in `~/.clearml/agent-venv` |
| `ModuleNotFoundError: clearml` (task venv) | `clearml` missing from root `requirements.txt` | Add `clearml` to `requirements.txt`, push, then bust the venv cache (next row) |
| Same `ModuleNotFoundError` after pushing the fix | Agent reused a stale cached venv for an unchanged task hash | `rm -rf ~/.clearml/venvs-cache/*` |
| `fatal: reference is not a tree` | Agent fetches from the remote; your local changes weren't pushed | `git push` before re-triggering the task |
| Pipeline deadlock on a single-GPU host | Controller and step routed to the same queue | Run two agents: `clearml-agent daemon --queue default --detach` and `clearml-agent daemon --queue gpu --detach` |
| How to stop a daemon | The detached process keeps running until killed | `clearml-agent daemon --stop` (once per active queue), or `pkill -f 'clearml-agent daemon'`; verify with `pgrep -af 'clearml-agent daemon'` |
| Agents visible in `ps` but never pick up tasks | Stale credentials after a server wipe | Update the `credentials` block in `~/clearml.conf` on every machine, then restart daemons |
| `Could not find queue 'gpu'` or `'default'` | Queue was wiped or never created | Create via UI (**Workers & Queues → +**), or re-run `bash clearml_agent_setup.sh` |
| `Could not find queue 'gpu-1x'` (or `-2x` / `-4x`) | Named queue not yet created | Re-run `bash clearml_agent_setup.sh --mode 1x` (no `--start` needed — it creates the queue without starting daemons) |
| Model dispatched to `gpu` instead of expected `gpu-1x` | Model not in `conf/models.yml`, or HF ID typo | Add entry to `conf/models.yml` with exact HF model ID; check the routing printout at pipeline start |
| `FileNotFoundError: docker-compose-vllm.yml` | Step running from `/tmp` | `orch_dir` is already threaded through `clearml_pipeline.py` — check it is being passed to the step function |
| Container name conflict `orchestration-vllm-1 already in use` | Previous run failed without cleanup | `docker compose -f docker-compose-vllm.yml down` |
| Two translate steps racing for the GPU | Distributed mode, one GPU | Use `--local` to chain steps sequentially |
| `SSH Broken pipe` during `clearml_sync_data.sh` | SSH key not authorised | Re-run `ssh-copy-id <CANONICAL_HOST_IP>` |
| Model not appearing in discovery | Not in `hf_cache/hub/`, or is a scorer model | `ls hf_cache/hub/`; or force a specific set with `MODELS=org/model` |
| `docker: command not found` inside WSL2 | Docker Desktop WSL2 integration not enabled for the Ubuntu distro | Docker Desktop → **Settings → Resources → WSL Integration** → toggle Ubuntu on → **Apply & Restart** |
| `nvidia-smi: command not found` or no GPUs in containers (WSL2) | NVIDIA Container Toolkit not installed in WSL2, or Docker Desktop not restarted after install | Re-run the toolkit install from §W3; restart Docker Desktop fully (system tray → Quit, then relaunch) |
| Workers can SSH to the Windows host but land in PowerShell, not bash | WSL2 default shell not configured | Run the `New-ItemProperty` command from §W5 step 2 in PowerShell as Administrator |
| `curl http://<host>:8008` from worker returns nothing or times out | Docker's `DOCKER-USER` iptables chain blocks the Tailscale interface even when `ufw` allows it | Set up an SSH tunnel and point `~/clearml.conf` to `localhost` (see §4h) |
| SSH tunnel ports already in use on next session | `ssh -fNL` forks to background and stays alive until killed | `kill $(lsof -ti:8008 -ti:8080 -ti:8081)` or `pkill -f "ssh -fNL"` |
| Tasks appear on **host** instead of worker during a distributed run | Host has GPU-queue daemons running that consume `gpu`/`gpu-1x` tasks | Stop GPU-queue daemons on the host: `pkill -f 'clearml-agent daemon'`; restart only `clearml-agent daemon --queue default --detach` |
| `clearml_sync_data.sh` fails with "invalid user" or "Permission denied" | `CANONICAL_HOST` set to a bare IP — rsync requires `user@host` | Use `CANONICAL_HOST=<user>@<HOST_TAILSCALE_IP>` |
| Pipeline dispatches scoring steps to `gpu` queue with no consumer | `--queue` defaulted to `gpu` instead of `gpu-1x` | Pass `--queue gpu-1x` when triggering a distributed run |

---

## 6. Adding new pipeline steps

### 6a. Step contract

Step functions must be **self-contained**: all imports live inside the function
body. ClearML serialises the function and ships it to the worker — any top-level
import won't be present in the worker's environment.

```python
def _step_my_step(output_dir: str, orch_dir: str = "") -> dict:
    import os
    from clearml import Task

    if not orch_dir:
        orch_dir = os.environ.get("ORCH_DIR", os.getcwd())
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(orch_dir, output_dir)

    task = Task.current_task()
    logger = task.get_logger() if task else None

    # ... do work ...

    # Surface failures by raising — ClearML marks a step that returns
    # normally as COMPLETED regardless of what the dict contains.
    if something_went_wrong:
        raise RuntimeError("descriptive message")

    return {"my_output_key": "value"}
```

Always use `orch_dir` to anchor file paths — steps run with an arbitrary CWD
on the worker.

### 6b. Registering the step

Add an `add_function_step()` call inside `build_and_run()` in
`clearml_pipeline.py`. The pipeline is rebuilt from source on every invocation
— there is no frozen definition to migrate.

```python
pipe.add_function_step(
    name="my_step",
    function=_step_my_step,
    function_kwargs={
        "output_dir": output_dir,
        "orch_dir": HERE,
    },
    parents=["score_ifeval"],           # waits for this step before starting
    project_name="translation-evals",
    execution_queue=None if local else queue,
)
```

`parents` controls the DAG edge. Use `parents=[a, b, c]` to wait for multiple
upstream steps (fan-in / merge point).

### 6c. Branching patterns

**Build-time branch** — a CLI flag controls whether the step is registered at
all:
```python
if args.run_my_step:
    pipe.add_function_step(name="my_step", ...)
```

**Runtime skip** — `pre_execute_callback` returning `False` skips the step
without failing the pipeline:
```python
def skip_if_high_errors(pipeline, node, parameters):
    parent = pipeline.get_step_by_name("translate_xxx")
    err_count = max(
        parent.task.get_reported_scalars().get("error_count", {}).get("total", [0])
    )
    return err_count < 0.1 * parent.returns["n_rows"]

pipe.add_function_step(
    name="my_step",
    function=_step_my_step,
    pre_execute_callback=skip_if_high_errors,
    ...
)
```

### 6d. Wrapping existing scripts

If you have a standalone script (e.g. `smoke_rescore_ifeval.py`), don't rewrite
it — wrap it in a step function that shells out:

```python
def _step_rescore(output_dir: str, orch_dir: str = "") -> dict:
    import os, subprocess, sys
    if not orch_dir:
        orch_dir = os.environ.get("ORCH_DIR", os.getcwd())
    rc = subprocess.call(
        [sys.executable, "smoke_rescore_ifeval.py"],
        cwd=orch_dir,
        env={**os.environ, "OUTPUT_DIR": output_dir},
    )
    if rc != 0:
        raise RuntimeError(f"smoke_rescore_ifeval.py exited {rc}")
    return {"output_dir": output_dir}
```

---

## 7. Adding new visualisation artifacts

Every step function gets a ClearML task via `Task.current_task()`. The task's
logger handles all visualisation; the task itself handles file uploads.

```python
task = Task.current_task()
logger = task.get_logger() if task else None
```

The `if task else None` guard lets the step run unchanged outside of ClearML
(e.g. direct local testing without `Task.init()`).

### 7a. What is already logged

| Step | Scalars | Plots | Artifacts |
|------|---------|-------|-----------|
| `_step_translate` | throughput_req_s, elapsed_seconds, error_count, row_count (per dataset) | — | translation CSVs, run log |
| `_step_score_ifeval` | IFEval clean % (per model × dataset) | IFEval summary table | scored CSVs |
| `_step_score_xcomet_qe` | xCOMET-QE mean / median / p10 (per model × dataset) | score distribution histogram, worst-translations table, per-language breakdown, summary table | scored CSVs |
| `_step_score_xcomet_ref` | xCOMET-ref mean (per model × dataset) | ref summary table | scored CSVs |

### 7b. Scalars

Scalars appear on the step's **Plots** tab and can be compared across runs in
the experiment comparison view.

```python
logger.report_scalar(
    title="my_metric",    # groups related series under one chart
    series="dataset_name",
    iteration=0,          # use 0 for single-value metrics; increment for time series
    value=42.0,
)
```

### 7c. Histograms

```python
logger.report_histogram(
    title="score distribution",
    series="my_model",
    values=[0.72, 0.85, 0.61, ...],    # list or 1D numpy array
    iteration=0,
    xaxis="Score",
    yaxis="Count",
)
```

### 7d. Tables

Tables render as interactive plots in the UI. Pass a pandas DataFrame or a list
of dicts:

```python
import pandas as pd

logger.report_table(
    title="summary",
    series="all models × datasets",
    iteration=0,
    table_plot=pd.DataFrame(rows),
)
```

### 7e. File artifacts

Artifacts are stored on the fileserver, linked to the task, and accessible from
the **Artifacts** tab. They can also be fetched programmatically by other steps.

```python
# Single file
task.upload_artifact("my_output", "/path/to/file.csv")

# Directory
task.upload_artifact("my_outputs", "/path/to/directory/")

# Python object (pickled automatically)
task.upload_artifact("my_dict", {"key": "value"})
```

Artifact names must be unique within a task. Uploading twice with the same name
overwrites the previous version.

---

## Appendix: Under the hood

Design decisions that aren't obvious from the code. Read this when something
breaks unexpectedly or when you're wondering why a thing is the way it is.

**Self-hosted server on one canonical host.**
`clearml-server/docker-compose.yml` runs the full stack on one machine. The
unified `allegroai/clearml:latest` image needs a `command:` directive
(`["apiserver"]`, `["webserver"]`, etc.) — omitting it silently breaks routing.
DB services on the compose network are named `elasticsearch`, `mongo`, `redis`
(not `localhost`) — set `CLEARML_ELASTIC_SERVICE_HOST` etc. on every service.

**Agent in an isolated venv.**
`clearml_agent_setup.sh` installs the agent into `~/.clearml/agent-venv` so the
daemon never conflicts with project dependencies. Each task still builds its own
per-task venv on the fly from `clearml_requirements.txt`.

**`~/clearml.conf`, not `.clearml.conf`. URLs in double quotes.**
The file must be at `~/clearml.conf` (no dot prefix) — `.clearml.conf` is
silently ignored. URLs must be wrapped in double quotes — HOCON treats `//` as a
comment delimiter, so bare `http://host:port` silently becomes `http:`.
`conf/clearml.conf.template` bakes both fixes in.

**`add_function_step`, not pre-registered tasks.**
Step functions are defined inline and registered via
`PipelineController.add_function_step(...)`. ClearML serialises the function
body and ships it to the worker — no chicken-and-egg problem of needing task IDs
before any task has been registered. Trade-off: all imports must be inside the
function body, and steps run from `/tmp`, so any repo paths must be passed
explicitly as parameters — that's why `orch_dir=HERE` is threaded through
everywhere.

**Local mode chains steps; distributed mode fans out.**
On the canonical host (one GPU) two simultaneous translate steps would fight
over the GPU and the `orchestration-vllm-1` container name. `build_and_run()`
detects `--local` and chains each translate step as a parent of the next. In
distributed mode, steps have no parents and fan out to separate workers.

**Pipeline controller on `default`, GPU steps on `gpu`.**
The controller is a Python process that spends its life waiting on child steps.
Routing it to the same queue as the GPU steps on a single-GPU host pins the only
agent slot — the child step never gets dequeued. With the split, a lightweight
`clearml-agent --queue default` runs the controller and a separate
`clearml-agent --queue gpu` runs the steps. If you have many GPU workers and
don't mind the controller occupying one, you can collapse to a single queue —
but keep the split for the canonical single-GPU layout.

**Translation outputs published as ClearML Datasets.**
Each translate step publishes its output CSVs as a Dataset named
`translations/<safe_model>/<run_id>`. Scoring steps can fetch this by Dataset
ID via `--translation-dataset`, allowing re-scoring without re-translating.
Files are stored on the fileserver and cached at `~/.clearml/cache/` on each
worker after the first pull.
