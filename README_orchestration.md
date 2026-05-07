# Translation Eval Orchestrator

End-to-end translation benchmarking: **for each cached model, stand up vLLM,
translate every eval JSONL, tear down, repeat — then score the outputs with
xCOMET and IFEval.**

Designed to be run overnight on a single GPU. One bash invocation drives the
whole pipeline; failures on one model don't abort the rest.

---

## What lives where

```
orchestration/                     # this folder — portable orchestration logic
├── README.md
├── .env.example                   # copy to .env, fill in HF_TOKEN
├── Dockerfile.scorer              # scorer image (torch + comet + transformers)
├── docker-compose-vllm.yml        # vLLM serving stack, parametrised by $MODEL
├── docker-compose-scorer.yml      # scorer service
├── run_all_evals.py               # orchestrator — main entry point
├── batch_translate.py             # per-model translation driver
├── score_translations.py          # runs inside the scorer container
└── requirements.txt               # host-side deps for the orchestrator

../                                # parent directory (repo root)
├── hf_cache/                      # shared HF cache — models pulled here
│   └── hub/
│       ├── models--org--name/     # one dir per cached model
│       └── ...
├── mono_sea.jsonl                 # eval JSONL (source, translation, lang_codes)
├── mono_me.jsonl
└── .env                           # optional — docker compose also picks this up
```

The orchestration folder is self-contained, but expects **shared resources
(HF cache, eval JSONLs) in the parent directory**. All paths in the scripts
and compose files resolve relative to this folder.

---

## Prerequisites

- Linux host with NVIDIA GPU and recent drivers
- Docker + Docker Compose, with the NVIDIA container runtime
  (`docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` should work)
- Python 3.10+ on the host (for running `run_all_evals.py` and `batch_translate.py`)
- ~50 GB free disk (model weights + scorer image + results)
- A HuggingFace account with access to gated models you plan to use (see
  [Adding a new model](#adding-a-new-model))

---

## One-time setup

From inside this `orchestration/` folder:

```bash
# 1. Python deps for the orchestrator itself (batch translation runs on the host)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Copy the env template and paste your HuggingFace token
cp .env.example .env
$EDITOR .env                 # replace hf_REPLACE_ME with your real token
chmod 600 .env

# 3. Build the scorer image (only needed once)
docker compose -f docker-compose-scorer.yml build

# 4. Verify the scorer container boots and sees the GPU
docker compose -f docker-compose-scorer.yml run --rm scorer \
  python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The build step downloads PyTorch, transformers, and unbabel-comet — a few
minutes depending on bandwidth. Subsequent runs reuse the cached layers.

---

## Adding a new model

The orchestrator auto-discovers any model in `../hf_cache/hub/`. To add one:

### 1. Request access if it's gated

Many strong translation models (gemma-4, Meta Llama, etc.) require you to
accept a license on the model's HuggingFace page before you can download it.
Visit e.g. `https://huggingface.co/google/gemma-4-E2B-it`, click "Agree and
access", then make sure your `.env` contains a token with access.

### 2. Pull the weights into the shared cache

Using the `huggingface-cli` (simplest):

```bash
# From this folder (the venv needs huggingface_hub, which comes with transformers):
pip install huggingface_hub
HF_HUB_CACHE=../hf_cache/hub huggingface-cli download google/gemma-4-E2B-it
```

Or set `HF_HOME=../hf_cache` once and let any tool (transformers, vLLM, etc.)
cache there.

You should now see `../hf_cache/hub/models--google--gemma-4-E2B-it/`.

### 3. (Optional) Verify vLLM can serve it

```bash
MODEL=google/gemma-4-E2B-it docker compose -f docker-compose-vllm.yml up -d
docker compose -f docker-compose-vllm.yml logs -f vllm
# Wait for "Application startup complete", then:
curl http://localhost:8000/v1/models
docker compose -f docker-compose-vllm.yml down
```

### 4. (Optional) Per-model vLLM flags

If a model needs specific `--quantization`, `--max-model-len`, or trust-
remote-code flags, either:

- Pass them per-run:
  ```bash
  VLLM_EXTRA_ARGS="--quantization fp8 --max-model-len 8192" \
    python run_all_evals.py
  ```
- Or add a dedicated compose file (e.g. `docker-compose-vllm-qwen.yml`) that
  hardcodes the flags, then point the orchestrator at it:
  ```bash
  COMPOSE_FILE=docker-compose-vllm-qwen.yml python run_all_evals.py
  ```

### 5. (Optional) Skip models

To run against a subset without removing weights from the cache:

```bash
MODELS=google/gemma-4-E2B-it,tencent/HY-MT1.5-1.8B-FP8 python run_all_evals.py
```

Known scoring/embedding repos (`Unbabel/*`, `cross-encoder/*`,
`sentence-transformers/*`) are auto-filtered even without `MODELS=`.

---

## Running the eval

### Smoke test (10 samples per JSONL, ~10–20 min total)

```bash
OUTPUT_DIR=results_smoke MAX_EXAMPLES=10 python run_all_evals.py
```

### Full overnight run

```bash
# Create the dir first if you want to tee output to a file
mkdir -p results_overnight

OUTPUT_DIR=results_overnight python run_all_evals.py \
  2>&1 | tee results_overnight/orchestrator.log
```

Run inside `tmux` or `screen` so a disconnected SSH doesn't kill it.

### Generalised usage (all knobs together)

Every variable below is optional. A minimal invocation is just
`python run_all_evals.py` — it'll run every model cached under
`../hf_cache/hub/` and score at the end.

```bash
source .venv/bin/activate

MODELS=provider/your-model-name \                       # comma-sep list; omit to use everything cached
OUTPUT_DIR=results_overnight \                          # where translations / logs / scored CSVs land
VLLM_EXTRA_ARGS="--quantization fp8" \                  # forwarded to the vLLM container: --quantization fp8, --enforce-eager, --max-model-len 2048, ...
COMPOSE_FILE=docker-compose-vllm.yml \                  # swap for a model-specific compose file if needed
MAX_EXAMPLES= \                                         # "" = full run; set e.g. 50 for a smoke test
SKIP_SCORING=0 \                                        # 1 to skip the final scorer pass
HF_TOKEN=hf_xxx \                                       # required for gated repos (all Gemma)
  python run_all_evals.py
```

Example — run `google/gemma-4-E4B-it` with on-the-fly FP8 into the
overnight folder:

```bash
MODELS=google/gemma-4-E4B-it \
VLLM_EXTRA_ARGS="--quantization fp8" \
OUTPUT_DIR=results_overnight \
  python run_all_evals.py
```

### Useful env vars

| Var | Default | Meaning |
|---|---|---|
| `OUTPUT_DIR` | `./results` | Where CSVs, logs, and scored CSVs land |
| `MAX_EXAMPLES` | *(all)* | Cap per JSONL. Use for smoke testing |
| `MODELS` | *(auto)* | Comma-separated allowlist of HF model ids |
| `COMPOSE_FILE` | `docker-compose-vllm.yml` | Override the vLLM stack |
| `VLLM_EXTRA_ARGS` | *(none)* | Extra CLI flags forwarded to the vLLM container (quantization, max-model-len, etc.) |
| `HF_TOKEN` | *(from `.env`)* | HuggingFace token for gated repos |
| `SCORE_METRICS` | `ifeval,qe` | Comma-separated subset of `ifeval`, `qe`, `ref` |
| `COMET_BATCH_SIZE` | `16` | xCOMET batch size. Drop to 8 if OOM |
| `COMET_FP16` | `1` | Cast xCOMET to FP16. `0` for FP32 |
| `IFEVAL_BATCH_SIZE` | `32` | Batch size for the NLI classifier (`cross-encoder/nli-deberta-v3-large`). Drop if VRAM is tight |
| `SKIP_SCORING` | `0` | Skip the scorer run at the end |
| `READY_TIMEOUT` | `600` | Seconds to wait for vLLM per model |

### After the run — summarise

A small post-processing script at the repo root rolls the scored CSVs
into a comparison table (one row per model × dataset, plus per-language
breakdown and best/worst/polluted exemplars):

```bash
cd ..    # repo root
.venv/bin/python summarize_results.py --results-dir orchestration/results_overnight
```

Writes `summary_overall.csv`, `summary_by_language.csv`,
`summary_examples.csv`, and `summary.json` into the same dir.

---

## Outputs

After a run, `OUTPUT_DIR/` contains:

```
results_overnight/
├── orchestrator.log                                # your tee output
├── google_gemma-4-E2B-it_run.log                   # per-model stdout
├── google_gemma-4-E2B-it_translations_mono_me.csv  # raw translations
├── google_gemma-4-E2B-it_translations_mono_me.csv.meta.json   # timing + config
├── google_gemma-4-E2B-it_mono_me_scored.csv        # with ifeval + comet_qe cols
└── ... (same 4 files per model × per JSONL)
```

**Scored CSV columns:**

| Column | Meaning |
|---|---|
| `source` | JSONL `text` — source sentence |
| `hypothesis` | Model translation |
| `reference` | JSONL `translation` (may be synthetic) |
| `ifeval` | 1.0 = clean, 0.0 = polluted with preamble/explanation |
| `comet_qe` | xCOMET-XL quality score, reference-free. Higher = better |
| `comet` | (only if `SCORE_METRICS` includes `ref`) Reference-based xCOMET |

Meta JSON (`*.meta.json`) has `elapsed_seconds`, `req_per_s`, `n_errors`,
`concurrency`, `num_shots`, `timestamp`.

---

## Troubleshooting

### Smoke-test quick sanity check

```bash
OUTPUT_DIR=results_smoke MAX_EXAMPLES=10 python run_all_evals.py
```

If this completes with scored CSVs, the pipeline is healthy.

### vLLM doesn't come up / timeout hit

- Check `docker compose -f docker-compose-vllm.yml logs -f vllm` for a CUDA
  OOM, missing license acceptance, or wrong CUDA image tag
- Ensure nothing else is on the GPU: `nvidia-smi`
- For gated models, confirm `HF_TOKEN` is exported / in `.env`

### Scorer OOMs

```
torch.OutOfMemoryError: CUDA out of memory.
```

Drop the batch size:
```bash
COMET_BATCH_SIZE=8 OUTPUT_DIR=results_overnight \
  docker compose -f docker-compose-scorer.yml run --rm scorer
```
Or fall back to FP32 if you suspect FP16 numerical issues:
```bash
COMET_FP16=0 COMET_BATCH_SIZE=4 ...
```

### "Unbabel/XCOMET-XL is not in the available_legacy_metrics"

You haven't accepted the XCOMET-XL license. Visit
https://huggingface.co/Unbabel/XCOMET-XL, click "Agree and access", and make
sure your token is in `.env`. Verify the container sees it:
```bash
docker compose -f docker-compose-scorer.yml run --rm scorer \
  bash -c 'echo "HF_TOKEN length: ${#HF_TOKEN}"'   # expect ~37
```

### "XLMRobertaTokenizer has no attribute build_inputs_with_special_tokens"

Version mismatch between `transformers` and `unbabel-comet`. The pinned
versions in `Dockerfile.scorer` (`transformers 4.44.x`, `unbabel-comet
2.2.2`) are known-good; if you bump them, test on the smoke set first.

### Scorer picks up an unknown model as an LLM

Add the repo to `SCORER_MODEL_IDS` or `SCORER_ORG_PREFIXES` in
`run_all_evals.py`, or use `MODELS=` to whitelist explicitly.

---

## Design notes

- **Per-model vLLM swap cost** is unavoidable (~1–3 min per model for weights
  load). That's the main overhead; translation itself is fast (vLLM does
  continuous batching at `CONCURRENCY=32` by default).
- **Scoring is slow relative to translation** because xCOMET-XL is a
  3.5B-param cross-encoder running per-row in PyTorch Lightning — no
  continuous batching, no kernel fusion, no quantization. FP16 + batch 16
  is the sweet spot on 16 GB; more VRAM = larger batch.
- **Scorer models live in the same HF cache as LLMs.** After the first
  scoring run, `../hf_cache/hub/` gets `Unbabel/XCOMET-XL` and
  `cross-encoder/nli-deberta-v3-large`. `discover_models()` filters these
  out so they're never served via vLLM.
- **Translations are captured once and re-scored on demand.** If you tweak
  metrics or COMET settings, skip translation and just re-run the scorer:
  ```bash
  OUTPUT_DIR=results_overnight \
    docker compose -f docker-compose-scorer.yml run --rm scorer
  ```
