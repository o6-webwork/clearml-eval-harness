# Translation Eval — Forward Roadmap

Milestone 1 (IFEval classifier swap to `nli-deberta-v3-large` + GPU +
batching) shipped with the current change. This doc captures what
comes next, so a teammate can pick it up cold.

Stabilisation principle throughout: **small, reversible changes, each
with its own acceptance test**. No big-bang rewrites.

---

## Milestone 2 — Shared artifacts across the Tailscale mesh

**Problem it solves.** Every team member re-downloads the same HF
weights and rebuilds the same container images on their own box. As
the model + eval list grows, this gets harder to manage and the team's
caches drift out of sync.

**Approach — minimum viable shared setup:**

### 2a. Container images → GitHub Container Registry (GHCR)

- Add semver tags to both images:
  - `ghcr.io/<org>/bench-scorer:<tag>` (currently
    [Dockerfile.scorer](Dockerfile.scorer))
  - `ghcr.io/<org>/vllm-openai-gemma4:<tag>` (currently pulled as
    `vllm/vllm-openai:gemma4-cu130`; tag on our side so we can pin
    across teammates)
- Add `orchestration/build_and_push.sh` — builds, tags, pushes both.
- Edit [docker-compose-scorer.yml](docker-compose-scorer.yml) to drop
  the `build:` block and use `image: ghcr.io/<org>/bench-scorer:<tag>`.
  Teammates now `docker compose pull` instead of building.
- Docs: one section in this folder's [README.md](README.md) covering
  "How to rebuild and republish images."

### 2b. HF weight cache → canonical host + rsync

- Designate one box (suggest the DGX — it'll do most eval work) as
  the canonical owner of `../hf_cache/`.
- Ship `orchestration/sync_hf_cache.sh`:
  ```bash
  rsync -avP --delete-after <tailscale-name>:~/evals/hf_cache/hub/ ../hf_cache/hub/
  ```
- Teammates run it before starting an eval. Stateless, resumable,
  easy mental model.
- Keep eval JSONLs (`mono_sea.jsonl`, `mono_me.jsonl`) in git — they're
  small enough.

### Explicitly NOT doing (defer until rsync friction bites)

- NFS over Tailscale — too-easy-to-break long-running mounts.
- S3 / MinIO as HF backend — real infra investment.
- ClearML Datasets — would tie this milestone to Milestone 3 without
  necessarily delivering more value.

### Acceptance criteria

1. A fresh box runs `docker compose pull` from the scorer compose file
   and gets a working image without building locally.
2. `sync_hf_cache.sh` completes from a fresh box; `ls ../hf_cache/hub/`
   matches the canonical host's listing exactly.
3. A `MAX_EXAMPLES=50` smoke eval runs end-to-end on the fresh box
   without any local image builds or ad-hoc `huggingface-cli` pulls.

---

## Milestone 3 — ClearML tracking POC

**Problem it solves.** Runs are identified by filename stems and
`.meta.json` blobs; there's no searchable history of what was run,
with which hyperparameters, on which hardware. "Did that new model
really beat the baseline?" requires manual diffing.

**Approach — tracking only**, against ClearML's free hosted tier
(app.clear.ml) so there's no server to stand up.

### Scope

- Add `clearml` to [requirements.txt](requirements.txt).
- Wrap [batch_translate.py](batch_translate.py) `main()` with
  `Task.init(project_name="translation-evals",
  task_name=f"translate/{model}/{dataset_stem}")`.
  - Auto-captures git state, requirements, stdout.
  - Manually log the existing `.meta.json` fields as scalars:
    `req_per_s`, `elapsed_seconds`, `n_errors`, `n_rows`,
    `concurrency`, `num_shots`.
  - Upload the output `*_translations_*.csv` as an artifact.
- Wrap `score_file()` in [score_translations.py](score_translations.py)
  with per-file `Task.init(..., task_name=f"score/{stem}")`:
  - Scalars: `mean_ifeval`, `mean_comet_qe`,
    `median_comet_qe`, `p10/p90_comet_qe`, `joint_pass_rate` (re-using
    the computation in the repo-root `summarize_results.py`).
  - Histograms: COMET-QE distribution per model × dataset (via
    `task.get_logger().report_histogram`).
  - Artifact: `*_scored.csv`.
- Wrap [run_all_evals.py](run_all_evals.py) top-level with a parent
  `Task.init` that records `MODELS`, `VLLM_EXTRA_ARGS`, `OUTPUT_DIR`
  as run-wide hyperparameters. Link child tasks via `parent=...` so a
  teammate can filter on the run.
- Support `CLEARML_OFFLINE=1` (or `CLEARML_NO_DEFAULT_SERVER=1` — pick
  whichever the ClearML SDK exposes) so smoke tests and local
  debugging can skip tracking without code changes.

### Explicitly out of scope

- `PipelineController` DAG / remote execution with Agents — the value
  here is answering "what did we run last week," not auto-dispatch.
  Revisit once tracking proves its worth.
- Multi-GPU parallel model execution — still one model at a time.

### Acceptance criteria

1. A smoke eval produces a task under the `translation-evals` project
   in the ClearML UI with hyperparameters, scalars, and CSV artifacts
   attached.
2. `CLEARML_OFFLINE=1 python run_all_evals.py` runs identically to
   today, with no network calls to ClearML.
3. Two teammates on different boxes run the same smoke eval; both
   tasks show up in the same project, comparable side-by-side in the
   UI.

---

## IFEval escalation path

If, after deploying Milestone 1, `deberta-v3-large` still misclassifies
a meaningful slice of outputs on our labeled set (< ~5 F1 points over
small, or clearly-polluted cases still get through), escalate to an
LLM-as-judge.

### Plan

- Serve `Qwen3-1.7B` (start small; bump to 4B if calibration is poor)
  via a sidecar vLLM container in the scorer stack.
- Judge prompt (draft):
  ```
  You are grading whether a machine translation output follows the
  instruction "Output only the translation, nothing else." Return
  exactly one token: 1 if the output contains only the translation
  (trailing whitespace is fine), 0 if it contains any preamble,
  postamble, explanation, or conversational filler.

  Translation output:
  <<<HYP>>>
  ```
- Replace `score_ifeval()` in
  [score_translations.py](score_translations.py) with an OpenAI-client
  call to the judge.
- **Don't skip calibration**: re-run `validate_ifeval.py` extended to
  include the LLM judge alongside the two DeBERTa models on the same
  labeled set. Accept the swap only if judge F1 clearly beats
  deberta-large.

### Why not jump straight here

Prompt engineering and calibration are real work. Deberta-large is a
5-minute drop-in; doing it first gives us a solid baseline to compare
any LLM judge against, and may just be enough.

### Acceptance criteria

- Judge F1 > deberta-large F1 by ≥5 absolute points on the labeled set.
- Scorer throughput stays above 100 rows/sec end-to-end (xCOMET +
  judge combined) — if not, bump the judge's vLLM concurrency or drop
  to a smaller judge model.
- The scorer compose stack can still be brought up with a single
  command; no manual coordination between the judge and xCOMET
  containers.
