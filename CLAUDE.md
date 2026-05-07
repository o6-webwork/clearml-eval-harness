# clearml-eval-harness — CLAUDE checkpoint

End-to-end translation eval harness for a Tailscale GPU mesh.
Standalone repo — not a subdirectory of slm-forge.

## What this repo does

1. For each model, stand up vLLM, translate every eval JSONL,
   tear down (`run_all_evals.py` or `clearml_pipeline.py`).
2. Score the translations with IFEval, xCOMET-QE, and xCOMET-ref
   (`score_translations.py` — runs inside the scorer container).
3. ClearML tracks runs, datasets, models, and dispatches steps to GPU
   workers via named queues. The controller runs on `default`.

Pipeline: `translate(N) → {score_ifeval, score_xcomet_qe, score_xcomet_ref}` fan-out.

## Layout

```
/
├── README.md                    ← full setup guide (host + worker + Windows)
├── README_orchestration.md      ← direct vLLM/Docker usage without ClearML
├── ROADMAP.md
├── clearml_pipeline.py          ← ClearML pipeline controller
├── clearml_agent_setup.sh       ← bootstrap agents (--mode 1x/2x/4x)
├── clearml_register_datasets.py ← register eval JSONLs as ClearML Datasets
├── clearml_sync_data.sh         ← rsync eval data + HF cache from canonical host
├── clearml_requirements.txt     ← installed into per-task agent venvs
├── run_all_evals.py             ← non-ClearML orchestrator (direct docker)
├── batch_translate.py           ← vLLM translation worker
├── score_translations.py        ← xCOMET + IFEval scorer (runs in container)
├── docker-compose-vllm.yml      ← vLLM serving stack
├── docker-compose-scorer.yml    ← scoring stack
├── Dockerfile.scorer
├── conf/
│   ├── clearml.conf.template    ← copy to ~/clearml.conf and fill in
│   ├── datasets.yml             ← stem → description for dataset registration
│   └── models.yml               ← HF model ID → queue + tensor_parallel_size
├── clearml-server/
│   └── docker-compose.yml       ← self-hosted ClearML server
├── data/                        ← eval JSONLs (gitignored) + lexicon tools
│   ├── README.md                ← lists required files and how to get them
│   ├── build_kamus_alay.py
│   └── kamus_alay.json
└── hf_cache/                    ← gitignored model weights
```

## Hard constraints

- **Two-queue split is load-bearing.** Pipeline controller goes on
  `default`; GPU steps on `gpu` (or `gpu-1x/2x/4x`). A single agent
  on one queue deadlocks (controller blocks waiting for its own child step).
- **`raise` on step failure, don't return success=False.** ClearML marks
  a function step that returns normally as COMPLETED regardless of dict
  contents.
- **Step functions must be self-contained.** ClearML serialises them;
  all imports live inside the function body.
- **Path resolution.** Steps run with `orch_dir` passed as a kwarg.
  Use it to anchor all file paths — agents land in arbitrary CWDs.
- **Prefer `Dataset.get(...).get_local_copy()` over raw file paths.**
  The `EVAL_PATHS` fallback exists only for direct (non-ClearML) runs.

## Common ops

```bash
# Smoke test (local, 10 rows)
MODELS=tencent/HY-MT1.5-1.8B-FP8 python clearml_pipeline.py --local --max-examples 10

# Full distributed run
python clearml_pipeline.py

# Score-only against historical translations
python clearml_pipeline.py --local --translation-dataset <id> --metrics qe,ref

# Score pre-translated local CSVs
python clearml_pipeline.py --local --translations-dir results_overnight --metrics qe,ref

# Register / refresh datasets
python clearml_register_datasets.py
```

## Pointers

- Full setup guide (host + worker + Windows): [README.md](README.md)
- Direct vLLM/Docker usage (no ClearML): [README_orchestration.md](README_orchestration.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)
- Eval data directory: [data/README.md](data/README.md)
