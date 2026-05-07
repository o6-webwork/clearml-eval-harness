"""
ClearML Pipeline Controller — Distributed run_all_evals.
=========================================================

Replaces the sequential model loop in run_all_evals.py with a ClearML
pipeline that fans out translate tasks across GPU workers, then fans out
three independent scoring branches in parallel.

    ┌────────────┐  ┌────────────┐       ┌────────────┐
    │ translate   │  │ translate   │  ...  │ translate   │
    │ model-1     │  │ model-2     │       │ model-N     │
    └─────┬──────┘  └─────┬──────┘       └─────┬──────┘
          │               │                     │
          └───────────────┴──────────┬──────────┘
                       ┌─────────────┼─────────────┐
                       ▼             ▼              ▼
               score_ifeval  score_xcomet_qe  score_xcomet_ref
               (NLI clean)   (ref-free)       (ref-based)

Uses add_function_step so pipeline works out of the box — no pre-registered
template tasks needed.

Usage:
  cd orchestration

  # Smoke test — runs locally on this machine, shows in ClearML UI
  MODELS=tencent/HY-MT1.5-1.8B-FP8,NiuTrans/LMT-60-4B \
    python clearml_pipeline.py --local --max-examples 10

  # Full local run (sequential, tracked in ClearML)
  python clearml_pipeline.py --local

  # Distributed — fans out to ClearML agent workers
  python clearml_pipeline.py

  # Only run specific scoring metrics
  python clearml_pipeline.py --local --metrics qe,ref

  # Override model list, custom output dir
  MODELS=google/gemma-4-E2B-it,NiuTrans/LMT-60-4B \
    python clearml_pipeline.py --output-dir results_overnight

  # Target a specific queue
  python clearml_pipeline.py --queue gpu-24gb
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Re-use the model discovery logic from run_all_evals.py.
sys.path.insert(0, HERE)
from run_all_evals import discover_models


# ---------------------------------------------------------------------------
# Step functions — must be self-contained (all imports inside) because
# ClearML serializes them and runs them in isolated task environments.
# ---------------------------------------------------------------------------

def _step_translate(model: str, compose_file: str, vllm_host: str,
                    output_dir: str, max_examples: str,
                    vllm_extra_args: str, ready_timeout: int,
                    orch_dir: str = "",
                    tensor_parallel_size: int = 1) -> dict:
    """Translate one model: start vLLM, run batch_translate.py, stop vLLM.
    Logs metrics and artifacts to the ClearML task created by the pipeline."""
    import json
    import os
    import subprocess
    import sys
    import time
    import requests

    if not orch_dir:
        orch_dir = os.environ.get("ORCH_DIR", os.getcwd())

    # Auto-download model weights if not already in the HF cache.
    # Runs before vLLM starts so vLLM always loads from cache, never from the
    # network mid-run. Safe to call on already-cached models (no-op).
    hf_hub_cache = os.path.join(orch_dir, "hf_cache", "hub")
    model_cache_dir = os.path.join(hf_hub_cache, "models--" + model.replace("/", "--"))
    if not os.path.isdir(model_cache_dir):
        print(f"  Model not in cache — downloading {model}...")
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN") or None
        snapshot_download(repo_id=model, cache_dir=hf_hub_cache, token=token)
        print(f"  Download complete.")
    else:
        print(f"  Model already cached: {model_cache_dir}")

    # Append tensor-parallel flag when more than one GPU is assigned.
    if tensor_parallel_size > 1:
        tp_flag = f"--tensor-parallel-size {tensor_parallel_size}"
        vllm_extra_args = f"{vllm_extra_args} {tp_flag}".strip() if vllm_extra_args else tp_flag

    # Resolve output_dir relative to orch_dir if not absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(orch_dir, output_dir)

    # Get the ClearML task for this step (created by PipelineController)
    from clearml import Task
    task = Task.current_task()
    logger = task.get_logger() if task else None

    def compose(*args, env=None):
        cmd = ["docker", "compose", "-f", compose_file, *args]
        print(f"  $ {' '.join(cmd)}")
        return subprocess.call(cmd, cwd=orch_dir, env=env)

    env = {**os.environ, "MODEL": model, "VLLM_EXTRA_ARGS": vllm_extra_args}

    try:
        # 1. Start vLLM
        if compose("up", "-d", env=env) != 0:
            raise RuntimeError(f"docker compose up failed for {model}")

        # 2. Wait for ready
        print(f"  Waiting for vLLM to serve {model} (timeout {ready_timeout}s)...")
        start = time.time()
        ready = False
        while time.time() - start < ready_timeout:
            try:
                resp = requests.get(f"{vllm_host}/v1/models", timeout=5)
                if resp.status_code == 200:
                    ids = [m["id"] for m in resp.json().get("data", [])]
                    if model in ids:
                        print(f"  vLLM ready ({int(time.time() - start)}s).")
                        ready = True
                        break
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(5)

        if not ready:
            raise RuntimeError(f"vLLM never became ready for {model}")

        # 3. Run batch_translate.py
        child_env = {
            **os.environ,
            "VLLM_HOST": vllm_host,
            "OUTPUT_DIR": output_dir,
            "MAX_EXAMPLES": max_examples,
        }
        os.makedirs(output_dir, exist_ok=True)
        safe = model.replace("/", "_").replace(":", "_")
        log_path = os.path.join(output_dir, f"{safe}_run.log")

        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "batch_translate.py"],
                cwd=orch_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                env=child_env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"batch_translate.py exited {proc.returncode}")

        # 4. Collect metadata and log to ClearML
        meta = {}
        for f in os.listdir(output_dir):
            if f.startswith(safe) and f.endswith(".meta.json"):
                with open(os.path.join(output_dir, f), encoding="utf-8") as fh:
                    meta[f] = json.load(fh)

        # Pin the HF revision the worker actually loaded as a hyperparameter
        # — same model name can map to different weights as the maintainer
        # pushes updates, so without this the run history is ambiguous.
        revisions = {m.get("model_revision") for m in meta.values()
                     if m.get("model_revision")}
        rev_value = None
        if task and revisions:
            # All meta files for one model should agree; flag if they don't.
            rev_value = next(iter(revisions)) if len(revisions) == 1 \
                else ",".join(sorted(revisions))
            task.connect({"model_revision": rev_value}, name="model")

        if logger:
            for meta_name, m in meta.items():
                dataset = meta_name.replace(f"{safe}_translations_", "").replace(".csv.meta.json", "")
                # Scalars — one series per dataset
                logger.report_scalar("throughput_req_s", dataset,
                                     iteration=0, value=m.get("req_per_s", 0))
                logger.report_scalar("elapsed_seconds", dataset,
                                     iteration=0, value=m.get("elapsed_seconds", 0))
                logger.report_scalar("error_count", dataset,
                                     iteration=0, value=m.get("n_errors", 0))
                logger.report_scalar("row_count", dataset,
                                     iteration=0, value=m.get("n_rows", 0))

            # Upload translation CSVs as artifacts
            for f in os.listdir(output_dir):
                if f.startswith(safe) and f.endswith(".csv") and not f.endswith(".meta.json"):
                    task.upload_artifact(f.replace(".csv", ""), os.path.join(output_dir, f))

            # Upload run log
            if os.path.isfile(log_path):
                task.upload_artifact("run_log", log_path)

        # Publish translation CSVs + meta as a ClearML Dataset so scoring steps
        # can be re-run against historical translations without re-translating.
        dataset_id = None
        try:
            from clearml import Dataset as ClearMLDataset
            run_id = task.id if task else "local"
            ds = ClearMLDataset.create(
                dataset_project="translation-evals",
                dataset_name=f"translations/{safe}/{run_id}",
            )
            for f in os.listdir(output_dir):
                if f.startswith(safe) and (f.endswith(".csv") or f.endswith(".meta.json")):
                    ds.add_files(os.path.join(output_dir, f))
            ds.finalize(auto_upload=True)
            dataset_id = ds.id
            print(f"  Published translation Dataset: {dataset_id}")
        except Exception as e:
            print(f"  Warning: Dataset publish failed (continuing): {e}")

        return {"model": model, "meta": meta, "dataset_id": dataset_id}

    finally:
        compose("down", env=env)


def _step_score_ifeval(output_dir: str, scorer_compose: str,
                       orch_dir: str = "",
                       translation_dataset_id: str = "",
                       translations_dir: str = "") -> dict:
    """Score translations with IFEval only (NLI clean/polluted classifier).
    Writes scored CSVs to output_dir/ifeval/."""
    import glob
    import os
    import subprocess

    if not orch_dir:
        orch_dir = os.environ.get("ORCH_DIR", os.getcwd())
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(orch_dir, output_dir)
    if translations_dir and not os.path.isabs(translations_dir):
        translations_dir = os.path.join(orch_dir, translations_dir)

    from clearml import Task
    task = Task.current_task()
    logger = task.get_logger() if task else None

    # When scoring against historical translations, fetch the Dataset and copy
    # CSVs into output_dir so the scorer container (which mounts output_dir) sees them.
    if translation_dataset_id:
        import shutil
        from clearml import Dataset as ClearMLDataset
        os.makedirs(output_dir, exist_ok=True)
        ds = ClearMLDataset.get(dataset_id=translation_dataset_id)
        ds_path = ds.get_local_copy()
        copied = sum(
            1 for fname in os.listdir(ds_path)
            if "_translations" in fname and fname.endswith(".csv")
            and not shutil.copy2(os.path.join(ds_path, fname), output_dir)
        )
        print(f"  Fetched {copied} translation CSV(s) from Dataset {translation_dataset_id}")

    # CSV source: explicit local dir > output_dir (populated by translate step or dataset fetch)
    csv_source_dir = translations_dir if translations_dir else output_dir

    metric_dir = os.path.join(output_dir, "ifeval")
    os.makedirs(metric_dir, exist_ok=True)

    try:
        rel_output = os.path.relpath(metric_dir, orch_dir)
        rel_glob = os.path.relpath(
            os.path.join(csv_source_dir, "*_translations*.csv"), orch_dir
        )
    except ValueError:
        rel_output = metric_dir
        rel_glob = os.path.join(csv_source_dir, "*_translations*.csv")

    env = {
        **os.environ,
        "SCORE_METRICS": "ifeval",
        "OUTPUT_DIR": rel_output,
        "INPUT_GLOB": rel_glob,
    }
    cmd = ["docker", "compose", "-f", scorer_compose, "run", "--rm", "scorer"]
    print(f"  $ SCORE_METRICS=ifeval OUTPUT_DIR={rel_output} {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=orch_dir, env=env)
    if rc != 0:
        raise RuntimeError(f"IFEval scorer exited with code {rc}")

    try:
        import pandas as pd

        scored_files = sorted(glob.glob(os.path.join(metric_dir, "*_scored.csv")))
        summary_rows = []
        for sf in scored_files:
            df = pd.read_csv(sf)
            stem = os.path.basename(sf).replace("_scored.csv", "")
            model_name = df["model"].iloc[0] if "model" in df.columns else stem
            if "ifeval" in df.columns:
                mean_if = df["ifeval"].mean()
                if logger:
                    logger.report_scalar("IFEval (clean %)", stem,
                                         iteration=0, value=round(mean_if * 100, 1))
                summary_rows.append({
                    "stem": stem, "model": model_name, "n_rows": len(df),
                    "mean_ifeval": round(mean_if, 4),
                    "ifeval_clean_pct": round(mean_if * 100, 1),
                })
            if task:
                task.upload_artifact(stem + "_ifeval", sf)

        if logger and summary_rows:
            logger.report_table(
                title="IFEval summary",
                series="all models × datasets",
                iteration=0,
                table_plot=pd.DataFrame(summary_rows),
            )
    except Exception as e:
        print(f"  Warning: IFEval post-analysis failed: {e}")
        import traceback
        traceback.print_exc()

    return {"metric": "ifeval", "metric_dir": metric_dir}


def _step_score_xcomet_qe(output_dir: str, scorer_compose: str,
                           orch_dir: str = "",
                           translation_dataset_id: str = "",
                           translations_dir: str = "") -> dict:
    """Score translations with xCOMET-QE only (reference-free).
    Writes scored CSVs to output_dir/qe/."""
    import glob
    import os
    import subprocess

    if not orch_dir:
        orch_dir = os.environ.get("ORCH_DIR", os.getcwd())
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(orch_dir, output_dir)
    if translations_dir and not os.path.isabs(translations_dir):
        translations_dir = os.path.join(orch_dir, translations_dir)

    from clearml import Task
    task = Task.current_task()
    logger = task.get_logger() if task else None

    if translation_dataset_id:
        import shutil
        from clearml import Dataset as ClearMLDataset
        os.makedirs(output_dir, exist_ok=True)
        ds = ClearMLDataset.get(dataset_id=translation_dataset_id)
        ds_path = ds.get_local_copy()
        copied = sum(
            1 for fname in os.listdir(ds_path)
            if "_translations" in fname and fname.endswith(".csv")
            and not shutil.copy2(os.path.join(ds_path, fname), output_dir)
        )
        print(f"  Fetched {copied} translation CSV(s) from Dataset {translation_dataset_id}")

    csv_source_dir = translations_dir if translations_dir else output_dir

    metric_dir = os.path.join(output_dir, "qe")
    os.makedirs(metric_dir, exist_ok=True)

    try:
        rel_output = os.path.relpath(metric_dir, orch_dir)
        rel_glob = os.path.relpath(
            os.path.join(csv_source_dir, "*_translations*.csv"), orch_dir
        )
    except ValueError:
        rel_output = metric_dir
        rel_glob = os.path.join(csv_source_dir, "*_translations*.csv")

    env = {
        **os.environ,
        "SCORE_METRICS": "qe",
        "OUTPUT_DIR": rel_output,
        "INPUT_GLOB": rel_glob,
    }
    cmd = ["docker", "compose", "-f", scorer_compose, "run", "--rm", "scorer"]
    print(f"  $ SCORE_METRICS=qe OUTPUT_DIR={rel_output} {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=orch_dir, env=env)
    if rc != 0:
        raise RuntimeError(f"xCOMET-QE scorer exited with code {rc}")

    try:
        import numpy as np
        import pandas as pd

        scored_files = sorted(glob.glob(os.path.join(metric_dir, "*_scored.csv")))
        summary_rows = []
        for sf in scored_files:
            df = pd.read_csv(sf)
            stem = os.path.basename(sf).replace("_scored.csv", "")
            model_name = df["model"].iloc[0] if "model" in df.columns else stem

            if "comet_qe" in df.columns:
                vals = df["comet_qe"].dropna()
                row = {
                    "stem": stem, "model": model_name, "n_rows": len(df),
                    "mean_comet_qe": round(vals.mean(), 4),
                    "median_comet_qe": round(vals.median(), 4),
                    "p10_comet_qe": round(np.percentile(vals, 10), 4),
                    "p90_comet_qe": round(np.percentile(vals, 90), 4),
                }
                summary_rows.append(row)

                if logger:
                    logger.report_scalar("xCOMET-QE mean", stem,
                                         iteration=0, value=row["mean_comet_qe"])
                    logger.report_scalar("xCOMET-QE median", stem,
                                         iteration=0, value=row["median_comet_qe"])
                    logger.report_scalar("xCOMET-QE p10", stem,
                                         iteration=0, value=row["p10_comet_qe"])
                    logger.report_histogram(
                        title="xCOMET-QE distribution", series=stem,
                        values=vals.tolist(), iteration=0,
                        xaxis="Score", yaxis="Count",
                    )
                    if "lang_codes" in df.columns:
                        for lang, score in df.groupby("lang_codes")["comet_qe"].mean().items():
                            logger.report_scalar(
                                f"xCOMET-QE by language ({stem})", str(lang),
                                iteration=0, value=round(score, 4),
                            )
                    if "source" in df.columns:
                        cols = [c for c in ["model", "lang_codes", "source",
                                            "hypothesis", "comet_qe"] if c in df.columns]
                        worst = df.nsmallest(min(5, len(df)), "comet_qe")[cols]
                        logger.report_table(
                            title=f"Worst translations ({stem})",
                            series="lowest xCOMET-QE",
                            iteration=0,
                            table_plot=worst,
                        )

            if task:
                task.upload_artifact(stem + "_qe", sf)

        if logger and summary_rows:
            logger.report_table(
                title="xCOMET-QE summary",
                series="all models × datasets",
                iteration=0,
                table_plot=pd.DataFrame(summary_rows),
            )
    except Exception as e:
        print(f"  Warning: xCOMET-QE post-analysis failed: {e}")
        import traceback
        traceback.print_exc()

    return {"metric": "qe", "metric_dir": metric_dir}


def _step_score_xcomet_ref(output_dir: str, scorer_compose: str,
                            orch_dir: str = "",
                            translation_dataset_id: str = "",
                            translations_dir: str = "") -> dict:
    """Score translations with xCOMET-ref only (reference-based, src+mt+ref).
    Writes scored CSVs to output_dir/ref/."""
    import glob
    import os
    import subprocess

    if not orch_dir:
        orch_dir = os.environ.get("ORCH_DIR", os.getcwd())
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(orch_dir, output_dir)
    if translations_dir and not os.path.isabs(translations_dir):
        translations_dir = os.path.join(orch_dir, translations_dir)

    from clearml import Task
    task = Task.current_task()
    logger = task.get_logger() if task else None

    if translation_dataset_id:
        import shutil
        from clearml import Dataset as ClearMLDataset
        os.makedirs(output_dir, exist_ok=True)
        ds = ClearMLDataset.get(dataset_id=translation_dataset_id)
        ds_path = ds.get_local_copy()
        copied = sum(
            1 for fname in os.listdir(ds_path)
            if "_translations" in fname and fname.endswith(".csv")
            and not shutil.copy2(os.path.join(ds_path, fname), output_dir)
        )
        print(f"  Fetched {copied} translation CSV(s) from Dataset {translation_dataset_id}")

    csv_source_dir = translations_dir if translations_dir else output_dir

    metric_dir = os.path.join(output_dir, "ref")
    os.makedirs(metric_dir, exist_ok=True)

    try:
        rel_output = os.path.relpath(metric_dir, orch_dir)
        rel_glob = os.path.relpath(
            os.path.join(csv_source_dir, "*_translations*.csv"), orch_dir
        )
    except ValueError:
        rel_output = metric_dir
        rel_glob = os.path.join(csv_source_dir, "*_translations*.csv")

    env = {
        **os.environ,
        "SCORE_METRICS": "ref",
        "OUTPUT_DIR": rel_output,
        "INPUT_GLOB": rel_glob,
    }
    cmd = ["docker", "compose", "-f", scorer_compose, "run", "--rm", "scorer"]
    print(f"  $ SCORE_METRICS=ref OUTPUT_DIR={rel_output} {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=orch_dir, env=env)
    if rc != 0:
        raise RuntimeError(f"xCOMET-ref scorer exited with code {rc}")

    try:
        import pandas as pd

        scored_files = sorted(glob.glob(os.path.join(metric_dir, "*_scored.csv")))
        summary_rows = []
        for sf in scored_files:
            df = pd.read_csv(sf)
            stem = os.path.basename(sf).replace("_scored.csv", "")
            model_name = df["model"].iloc[0] if "model" in df.columns else stem
            if "comet" in df.columns:
                mean_ref = df["comet"].mean()
                if logger:
                    logger.report_scalar("xCOMET-ref mean", stem,
                                         iteration=0, value=round(mean_ref, 4))
                summary_rows.append({
                    "stem": stem, "model": model_name, "n_rows": len(df),
                    "mean_comet_ref": round(mean_ref, 4),
                })
            if task:
                task.upload_artifact(stem + "_ref", sf)

        if logger and summary_rows:
            logger.report_table(
                title="xCOMET-ref summary",
                series="all models × datasets",
                iteration=0,
                table_plot=pd.DataFrame(summary_rows),
            )
    except Exception as e:
        print(f"  Warning: xCOMET-ref post-analysis failed: {e}")
        import traceback
        traceback.print_exc()

    return {"metric": "ref", "metric_dir": metric_dir}


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

_METRIC_STEPS = {
    "ifeval": (_step_score_ifeval, "score_ifeval"),
    "qe": (_step_score_xcomet_qe, "score_xcomet_qe"),
    "ref": (_step_score_xcomet_ref, "score_xcomet_ref"),
}


def build_and_run(
    models: list[str],
    output_dir: str,
    queue: str,
    max_examples: str,
    compose_file: str,
    metrics: set[str],
    vllm_extra_args: str,
    local: bool,
    translation_dataset_id: str = "",
    translations_dir: str = "",
) -> None:
    import yaml
    from clearml.automation import PipelineController

    # Load per-model queue/tp configuration.  Falls back gracefully when the
    # file is absent (e.g. first run, or the user hasn't edited it yet).
    models_conf_path = os.path.join(HERE, "conf", "models.yml")
    models_conf: dict = {}
    try:
        with open(models_conf_path, encoding="utf-8") as fh:
            models_conf = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        pass

    pipe = PipelineController(
        name="translation-eval-run",
        project="translation-evals/pipelines",
        version="1.0",
        add_pipeline_tags=True,
    )

    # Set ORCH_DIR so step functions know where batch_translate.py lives
    pipe.add_parameter("orch_dir", HERE)
    # Persist CLI args that aren't otherwise recoverable on remote re-execution
    # (argparse sees its defaults, not the original flags, when run by the agent).
    pipe.add_parameter("models_csv", ",".join(models))
    pipe.add_parameter("metrics_csv", ",".join(sorted(metrics)))

    # --- Translate steps (one per model) ---
    # Skipped entirely when --translation-dataset is supplied (score-only mode).
    # In local mode, chain sequentially (one GPU, one vLLM container at a time).
    # In distributed mode, steps have no parents so workers run them in parallel.
    translate_step_names = []
    if not translation_dataset_id and not translations_dir:
        for i, model in enumerate(models):
            safe = model.replace("/", "_").replace(":", "_")
            step_name = f"translate_{safe}"

            # Per-model queue, tensor-parallel size, and extra vLLM args from
            # conf/models.yml; fall back to CLI values when model is not listed.
            # Model-level vllm_extra_args are appended after the CLI flag so they
            # win on duplicate flags (e.g. --quantization).
            model_cfg = models_conf.get(model, {})
            step_queue = model_cfg.get("queue", queue) if not local else None
            tp_size = int(model_cfg.get("tensor_parallel_size", 1))
            model_extra = model_cfg.get("vllm_extra_args", "")
            merged_extra = " ".join(
                part for part in (vllm_extra_args, model_extra) if part
            )

            parents = None
            if local and i > 0:
                parents = [translate_step_names[-1]]

            translate_step_names.append(step_name)

            pipe.add_function_step(
                name=step_name,
                function=_step_translate,
                function_kwargs={
                    "model": model,
                    "compose_file": compose_file,
                    "vllm_host": "http://localhost:8000",
                    "output_dir": output_dir,
                    "max_examples": max_examples,
                    "vllm_extra_args": merged_extra,
                    "ready_timeout": 600,
                    # Local: anchor to this machine's repo root.
                    # Distributed: pass "" so the step resolves ORCH_DIR from
                    # the worker's environment (set when starting clearml-agent).
                    "orch_dir": HERE if local else "",
                    "tensor_parallel_size": tp_size,
                },
                parents=parents,
                project_name="translation-evals",
                execution_queue=step_queue,
            )

    # --- Scoring steps — three independent branches, all fanning out from translate ---
    # Each branch writes to its own subdirectory (output_dir/ifeval/, /qe/, /ref/)
    # so parallel execution on distributed workers doesn't race on the same files.
    # In local mode the steps are chained sequentially to avoid GPU OOM and the
    # Docker network creation race that occurs when all three containers start at once.
    prev_score_step: str | None = None
    for metric in ("ifeval", "qe", "ref"):
        if metric not in metrics:
            continue
        fn, step_name = _METRIC_STEPS[metric]
        if local and prev_score_step is not None:
            step_parents = [prev_score_step]
        else:
            step_parents = translate_step_names or None
        pipe.add_function_step(
            name=step_name,
            function=fn,
            function_kwargs={
                "output_dir": output_dir,
                "scorer_compose": "docker-compose-scorer.yml",
                "orch_dir": HERE if local else "",
                "translation_dataset_id": translation_dataset_id,
                "translations_dir": translations_dir,
            },
            parents=step_parents,
            project_name="translation-evals",
            execution_queue=None if local else queue,
        )
        if local:
            prev_score_step = step_name

    n_models = len(models)
    if translation_dataset_id:
        prefix = f"score-only from Dataset {translation_dataset_id}"
    elif translations_dir:
        prefix = f"score-only from local dir '{translations_dir}'"
    else:
        prefix = f"{n_models} model(s)"
    if local:
        print(f"Starting pipeline LOCALLY — {prefix}, metrics={sorted(metrics)}.")
        print("Each step creates a ClearML Task visible in the UI.\n")
        pipe.start_locally(run_pipeline_steps_locally=True)
    else:
        print(f"Starting pipeline — {prefix} on queue '{queue}', metrics={sorted(metrics)}.")
        print("Monitor in the ClearML web UI.\n")
        pipe.start(queue="default")


_VALID_METRICS = {"ifeval", "qe", "ref"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval pipeline via ClearML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--local", action="store_true",
                        help="Run on this machine (still tracked in ClearML UI)")
    parser.add_argument("--queue", default="gpu",
                        help="Fallback ClearML queue for models not listed in "
                             "conf/models.yml (default: gpu)")
    parser.add_argument("--output-dir",
                        default=os.path.join(HERE, "results"),
                        help="Output directory for CSVs and logs")
    parser.add_argument("--max-examples", default="",
                        help="Per-file row cap (blank = all)")
    parser.add_argument("--compose-file", default="docker-compose-vllm.yml")
    parser.add_argument("--metrics", default="ifeval,qe,ref",
                        help="Comma-separated scoring metrics to run "
                             "(default: ifeval,qe,ref). Pass empty string to skip scoring.")
    parser.add_argument("--vllm-extra-args", default="",
                        help="Extra args passed to vLLM (e.g. '--quantization fp8')")
    parser.add_argument("--translation-dataset", default="",
                        metavar="DATASET_ID",
                        help="ClearML Dataset ID of prior translation outputs. "
                             "Skips translate steps and runs scoring only.")
    parser.add_argument("--translations-dir", default="",
                        metavar="DIR",
                        help="Local directory containing pre-translated "
                             "*_translations*.csv files. Skips translate steps "
                             "and scores from this directory. Use with --local.")
    args = parser.parse_args()

    raw_metrics = {m.strip().lower() for m in args.metrics.split(",") if m.strip()}
    unknown = raw_metrics - _VALID_METRICS
    if unknown:
        print(f"Unknown metrics: {sorted(unknown)}. Valid: {sorted(_VALID_METRICS)}")
        sys.exit(1)

    if args.translations_dir and not args.local:
        print("Warning: --translations-dir points at a local path and only works "
              "with --local. Add --local or use --translation-dataset for distributed runs.")
        sys.exit(1)

    # When clearml-agent re-executes the pipeline controller on a remote worker,
    # CLEARML_TASK_ID is set but hf_cache won't exist in the agent's task clone
    # and argparse sees its defaults instead of the original CLI flags.
    # Restore both from the stored pipeline parameters.
    if os.environ.get("CLEARML_TASK_ID"):
        from clearml import Task as _Task
        _task = _Task.current_task()
        _get = lambda k: (_task.get_parameter(f"pipeline/{k}") or "") if _task else ""
        models = [m.strip() for m in _get("models_csv").split(",") if m.strip()]
        _metrics_csv = _get("metrics_csv")
        if _metrics_csv:
            raw_metrics = {m.strip().lower() for m in _metrics_csv.split(",") if m.strip()}
    else:
        models = discover_models()
        if not models and not args.translation_dataset and not args.translations_dir:
            print("No models discovered. Set MODELS env var or populate hf_cache/hub/.")
            sys.exit(1)

    if models:
        import yaml as _yaml
        _models_conf_path = os.path.join(HERE, "conf", "models.yml")
        _models_conf: dict = {}
        try:
            with open(_models_conf_path, encoding="utf-8") as _fh:
                _models_conf = _yaml.safe_load(_fh) or {}
        except FileNotFoundError:
            pass
        print(f"Models ({len(models)}):")
        for m in models:
            cfg = _models_conf.get(m, {})
            q = cfg.get("queue", args.queue)
            tp = int(cfg.get("tensor_parallel_size", 1))
            tag = f"queue={q}, tp={tp}" if not args.local else "local"
            print(f"  - {m}  ({tag})")
        print()

    build_and_run(
        models=models,
        output_dir=args.output_dir,
        queue=args.queue,
        max_examples=args.max_examples,
        compose_file=args.compose_file,
        metrics=raw_metrics,
        vllm_extra_args=args.vllm_extra_args,
        local=args.local,
        translation_dataset_id=args.translation_dataset,
        translations_dir=args.translations_dir,
    )


if __name__ == "__main__":
    main()
