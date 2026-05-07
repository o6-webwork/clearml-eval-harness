"""
Orchestrator
============
Runs batch_translate.py against every model cached under hf_cache/hub,
swapping the vLLM container between models, then scores the resulting CSVs
with the scorer container.

For each model:
  1. MODEL=<id> docker compose -f docker-compose-vllm.yml up -d
  2. Poll /v1/models until ready
  3. python batch_translate.py    (stdout/stderr tee'd to a per-model log)
  4. docker compose -f docker-compose-vllm.yml down

Finally:
  docker compose -f docker-compose-scorer.yml run --rm scorer

A failure on one model is logged and the run continues.

Usage (from inside this folder):
  python run_all_evals.py
  MODELS=google/gemma-4-E2B-it,tencent/HY-MT1.5-1.8B-FP8 python run_all_evals.py
  OUTPUT_DIR=results_overnight python run_all_evals.py
  MAX_EXAMPLES=10 OUTPUT_DIR=results_smoke python run_all_evals.py
  SKIP_SCORING=1 python run_all_evals.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
HF_HUB_DIR = os.environ.get("HF_HUB_DIR") or os.path.join(HERE, "hf_cache", "hub")
COMPOSE_FILE = os.environ.get("COMPOSE_FILE", "docker-compose-vllm.yml")
SCORER_COMPOSE = os.environ.get("SCORER_COMPOSE", "docker-compose-scorer.yml")
VLLM_HOST = os.environ.get("VLLM_HOST", "http://localhost:8000")
# Output folder for translations + logs + scored CSVs. Overridable so the
# overnight run doesn't clobber the smoke test.
OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(HERE, "results")
# Passed through to batch_translate.py. "" = full run.
MAX_EXAMPLES = os.environ.get("MAX_EXAMPLES", "")
READY_TIMEOUT = int(os.environ.get("READY_TIMEOUT", "600"))

# Cached HF repos that are NOT translation LLMs (scoring / embedding models,
# etc.). The scorer container downloads these into the shared hf_cache, so
# without filtering they'd get picked up as models to serve via vLLM.
SCORER_MODEL_IDS = {
    "Unbabel/XCOMET-XL",
    "Unbabel/wmt22-cometkiwi-da",
    "cross-encoder/nli-deberta-v3-small",
    "sentence-transformers/LaBSE",
    "sentence-transformers/all-MiniLM-L6-v2",
    # Encoder-only models — not generative, can't be served by vLLM as an LLM.
    "facebook/xlm-roberta-xl",
}
SCORER_ORG_PREFIXES = ("sentence-transformers/", "cross-encoder/", "Unbabel/")


def discover_models() -> list[str]:
    """Return HF model ids cached under the shared hf_cache as '{org}/{name}'."""
    override = os.environ.get("MODELS")
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]

    if not os.path.isdir(HF_HUB_DIR):
        raise SystemExit(f"HF cache not found: {HF_HUB_DIR}")

    models, skipped = [], []
    for entry in sorted(os.listdir(HF_HUB_DIR)):
        if not entry.startswith("models--"):
            continue
        parts = entry.split("--", 2)
        if len(parts) != 3:
            continue
        _, org, name = parts
        model_id = f"{org}/{name}"
        if model_id in SCORER_MODEL_IDS or model_id.startswith(SCORER_ORG_PREFIXES):
            skipped.append(model_id)
            continue
        models.append(model_id)
    if skipped:
        print(f"Skipping {len(skipped)} scoring/embedding model(s): {skipped}")
    return models


def compose(*args: str, env: dict | None = None) -> int:
    cmd = ["docker", "compose", "-f", COMPOSE_FILE, *args]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=HERE, env=env)


def wait_for_ready(expected_model: str, timeout: int) -> bool:
    """Poll vLLM /v1/models until it reports the expected model."""
    print(f"  Waiting for vLLM to serve {expected_model} (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{VLLM_HOST}/v1/models", timeout=5)
            if resp.status_code == 200:
                ids = [m["id"] for m in resp.json().get("data", [])]
                if expected_model in ids:
                    print(f"  vLLM ready ({int(time.time() - start)}s).")
                    return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(5)
    return False


def run_batch_translate(model: str) -> int:
    """Run batch_translate.py, tee-ing output to a per-model log file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = model.replace("/", "_").replace(":", "_")
    log_path = os.path.join(OUTPUT_DIR, f"{safe}_run.log")
    print(f"  Log → {log_path}")

    child_env = {
        **os.environ,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MAX_EXAMPLES": MAX_EXAMPLES,
    }

    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "batch_translate.py"],
            cwd=HERE,
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
        return proc.returncode


def run_one(model: str) -> bool:
    print("\n" + "#" * 70)
    print(f"# MODEL: {model}")
    print(f"# START: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print("#" * 70)

    env = {**os.environ, "MODEL": model}

    try:
        if compose("up", "-d", env=env) != 0:
            print(f"  FAILED: docker compose up for {model}")
            return False

        if not wait_for_ready(model, READY_TIMEOUT):
            print(f"  FAILED: vLLM never became ready for {model}")
            return False

        rc = run_batch_translate(model)
        if rc != 0:
            print(f"  batch_translate.py exited with code {rc}")
            return False
        return True
    finally:
        compose("down", env=env)


def main() -> None:
    models = discover_models()
    if not models:
        raise SystemExit("No models discovered.")

    print(f"Orchestrator — {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"Compose file : {COMPOSE_FILE}")
    print(f"HF cache     : {HF_HUB_DIR}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Max examples : {MAX_EXAMPLES or 'all'}  (per file)")
    print(f"Models ({len(models)}):")
    for m in models:
        print(f"  - {m}")

    succeeded, failed = [], []
    for model in models:
        (succeeded if run_one(model) else failed).append(model)

    print("\n" + "=" * 70)
    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed models:")
        for m in failed:
            print(f"  - {m}")

    if os.environ.get("SKIP_SCORING") == "1":
        print("\nSKIP_SCORING=1 — not running scorer.")
        return

    print("\nRunning scorer container over all output CSVs...")
    # The scorer compose bind-mounts this folder at /app, so OUTPUT_DIR must
    # resolve inside /app. Translate absolute host paths to relative when
    # they're under this folder; otherwise pass through.
    scorer_output = OUTPUT_DIR
    if os.path.isabs(OUTPUT_DIR):
        try:
            scorer_output = os.path.relpath(OUTPUT_DIR, HERE)
        except ValueError:
            pass
    scorer_cmd = [
        "docker", "compose",
        "-f", SCORER_COMPOSE,
        "run", "--rm", "scorer",
    ]
    print(f"  $ OUTPUT_DIR={scorer_output} {' '.join(scorer_cmd)}")
    subprocess.call(
        scorer_cmd,
        cwd=HERE,
        env={**os.environ, "OUTPUT_DIR": scorer_output},
    )


if __name__ == "__main__":
    main()
