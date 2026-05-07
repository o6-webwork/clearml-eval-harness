"""
Batch Translation (vLLM)
========================
Translates every entry in a JSONL eval file via the vLLM OpenAI-compatible
API and writes a single CSV. Dispatches requests concurrently so vLLM's
continuous batcher stays saturated.

Output columns: model, id1, id2, lang_codes, language, source, hypothesis,
                reference, error

Setup (usually called via run_all_evals.py):
  1. Bring up vLLM (docker-compose-vllm.yml).
  2. source .venv/bin/activate        # or wherever your venv lives
  3. python batch_translate.py

Environment variables:
  VLLM_HOST        default http://localhost:8000
  EVAL_PATHS       comma-separated JSONLs. Default: data/mono_sea.jsonl, data/mono_me.jsonl
  MAX_EXAMPLES     per-file cap. Default: unlimited (blank / "none")
  OUTPUT_DIR       default ./results
  HF_TOKEN         passed to vLLM for gated models
"""

import asyncio
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_HOST = os.environ.get("VLLM_HOST", "http://localhost:8000")

_HERE = os.path.dirname(os.path.abspath(__file__))

# JSONL eval files. Two modes:
#   - EVAL_DATASETS=mono_sea,mono_me  → pull from ClearML Datasets (project
#     "translation-evals"). Each dataset's local copy is globbed for *.jsonl.
#   - EVAL_PATHS=/abs/path/a.jsonl,...  → use local files directly.
# Defaults to local files in the parent directory if neither is set.
EVAL_DATASETS: list[str] = [
    n.strip() for n in os.environ.get("EVAL_DATASETS", "").split(",") if n.strip()
]

EVAL_PATHS: list[str] = [
    p.strip()
    for p in os.environ.get(
        "EVAL_PATHS",
        ",".join([
            os.path.join(_HERE, "data", "mono_sea.jsonl"),
            os.path.join(_HERE, "data", "mono_me.jsonl"),
        ]),
    ).split(",")
    if p.strip()
]

# ClearML project that hosts eval datasets. Matches clearml_register_datasets.py.
EVAL_DATASETS_PROJECT = os.environ.get("EVAL_DATASETS_PROJECT", "translation-evals")

# Override with MAX_EXAMPLES env var ("" or "none" → all).
_max_env = os.environ.get("MAX_EXAMPLES", "").strip().lower()
MAX_EXAMPLES: int | None = (
    None if _max_env in ("", "none", "null") else int(_max_env)
)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(_HERE, "results")

# Matches translate_ollama's num_predict default in the original translation.py.
MAX_TOKENS = 1024

# 0 = zero-shot, N>0 = N few-shot exemplars drawn from the start of the file.
NUM_SHOTS = 0

# In-flight requests to vLLM. Higher → better batcher utilisation until the
# GPU saturates. 32 is a safe default for a single A4000/A5000.
CONCURRENCY = 32


# ---------------------------------------------------------------------------
# Prompt construction — byte-for-byte equivalent to translate_ollama()
# ---------------------------------------------------------------------------

def get_source_language_name(example: dict) -> str | None:
    return example.get("lang_codes")


def build_prompt(
    text: str,
    source_language_name: str | None,
    few_shot_examples: list[dict] | None = None,
) -> str:
    """Reproduces translate_ollama()'s prompt string exactly."""
    prompt_parts: list[str] = []

    if few_shot_examples:
        prompt_parts.append("The following are example translations:")
        for ex in few_shot_examples:
            example_language = ex.get("language") or ex.get("src_lang_name") or "Unknown"
            prompt_parts.append(f"Source ({example_language}): {ex['src']}")
            prompt_parts.append(f"Translation: {ex['ref']}")
            prompt_parts.append("")
        language_phrase = f"{source_language_name} " if source_language_name else ""
        prompt_parts.append(
            f"Translate the following {language_phrase}text to English. Output only the translation, nothing else."
        )
        prompt_parts.append(f"Source ({source_language_name or 'Unknown'}): {text}")
        prompt_parts.append("Translation:")
    else:
        language_phrase = f"{source_language_name} " if source_language_name else ""
        prompt_parts.append(
            f"Translate the following {language_phrase}text to English. Output only the translation, nothing else."
        )
        prompt_parts.append(f"\n{text}")

    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# HF revision resolution
# ---------------------------------------------------------------------------

def resolve_hf_revision(model_id: str) -> str | None:
    """Best-effort lookup of the HF commit SHA backing `model_id`.

    Tries the local HF cache first so offline runs (the common case on the
    Tailscale mesh) still pin a revision. Falls back to the Hub API. Returns
    None on failure — callers should treat this as advisory metadata, not a
    hard requirement.
    """
    candidate_caches: list[str] = []
    for env_var in ("HF_HUB_DIR", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env_var):
            candidate_caches.append(os.environ[env_var])
    candidate_caches.append(os.path.join(_HERE, "hf_cache", "hub"))
    candidate_caches.append(
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    )

    org_name = model_id.replace("/", "--")
    for cache_root in candidate_caches:
        refs_dir = os.path.join(cache_root, f"models--{org_name}", "refs")
        if not os.path.isdir(refs_dir):
            continue
        # Prefer 'main' but fall through to whatever ref is there.
        ref_names = ["main"] + [r for r in os.listdir(refs_dir) if r != "main"]
        for ref_name in ref_names:
            ref_path = os.path.join(refs_dir, ref_name)
            if not os.path.isfile(ref_path):
                continue
            try:
                with open(ref_path, encoding="utf-8") as fh:
                    sha = fh.read().strip()
                if sha:
                    return sha
            except OSError:
                continue

    try:
        from huggingface_hub import HfApi
        return HfApi().repo_info(model_id).sha
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Eval-dataset resolution (ClearML Dataset → local JSONL paths)
# ---------------------------------------------------------------------------

def resolve_eval_datasets(names: list[str], project: str) -> list[str]:
    """Materialise each ClearML Dataset locally and return JSONL paths.

    Hard-fails with a clear message if the SDK or server is unreachable —
    the caller passed dataset names explicitly, so silently falling back
    would mask configuration mistakes.
    """
    from clearml import Dataset  # imported here so non-ClearML runs avoid the dep

    paths: list[str] = []
    for name in names:
        ds = Dataset.get(dataset_project=project, dataset_name=name)
        local_dir = ds.get_local_copy()
        jsonls = sorted(glob.glob(os.path.join(local_dir, "*.jsonl"))) \
            + sorted(glob.glob(os.path.join(local_dir, "**", "*.jsonl"), recursive=True))
        # de-dupe while preserving order
        seen: set[str] = set()
        unique = [p for p in jsonls if not (p in seen or seen.add(p))]
        if not unique:
            raise SystemExit(
                f"Dataset '{name}' resolved to {local_dir} but contains no "
                f"*.jsonl files. Did the dataset get registered correctly?"
            )
        paths.extend(unique)
    return paths


# ---------------------------------------------------------------------------
# vLLM readiness
# ---------------------------------------------------------------------------

def wait_for_vllm(timeout: int = 300, interval: int = 5) -> str:
    """Block until vLLM is serving a model. Returns the model id."""
    print(f"Waiting for vLLM at {VLLM_HOST} (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{VLLM_HOST}/v1/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    model_id = models[0]["id"]
                    print(f"vLLM ready. Serving: {model_id}")
                    return model_id
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(interval)
    print("ERROR: vLLM not ready. Is the container running?")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_data(path: str, max_examples: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if max_examples is not None:
        data = data[:max_examples]
    return data


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

async def translate_one(
    client: AsyncOpenAI,
    model: str,
    entry: dict,
    few_shot_examples: list[dict] | None,
    sem: asyncio.Semaphore,
) -> dict:
    language_name = get_source_language_name(entry)
    prompt = build_prompt(entry.get("text", ""), language_name, few_shot_examples)

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0,
            )
            hypothesis = (resp.choices[0].message.content or "").strip()
            error = None
        except Exception as e:
            hypothesis = None
            error = str(e)

    return {
        "model": model,
        "id1": entry.get("id1"),
        "id2": entry.get("id2"),
        "lang_codes": entry.get("lang_codes"),
        "language": language_name,
        "source": entry.get("text"),
        "hypothesis": hypothesis,
        "reference": entry.get("translation"),
        "error": error,
    }


async def run_batch(
    client: AsyncOpenAI,
    model: str,
    data: list[dict],
    few_shot_examples: list[dict] | None,
) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    n = len(data)
    done_count = 0
    results: list[dict | None] = [None] * n

    async def runner(i: int, entry: dict) -> None:
        nonlocal done_count
        row = await translate_one(client, model, entry, few_shot_examples, sem)
        results[i] = row
        done_count += 1
        if done_count % 25 == 0 or done_count == n:
            print(f"  [{done_count:>{len(str(n))}}/{n}] completed")

    await asyncio.gather(*(runner(i, e) for i, e in enumerate(data)))
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_csv(model: str, rows: list[dict], output_dir: str, eval_path: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model.replace(":", "_").replace("/", "_")
    input_stem = os.path.splitext(os.path.basename(eval_path))[0]
    path = os.path.join(output_dir, f"{safe_name}_translations_{input_stem}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"Batch translation (vLLM) — {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"vLLM host    : {VLLM_HOST}")

    if EVAL_DATASETS:
        print(f"Eval datasets: {EVAL_DATASETS}  (project={EVAL_DATASETS_PROJECT})")
        eval_paths = resolve_eval_datasets(EVAL_DATASETS, EVAL_DATASETS_PROJECT)
        print(f"  resolved → {eval_paths}")
    else:
        eval_paths = EVAL_PATHS
        print(f"Eval files   : {eval_paths}")

    print(f"Max examples : {MAX_EXAMPLES or 'all'}  (per file)")
    print(f"Num shots    : {NUM_SHOTS}")
    print(f"Concurrency  : {CONCURRENCY}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print()

    model = wait_for_vllm()
    revision = resolve_hf_revision(model)
    print(f"Model rev    : {revision or 'unresolved'}")
    client = AsyncOpenAI(base_url=f"{VLLM_HOST}/v1", api_key="unused")

    for eval_path in eval_paths:
        print("=" * 60)
        print(f"INPUT: {eval_path}")
        print(f"MODEL: {model}")
        print("=" * 60)

        data = load_eval_data(eval_path, max_examples=MAX_EXAMPLES)

        few_shot_examples: list[dict] | None = None
        if NUM_SHOTS > 0:
            if NUM_SHOTS >= len(data):
                raise ValueError(
                    f"NUM_SHOTS ({NUM_SHOTS}) must be less than the dataset size ({len(data)}) for {eval_path}"
                )
            shot_data = data[:NUM_SHOTS]
            data = data[NUM_SHOTS:]
            few_shot_examples = [
                {
                    "src": ex.get("text", ""),
                    "ref": ex.get("translation", ""),
                    "language": get_source_language_name(ex),
                }
                for ex in shot_data
            ]
            print(f"Few-shot mode: {NUM_SHOTS} exemplars reserved, {len(data)} to translate\n")
        else:
            print(f"Zero-shot mode — {len(data)} to translate\n")

        t0 = time.perf_counter()
        rows = await run_batch(client, model, data, few_shot_examples)
        elapsed = time.perf_counter() - t0

        csv_path = save_csv(model, rows, OUTPUT_DIR, eval_path)
        n_errors = sum(1 for r in rows if r["error"])

        meta = {
            "model": model,
            "model_revision": revision,
            "eval_path": eval_path,
            "csv_path": csv_path,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_rows": len(rows),
            "n_errors": n_errors,
            "num_shots": NUM_SHOTS,
            "concurrency": CONCURRENCY,
            "elapsed_seconds": round(elapsed, 3),
            "req_per_s": round(len(rows) / elapsed, 3) if elapsed > 0 else None,
        }
        meta_path = f"{csv_path}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"\n  Translations → {csv_path}")
        print(f"  Run metadata → {meta_path}")
        print(
            f"  Total elapsed: {elapsed:.1f}s "
            f"({len(rows) / elapsed:.2f} req/s, {n_errors} errors)\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
