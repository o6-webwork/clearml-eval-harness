"""
Register eval JSONL files as ClearML Datasets.

Scans a directory (recursively) for *.jsonl files and registers each as a
ClearML Dataset. Descriptions are read from conf/datasets.yml. Files with no
entry in that file prompt for confirmation before registering with a generic
description, or can be skipped.

Usage:
  cd orchestration

  # Scan default location (bench/ — picks up bench/*.jsonl and
  # bench/orchestration/data/*.jsonl in one pass)
  python clearml_register_datasets.py

  # Point at a specific folder
  python clearml_register_datasets.py --datasets-dir /path/to/jsonls
"""

import argparse
import os
from datetime import date

import yaml
from clearml import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
METADATA_FILE = os.path.join(HERE, "conf", "datasets.yml")
PROJECT = "translation-evals"

# Directory names to skip during the recursive scan. Extend this list if you
# add new non-data subdirectories under bench/.
_SKIP_DIRS = {
    "hf_cache", ".git", "__pycache__", ".venv", "venv",
    "node_modules", ".clearml",
}


def load_metadata(path: str) -> dict[str, str]:
    """Return stem → description mapping from YAML. Empty dict if file missing."""
    if not os.path.isfile(path):
        print(f"  Warning: metadata file not found at {path}")
        print("  All files will require interactive confirmation.")
        return {}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {k: str(v) for k, v in data.items()}


def find_jsonl_files(directory: str) -> list[str]:
    """Walk directory tree recursively for *.jsonl, pruning non-data subtrees."""
    if not os.path.isdir(directory):
        raise SystemExit(f"Datasets directory not found: {directory}")
    found = []
    for dirpath, dirnames, filenames in os.walk(directory):
        # Prune in-place: skip known non-data dirs and results/* output dirs
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith("results")
        )
        for f in sorted(filenames):
            if f.endswith(".jsonl"):
                found.append(os.path.join(dirpath, f))
    return found


def prompt_description(stem: str, filepath: str) -> str | None:
    """
    Prompt user to confirm a generic description when none is in metadata.
    Returns the description string, or None to skip this file.
    """
    generic = f"Registered {date.today().strftime('%d-%m-%Y')} — {os.path.basename(filepath)}"
    rel = os.path.relpath(filepath, REPO_ROOT)
    print(f"\n  No description found for '{stem}' ({rel}).")
    print(f"  Generic description: \"{generic}\"")
    while True:
        answer = input("  Register with generic description? [y/skip]: ").strip().lower()
        if answer == "y":
            return generic
        if answer in ("skip", "s", "n", ""):
            print(f"  Skipping {os.path.basename(filepath)}.")
            return None
        print("  Please enter 'y' to accept or 'skip' to skip.")


def register_one(path: str, description: str) -> None:
    name = os.path.splitext(os.path.basename(path))[0]
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"Registering: {name}  ({size_mb:.1f} MB)")
    print(f"  Description: {description}")

    ds = Dataset.create(
        dataset_project=PROJECT,
        dataset_name=name,
        description=description,
    )
    ds.add_files(path)
    ds.upload(output_url=None)
    ds.finalize()
    print(f"  Done — dataset ID: {ds.id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register eval JSONL files as ClearML Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets-dir",
        default=REPO_ROOT,
        metavar="DIR",
        help="Root directory to scan for *.jsonl files (default: bench/). "
             "Scanned recursively; results/ and hf_cache/ subtrees are skipped.",
    )
    args = parser.parse_args()

    metadata = load_metadata(METADATA_FILE)
    files = find_jsonl_files(args.datasets_dir)

    if not files:
        print(f"No .jsonl files found under {args.datasets_dir}")
        return

    print(f"Registering datasets to project '{PROJECT}'")
    print(f"Scanning:  {args.datasets_dir}  ({len(files)} file(s) found)")
    print(f"Metadata:  {METADATA_FILE}")

    registered, skipped = 0, 0
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        description = metadata.get(stem)

        if description is None:
            description = prompt_description(stem, path)
            if description is None:
                skipped += 1
                continue

        register_one(path, description)
        registered += 1

    print(f"\n{'=' * 50}")
    print(f"Done — {registered} registered, {skipped} skipped.")
    print(f"View in the ClearML UI: Project '{PROJECT}' → Datasets")


if __name__ == "__main__":
    main()
