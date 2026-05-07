#!/usr/bin/env bash
# clearml_sync_data.sh — Pull eval data + HF model weights from the canonical host.
#
# Designed for the Tailscale mesh: every worker runs this before starting an
# eval to ensure it has up-to-date JSONL files and cached model weights.
#
# What it syncs:
#   1. JSONL eval files (mono_sea.jsonl, mono_me.jsonl, etc.) → data/
#   2. HF model cache (weights, tokenizers) → hf_cache/
#
# The canonical host is the machine that has the most complete / freshest copy
# of the HF cache. Set CANONICAL_HOST to its Tailscale hostname or IP.
#
# Usage (from anywhere — paths are computed from the script's own location):
#   CANONICAL_HOST=my-dgx bash clearml_sync_data.sh
#   CANONICAL_HOST=100.64.0.1 bash clearml_sync_data.sh
#   CANONICAL_HOST=my-dgx SYNC_DATA_ONLY=1 bash clearml_sync_data.sh
#   CANONICAL_HOST=my-dgx SYNC_WEIGHTS_ONLY=1 bash clearml_sync_data.sh
#
# REMOTE_DATA_DIR overrides the source location on the canonical host. Default
# matches the clearml-eval-harness layout (~/Desktop/clearml-eval-harness).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

CANONICAL_HOST="${CANONICAL_HOST:?Set CANONICAL_HOST to the Tailscale hostname or IP of the canonical data machine}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-~/Desktop/clearml-eval-harness}"

echo "=== Data Sync ==="
echo "Canonical host : $CANONICAL_HOST"
echo "Remote dir     : $REMOTE_DATA_DIR"
echo "Local repo     : $REPO_ROOT"
echo ""

# --- 1. JSONL eval files ----------------------------------------------------
if [[ "${SYNC_WEIGHTS_ONLY:-}" != "1" ]]; then
    echo "--- Syncing eval JSONL files ---"
    mkdir -p "$REPO_ROOT/data"
    rsync -avP --include='*.jsonl' --exclude='*' \
        "$CANONICAL_HOST:$REMOTE_DATA_DIR/data/" \
        "$REPO_ROOT/data/"
    echo ""
fi

# --- 2. HF model cache ------------------------------------------------------
if [[ "${SYNC_DATA_ONLY:-}" != "1" ]]; then
    echo "--- Syncing HF model cache ---"
    echo "  This may take a while for large models (rsync is resumable)."
    echo ""
    mkdir -p "$REPO_ROOT/hf_cache/hub"
    rsync -avP --delete-after \
        "$CANONICAL_HOST:$REMOTE_DATA_DIR/hf_cache/hub/" \
        "$REPO_ROOT/hf_cache/hub/"
    echo ""
fi

echo "=== Sync complete ==="
echo ""
echo "Verify with:"
echo "  ls $REPO_ROOT/data/*.jsonl"
echo "  ls $REPO_ROOT/hf_cache/hub/ | head -20"
