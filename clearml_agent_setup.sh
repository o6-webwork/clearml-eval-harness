#!/usr/bin/env bash
# clearml_agent_setup.sh — Bootstrap a ClearML agent on a worker machine.
#
# Run this once on each GPU terminal. It:
#   1. Creates an isolated venv for the agent (won't touch your working .venv)
#   2. Installs clearml + clearml-agent (both are required)
#   3. Copies the config template to ~/clearml.conf (if not already present)
#   4. Creates the necessary ClearML queues.
#      The two-queue split (default / gpu-*) is load-bearing — the pipeline
#      controller goes on default, GPU work goes on the gpu queue.
#      A single agent on one queue deadlocks (see README_clearml.md §4).
#   5. With --start, launches one daemon per queue with the correct
#      CUDA_VISIBLE_DEVICES assignment.  Without --start, prints the
#      commands to start them yourself.
#
# Prerequisites:
#   - Python 3.10+ available as `python3`
#   - The ClearML server is already running (orchestration/clearml-server/)
#   - SSH key set up for rsync if you'll use clearml_sync_data.sh:
#       ssh-keygen -t ed25519    (enter through all prompts)
#       ssh-copy-id <CANONICAL_HOST_IP>
#
# Usage (from repo root):
#
#   # Legacy single-GPU mode — one daemon on the 'gpu' queue, no GPU binding
#   bash orchestration/clearml_agent_setup.sh
#   bash orchestration/clearml_agent_setup.sh --start
#
#   # 1x mode — one daemon per GPU, each bound to its own GPU
#   bash orchestration/clearml_agent_setup.sh --mode 1x --start
#
#   # 2x mode — one daemon per GPU pair (for models needing 2×48 GB)
#   bash orchestration/clearml_agent_setup.sh --mode 2x --start
#
#   # 4x mode — one daemon for all GPUs (for models needing 4×48 GB)
#   bash orchestration/clearml_agent_setup.sh --mode 4x --start
#
# Queue naming convention (matches conf/models.yml):
#   gpu      legacy / single GPU, no explicit binding
#   gpu-1x   one GPU per task   (--mode 1x)
#   gpu-2x   two GPUs per task  (--mode 2x)
#   gpu-4x   four GPUs per task (--mode 4x)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_VENV="$HOME/.clearml/agent-venv"
CONF_TEMPLATE="$SCRIPT_DIR/conf/clearml.conf.template"
# ClearML looks for ~/clearml.conf (no dot prefix).
CONF_TARGET="$HOME/clearml.conf"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
START=false
MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)
            START=true
            shift
            ;;
        --mode)
            if [[ $# -lt 2 ]]; then
                echo "Error: --mode requires a value (1x, 2x, or 4x)"
                exit 1
            fi
            MODE="$2"
            shift 2
            ;;
        --help|-h)
            sed -n '/^# Usage/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $1  (run with --help to see usage)"
            exit 1
            ;;
    esac
done

case "$MODE" in
    ""|1x|2x|4x) ;;
    *) echo "Invalid --mode '$MODE'. Valid values: 1x, 2x, 4x"; exit 1 ;;
esac

# Compute GPU queue name and full queue list
if [[ -z "$MODE" ]]; then
    GPU_QUEUE="gpu"
else
    GPU_QUEUE="gpu-$MODE"
fi
QUEUES=("default" "$GPU_QUEUE")

echo "=== ClearML Agent Setup ==="
echo ""
if [[ -n "$MODE" ]]; then
    echo "  Mode   : $MODE (GPU queue: $GPU_QUEUE)"
else
    echo "  Mode   : legacy (GPU queue: gpu — no CUDA_VISIBLE_DEVICES binding)"
fi

# ---------------------------------------------------------------------------
# 1. Create agent venv
# ---------------------------------------------------------------------------
if [ -d "$AGENT_VENV" ]; then
    echo "Agent venv already exists: $AGENT_VENV"
else
    echo "Creating agent venv at $AGENT_VENV ..."
    python3 -m venv "$AGENT_VENV"
fi
# shellcheck disable=SC1091
source "$AGENT_VENV/bin/activate"

echo "Installing clearml + clearml-agent ..."
pip install --quiet --upgrade clearml clearml-agent

# ---------------------------------------------------------------------------
# 2. Config file
# ---------------------------------------------------------------------------
if [ -f "$CONF_TARGET" ]; then
    echo "Config already exists: $CONF_TARGET"
    echo "  (edit it manually if you need to change the server address)"
else
    echo "Copying config template to $CONF_TARGET ..."
    cp "$CONF_TEMPLATE" "$CONF_TARGET"
    echo ""
    echo "  *** IMPORTANT: edit $CONF_TARGET now ***"
    echo "  Replace CLEARML_HOST, ACCESS_KEY, and SECRET_KEY with real values."
    echo "  Remember: URLs must be in double quotes (HOCON treats // as a comment)."
    echo "  Or run: clearml-init"
    echo ""
fi

# ---------------------------------------------------------------------------
# 3. Verify connectivity
# ---------------------------------------------------------------------------
echo ""
echo "Testing connection to ClearML server ..."
if python -c "from clearml import Task; Task.set_offline(True)" 2>/dev/null; then
    echo "  clearml SDK importable — good."
else
    echo "  WARNING: could not import clearml. Check your config."
fi

# ---------------------------------------------------------------------------
# 4. Ensure queues exist
# ---------------------------------------------------------------------------
echo ""
echo "Ensuring queues exist: ${QUEUES[*]} ..."
QUEUES_JSON=$(printf '"%s",' "${QUEUES[@]}")
QUEUES_JSON="[${QUEUES_JSON%,}]"
python -c "
from clearml.backend_api.session.client import APIClient
queues = $QUEUES_JSON
try:
    client = APIClient()
    for q in queues:
        existing = client.queues.get_all(name=q)
        if not existing:
            client.queues.create(name=q)
            print(f'  Created queue: {q}')
        else:
            print(f'  Queue already exists: {q}')
except Exception as e:
    print(f'  Could not verify queues (server may be unreachable): {e}')
    print('  Create them manually:')
    print('    python -c \"from clearml.backend_api.session.client import APIClient;'
          ' c=APIClient(); [c.queues.create(name=q) for q in ' + repr(queues) + ']\"')
" 2>/dev/null || echo "  Skipped queue check (server unreachable or config incomplete)"

# ---------------------------------------------------------------------------
# 5. Detect GPUs (needed for mode-specific daemon binding)
# ---------------------------------------------------------------------------
GPU_COUNT=0
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
fi
if [[ -n "$MODE" ]]; then
    echo ""
    echo "Detected $GPU_COUNT GPU(s) on this host."
fi

# ---------------------------------------------------------------------------
# Helper: compute daemon start commands for the current mode
# Output: one shell command per line, ready to be eval'd or printed
# ---------------------------------------------------------------------------
_daemon_commands() {
    local mode="$1"
    local gpu_count="$2"
    local gpu_queue="$3"

    # Controller daemon — always unbound
    echo "clearml-agent daemon --queue default --detach"

    case "$mode" in
        1x)
            if [[ $gpu_count -eq 0 ]]; then
                # No nvidia-smi — start a single unbound agent and warn
                echo "# WARNING: no GPUs detected; starting one unbound agent"
                echo "clearml-agent daemon --queue $gpu_queue --detach"
            else
                for ((i=0; i<gpu_count; i++)); do
                    echo "CUDA_VISIBLE_DEVICES=$i clearml-agent daemon --queue $gpu_queue --detach"
                done
            fi
            ;;
        2x)
            local pairs=$(( gpu_count / 2 ))
            if [[ $pairs -eq 0 ]]; then
                echo "# WARNING: fewer than 2 GPUs detected ($gpu_count); starting one unbound agent"
                echo "clearml-agent daemon --queue $gpu_queue --detach"
            else
                for ((i=0; i<pairs; i++)); do
                    local gpus="$((i*2)),$((i*2+1))"
                    echo "CUDA_VISIBLE_DEVICES=$gpus clearml-agent daemon --queue $gpu_queue --detach"
                done
            fi
            ;;
        4x)
            if [[ $gpu_count -lt 4 ]]; then
                echo "# WARNING: fewer than 4 GPUs detected ($gpu_count); starting one unbound agent"
                echo "clearml-agent daemon --queue $gpu_queue --detach"
            else
                local gpus
                gpus=$(seq -s, 0 $((gpu_count-1)))
                echo "CUDA_VISIBLE_DEVICES=$gpus clearml-agent daemon --queue $gpu_queue --detach"
            fi
            ;;
        "")
            # Legacy mode — no GPU binding
            echo "clearml-agent daemon --queue $gpu_queue --detach"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# 6. Start or print daemons
# ---------------------------------------------------------------------------
if $START; then
    echo ""
    echo "Starting agent daemons ..."
    echo "  (to stop later: pkill -f 'clearml-agent daemon')"
    echo ""
    while IFS= read -r cmd; do
        # Print comment lines but don't execute them
        if [[ "$cmd" == \#* ]]; then
            echo "  $cmd"
            continue
        fi
        echo "  \$ $cmd"
        eval "$cmd"
    done < <(_daemon_commands "$MODE" "$GPU_COUNT" "$GPU_QUEUE")
else
    echo ""
    echo "=== Setup complete ==="
    echo ""
    echo "To start the agents, run:"
    echo "  source $AGENT_VENV/bin/activate"
    echo ""
    while IFS= read -r cmd; do
        echo "  $cmd"
    done < <(_daemon_commands "$MODE" "$GPU_COUNT" "$GPU_QUEUE")
    echo ""
    echo "Or start everything via this script:"
    if [[ -n "$MODE" ]]; then
        echo "  bash orchestration/clearml_agent_setup.sh --mode $MODE --start"
    else
        echo "  bash orchestration/clearml_agent_setup.sh --start"
    fi
    echo ""
    echo "To stop all daemons later:"
    echo "  pkill -f 'clearml-agent daemon'"
fi
