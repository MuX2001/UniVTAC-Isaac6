#!/usr/bin/env bash
# Collect one successful, deterministic scripted episode per contact-rich
# UniVTAC task. No learned policy, checkpoint, or openpi inference code is
# imported by this launcher.
set -euo pipefail

UNIVTAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$UNIVTAC_ROOT"

GPU_ID="${1:-0}"
MAX_SEED="${MAX_SEED:-20}"
CONFIG="${CONFIG:-isaac4_baseline_raw}"
TASKS="${TASKS:-lift_bottle lift_can insert_HDMI insert_hole insert_tube pull_out_key put_bottle_in_shelf}"

if ! command -v python >/dev/null 2>&1; then
    echo "[baseline][ERROR] python was not found; activate univtac-isaac4-baseline first." >&2
    exit 1
fi

python - <<'PY'
import importlib.util

required = ("isaacsim", "isaaclab", "tacex_uipc")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing Isaac Sim 4 baseline dependencies: " + ", ".join(missing) +
        ". Activate the separate Isaac-4 environment first."
    )
PY

for task in $TASKS; do
    echo "[baseline] task=$task config=$CONFIG gpu=$GPU_ID"
    python scripts/collect_data.py "$task" "$CONFIG" \
        --gpu "$GPU_ID" \
        --episode_num 1 \
        --start_seed 0 \
        --max_seed "$MAX_SEED"

    episode_dir="data/isaac4_baseline/$task/$CONFIG/hdf5"
    if [[ ! -d "$episode_dir" ]]; then
        echo "[baseline][ERROR] $task did not create $episode_dir" >&2
        exit 1
    fi
    episode="$(find "$episode_dir" -maxdepth 1 -type f -name '*.hdf5' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
    if [[ -z "$episode" ]]; then
        echo "[baseline][ERROR] $task did not produce an HDF5 episode" >&2
        exit 1
    fi
    python scripts/inspect_baseline_episode.py "$episode"
done
