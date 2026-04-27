#!/usr/bin/env bash
# ---------------------------------------------------------------
# QUEST — run training for one or both datasets.
#
# Usage:
#   bash run.sh nl27k                    # CDL baseline, single seed
#   bash run.sh nl27k multi              # CDL baseline, 3 seeds
#   bash run.sh nl27k single 0 --cpn     # + Confidence Propagation
#   bash run.sh nl27k single 0 --srt     # + Spectral Relational Transform
#   bash run.sh nl27k multi 0 --cpn --srt  # full QUEST model, 3 seeds
#   bash run.sh all multi 0 --cpn --srt    # both datasets, full model
# ---------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

DATASET="${1:-nl27k}"
MODE="${2:-single}"
GPU="${3:-0}"
shift 3 2>/dev/null || true
EXTRA_FLAGS="$*"       # --cpn --srt etc.
DATA_PATH="dataset"

if [ "$MODE" = "multi" ]; then
    SEEDS="42,123,456"
else
    SEEDS="42"
fi

run_one() {
    local ds="$1"
    echo "============================================================"
    echo "  QUEST: $ds | seeds=$SEEDS | GPU=$GPU | flags: $EXTRA_FLAGS"
    echo "============================================================"
    python train.py \
        --dataset "$ds" \
        --data_path "$DATA_PATH" \
        --seeds "$SEEDS" \
        --gpu "$GPU" \
        $EXTRA_FLAGS
}

if [ "$DATASET" = "all" ]; then
    run_one nl27k
    run_one cn15k
else
    run_one "$DATASET"
fi

echo ""
echo "Done.  Results in output/<dataset>/seed_*/"
