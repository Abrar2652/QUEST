#!/usr/bin/env bash
# ---------------------------------------------------------------
# Run all experiments in parallel on different GPUs:
#   GPU 0: ssCDL baseline — NL27k (500 epochs)
#   GPU 1: ssCDL baseline — CN15k (300 epochs)
#   GPU 2: QUEST — NL27k (500 epochs)
#   GPU 3: QUEST — CN15k (300 epochs)
# ---------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH=src
PYTHON=/nas/home/jahin/QUEST/.venv/bin/python

rm -rf output training

echo "=== Launching 4 parallel experiments ==="

# GPU 0: ssCDL NL27k
echo "[GPU 0] ssCDL NL27k (500 epochs)"
nohup $PYTHON sscdl_run.py --gpu 0 \
    --load_config --config_path config/nl27k/ssCDL_nl27k.yaml \
    > logs_sscdl_nl27k.txt 2>&1 &
PID0=$!

# GPU 1: ssCDL CN15k
echo "[GPU 1] ssCDL CN15k (300 epochs)"
nohup $PYTHON sscdl_run.py --gpu 1 \
    --load_config --config_path config/cn15k/ssCDL_cn15k.yaml \
    > logs_sscdl_cn15k.txt 2>&1 &
PID1=$!

# GPU 2: QUEST NL27k
echo "[GPU 2] QUEST NL27k (500 epochs)"
nohup $PYTHON quest_run.py --gpu 2 \
    --load_config --config_path config/nl27k/QUEST_nl27k.yaml \
    > logs_quest_nl27k.txt 2>&1 &
PID2=$!

# GPU 3: QUEST CN15k
echo "[GPU 3] QUEST CN15k (300 epochs)"
nohup $PYTHON quest_run.py --gpu 3 \
    --load_config --config_path config/cn15k/QUEST_cn15k.yaml \
    > logs_quest_cn15k.txt 2>&1 &
PID3=$!

echo ""
echo "PIDs: ssCDL_nl27k=$PID0  ssCDL_cn15k=$PID1  QUEST_nl27k=$PID2  QUEST_cn15k=$PID3"
echo ""
echo "Monitor with:"
echo "  tail -f logs_sscdl_nl27k.txt"
echo "  tail -f logs_quest_nl27k.txt"
echo ""
echo "Check progress:"
echo "  ls -la output/confidence_prediction/*/ssCDL*/*.ckpt"
echo "  ls -la output/confidence_prediction/*/ssCDL_QUEST*/*.ckpt"
