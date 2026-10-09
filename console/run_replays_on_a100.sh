#!/usr/bin/env bash
set -euo pipefail

WORK=/mnt/data/benyun/workspace/pi05_cup_clean_20260923
BASE=/mnt/data/benyun/umi_arena_pi05_20260922
EVAL=$BASE/evaluation
PY=/home/benyun/.venvs/umi_arena_pi05/bin/python
export PYTHONPATH="$WORK/scripts:$BASE/openpi/src:$EVAL"
export HF_HOME=/mnt/data/benyun/workspace/huggingface-cache
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

for step in 10000 20000 30000; do
  identifier="pi05-cup-clean-$step"
  output="$WORK/replay_results/$identifier"
  checkpoint="$WORK/checkpoints/pi05_cup_clean_success/pi05_cup_clean_v1/$step"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing replay: $output" >&2
    exit 1
  fi
  if [[ ! -f "$checkpoint/_CHECKPOINT_METADATA" ]]; then
    echo "Incomplete checkpoint: $checkpoint" >&2
    exit 1
  fi
  # Let other GPU users finish before loading the next checkpoint.
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "GPU busy; waiting before $identifier" >&2
    sleep 60
  done
  sleep 15
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU became busy before $identifier; stopping without overlap" >&2
    exit 1
  fi
  echo "MODEL_REPLAY_START $identifier"
  ionice -c 3 nice -n 10 "$PY" "$WORK/scripts/replay_pi05_cup.py" \
    --evaluation-root "$EVAL" \
    --openpi-root "$BASE/openpi" \
    --dataset /mnt/data/benyun/workspace/yubi-corl2026-umi-arena \
    --suite "$EVAL/suites/practice-cup-smartphone.json" \
    --train-manifest "$WORK/selection/manifest.json" \
    --checkpoint "$checkpoint" \
    --checkpoint-id "$identifier" \
    --output "$output"
  echo "MODEL_REPLAY_DONE $identifier"
done
