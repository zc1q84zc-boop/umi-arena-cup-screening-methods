#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
step="${2:-5000}"
if [[ "$mode" != smoke && "$mode" != retry_first && "$mode" != resume ]]; then
  echo "usage: launch_openwam_replay.sh smoke|retry_first|resume [5000|10000]" >&2
  exit 2
fi
if [[ "$step" != 5000 && "$step" != 10000 ]]; then
  echo "checkpoint step must be 5000 or 10000" >&2
  exit 2
fi
work=/mnt/data/benyun/workspace/umi_cup_multimodel_20260925
old=/mnt/data/benyun/workspace/openwam_charger_full_20260922
py="$old/.venv/bin/python"
checkpoint="$work/openwam/checkpoints/2026-09-25_18-55-32"
output="$work/replays/openwam-cup-clean-$step"

exec 9>"$work/replays/openwam-$step.lock"
flock -n 9 || { echo "OpenWAM replay already running" >&2; exit 1; }
"$py" "$work/scripts/gpu_guard.py" --gpu 6
[[ -f "$checkpoint/checkpoint_step_$step.safetensors" ]] || { echo "weight missing" >&2; exit 1; }
[[ -f "$checkpoint/config.yaml" && -f "$checkpoint/normalization_stats.npy" ]] || {
  echo "deploy config or normalization missing" >&2; exit 1;
}

if [[ "$mode" == smoke ]]; then
  [[ ! -e "$output" ]] || { echo "replay output exists; refusing overwrite" >&2; exit 1; }
  extra=(--limit-episodes 1)
else
  [[ -f "$output/report.json" ]] || { echo "smoke report missing" >&2; exit 1; }
  extra=(--resume)
  if [[ "$mode" == retry_first ]]; then
    extra+=(--limit-episodes 1)
  fi
fi

export CUDA_VISIBLE_DEVICES=6
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$work/openwam/torch_extensions_cu128"
export PYTHONPATH="$work/evaluation:$old/OpenWAM:$work/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
cd "$work"
exec nice -n 19 ionice -c 3 "$py" "$work/scripts/replay_openwam_cup.py" \
  --evaluation-root "$work/evaluation" \
  --openwam-root "$old/OpenWAM" \
  --dataset /mnt/data/benyun/workspace/yubi-corl2026-umi-arena \
  --suite "$work/evaluation/suites/practice-cup-smartphone.json" \
  --train-manifest "$work/selection/manifest.json" \
  --prepared "$work/openwam/prepared" \
  --checkpoint-dir "$checkpoint" \
  --checkpoint-name "checkpoint_step_$step.safetensors" \
  --checkpoint-id "openwam-cup-clean-$step" \
  --output "$output" "${extra[@]}"
