#!/usr/bin/env bash
set -euo pipefail

step="${1:-10000}"
if [[ "$step" != 5000 && "$step" != 10000 ]]; then
  echo "usage: launch_openwam_sim_server.sh 5000|10000" >&2
  exit 2
fi
work=/mnt/data/benyun/workspace/umi_cup_multimodel_20260925
old=/mnt/data/benyun/workspace/openwam_charger_full_20260922
py="$old/.venv/bin/python"
checkpoint="$work/openwam/checkpoints/2026-09-25_18-55-32"

exec 9>"$work/openwam/sim_online.lock"
flock -n 9 || { echo "OpenWAM simulator service already running" >&2; exit 1; }
"$py" "$work/scripts/gpu_guard.py" --gpu 5
[[ -s "$checkpoint/checkpoint_step_$step.safetensors" ]] || {
  echo "OpenWAM checkpoint missing" >&2; exit 1;
}
[[ -s "$checkpoint/config.yaml" && -s "$checkpoint/normalization_stats.npy" ]] || {
  echo "OpenWAM deploy config or normalization missing" >&2; exit 1;
}

export CUDA_VISIBLE_DEVICES=5
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$work/openwam/torch_extensions_cu128"
export PYTHONPATH="$work/evaluation:$old/OpenWAM:$work/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
cd "$work"
exec nice -n 19 ionice -c 3 "$py" "$work/scripts/openwam_sim_server.py" \
  --checkpoint-dir "$checkpoint" --checkpoint-name "checkpoint_step_$step.safetensors" \
  --prepared "$work/openwam/prepared" --openwam-root "$old/OpenWAM" \
  --model-id "openwam-cup-clean-$step" --port 18787
