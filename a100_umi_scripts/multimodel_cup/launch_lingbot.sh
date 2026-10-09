#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
if [[ "$mode" != smoke && "$mode" != train ]]; then
  echo "usage: launch_lingbot.sh smoke|train" >&2
  exit 2
fi
work=/mnt/data/benyun/workspace/umi_cup_multimodel_20260925
repo=/mnt/data/benyun/workspace/projects/lingbot-vla-v2
py=/home/benyun/miniconda3/envs/lingbotvla/bin/python
torchrun=/home/benyun/miniconda3/envs/lingbotvla/bin/torchrun

exec 9>"$work/lingbot/training.lock"
flock -n 9 || { echo "another LingBot launcher owns the lock" >&2; exit 1; }
"$py" "$work/scripts/gpu_guard.py" --gpu 1 2 3 4
[[ -f "$work/lingbot/norm_stats.json" ]] || { echo "clean-data norms missing" >&2; exit 1; }
[[ -f "$work/selection/verified_allowlist.json" ]] || { echo "audited allowlist missing" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTHONPATH="$repo:$work/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

cd "$repo"
if [[ "$mode" == smoke ]]; then
  [[ ! -e "$work/lingbot/smoke" ]] || { echo "smoke directory exists; refusing overwrite" >&2; exit 1; }
  exec "$torchrun" --standalone --nproc_per_node=4 "$work/scripts/train_lingbot_clean.py" \
    "$work/lingbot/config.yaml" --train.output_dir "$work/lingbot/smoke" \
    --train.max_steps 1 --train.save_steps 1
fi
[[ -f "$work/lingbot/smoke_verified.json" ]] || { echo "smoke checkpoint not verified" >&2; exit 1; }
[[ ! -e "$work/lingbot/checkpoints" ]] || { echo "training checkpoint directory exists; refusing overwrite" >&2; exit 1; }
exec "$torchrun" --standalone --nproc_per_node=4 "$work/scripts/train_lingbot_clean.py" \
  "$work/lingbot/config.yaml"
