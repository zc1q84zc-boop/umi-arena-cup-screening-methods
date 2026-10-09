#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
step="${2:-5000}"
if [[ "$mode" != smoke && "$mode" != resume ]]; then
  echo "usage: launch_lingbot_replay.sh smoke|resume [5000|10000]" >&2
  exit 2
fi
if [[ "$step" != 5000 && "$step" != 10000 ]]; then
  echo "checkpoint step must be 5000 or 10000" >&2
  exit 2
fi

work=/mnt/data/benyun/workspace/umi_cup_multimodel_20260925
repo=/mnt/data/benyun/workspace/projects/lingbot-vla-v2
py=/home/benyun/miniconda3/envs/lingbotvla/bin/python
checkpoint="$work/lingbot/checkpoints/checkpoints/global_step_$step/hf_ckpt"
output="$work/replays/lingbot-cup-clean-$step-causal"

exec 9>"$work/replays/lingbot-$step.lock"
flock -n 9 || { echo "LingBot replay already running" >&2; exit 1; }
"$py" "$work/scripts/gpu_guard.py" --gpu 6
[[ -f "$checkpoint/config.json" && -f "$checkpoint/model.safetensors.index.json" ]] || {
  echo "HF checkpoint config/index missing" >&2; exit 1;
}
for shard in 1 2 3; do
  printf -v name 'model-%05d-of-00003.safetensors' "$shard"
  [[ -s "$checkpoint/$name" ]] || { echo "HF checkpoint shard missing: $name" >&2; exit 1; }
done

if [[ "$mode" == smoke ]]; then
  [[ ! -e "$output" ]] || { echo "replay output exists; refusing overwrite" >&2; exit 1; }
  extra=(--limit-episodes 1)
else
  [[ -f "$output/report.json" ]] || { echo "smoke report missing" >&2; exit 1; }
  extra=(--resume)
fi

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="$work/evaluation:$repo:$repo/deploy:$work/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

cd "$work"
exec nice -n 19 ionice -c 3 "$py" "$work/scripts/replay_lingbot_cup.py" \
  --evaluation-root "$work/evaluation" \
  --lingbot-root "$repo" \
  --dataset /mnt/data/benyun/workspace/yubi-corl2026-umi-arena \
  --suite "$work/evaluation/suites/practice-cup-smartphone.json" \
  --train-manifest "$work/selection/manifest.json" \
  --hf-checkpoint "$checkpoint" \
  --robot-config-root "$work/configs" \
  --checkpoint-id "lingbot-cup-clean-$step" \
  --output "$output" "${extra[@]}"
