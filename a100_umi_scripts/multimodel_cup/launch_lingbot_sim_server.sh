#!/usr/bin/env bash
set -euo pipefail

step="${1:-5000}"
if [[ "$step" != 5000 && "$step" != 10000 ]]; then
  echo "usage: launch_lingbot_sim_server.sh 5000|10000" >&2
  exit 2
fi
work=/mnt/data/benyun/workspace/umi_cup_multimodel_20260925
repo=/mnt/data/benyun/workspace/projects/lingbot-vla-v2
py=/home/benyun/miniconda3/envs/lingbotvla/bin/python
checkpoint="$work/lingbot/checkpoints/checkpoints/global_step_$step/hf_ckpt"

exec 9>"$work/lingbot/sim_online.lock"
flock -n 9 || { echo "LingBot simulator service already running" >&2; exit 1; }
"$py" "$work/scripts/gpu_guard.py" --gpu 7
[[ -s "$checkpoint/model.safetensors.index.json" ]] || {
  echo "LingBot HF checkpoint index missing" >&2; exit 1;
}
for shard in 1 2 3; do
  printf -v name 'model-%05d-of-00003.safetensors' "$shard"
  [[ -s "$checkpoint/$name" ]] || { echo "HF shard missing: $name" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="$repo:$repo/deploy:$work/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
cd "$work"
exec nice -n 19 ionice -c 3 "$py" "$work/scripts/lingbot_sim_server.py" \
  --checkpoint "$checkpoint" --lingbot-root "$repo" --config-root "$work" \
  --model-id "lingbot-cup-clean-$step" --port 18784
