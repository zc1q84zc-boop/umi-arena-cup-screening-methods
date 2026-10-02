#!/usr/bin/env bash
set -euo pipefail

model_root=/home/lrl/workspace/umi_cup_models_5090_20260928
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$model_root/runtime/lingbot_py312_site_packages:$model_root/source/lingbot-vla-v2:$model_root/source/lingbot-vla-v2/deploy"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export QWEN3VL_PATH="$model_root/source/Qwen3-VL-4B-Instruct"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

cd "$model_root"
exec "$model_root/runtime/lingbot_venv/bin/python" \
  "$model_root/lingbot/scripts/lingbot_sim_server.py" \
  --checkpoint "$model_root/lingbot/10000/hf_ckpt" \
  --lingbot-root "$model_root/source/lingbot-vla-v2" \
  --config-root "$model_root" \
  --model-id lingbot-cup-clean-10000 \
  --port 18811
