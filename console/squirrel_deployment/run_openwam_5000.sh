#!/usr/bin/env bash
set -euo pipefail

model_root=/home/lrl/workspace/umi_cup_models_5090_20260928
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$model_root/openwam/runtime/site_packages:$model_root/runtime/lingbot_py312_site_packages:$model_root/openwam/source/OpenWAM:$model_root/openwam/scripts"
export TORCH_EXTENSIONS_DIR="$model_root/openwam/runtime/torch_extensions"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

cd "$model_root"
exec "$model_root/runtime/lingbot_venv/bin/python" \
  "$model_root/openwam/scripts/openwam_sim_server.py" \
  --checkpoint-dir "$model_root/openwam/5000" \
  --checkpoint-name checkpoint_step_5000.safetensors \
  --prepared "$model_root/openwam/prepared" \
  --openwam-root "$model_root/openwam/source/OpenWAM" \
  --model-id openwam-cup-clean-5000 \
  --port 18812
