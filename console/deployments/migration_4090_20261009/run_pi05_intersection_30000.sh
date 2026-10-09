#!/usr/bin/env bash
set -euo pipefail
[[ -z $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'Inference GPU1 occupied' >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
model_root=/home/claude/workspace/umi_cup_models_4090_20261009
intersection_root=/home/claude/workspace/umi_cup_intersection_models_4090_20261009
export OPENPI_DATA_HOME="$model_root/runtime/openpi_cache" HF_HOME="$model_root/runtime/huggingface_cache"
export PYTHONPATH="$model_root/runtime/pi05_compat:$model_root/runtime/pi05_py311_site_packages:$intersection_root/source/evaluation:$model_root/source/openpi/src:$model_root/source/openpi/packages/openpi-client/src"
exec "$model_root/runtime/pi05_venv/bin/python" "$intersection_root/scripts/intersection_server.py" \
  --family pi05 --checkpoint "$intersection_root/pi05/30000" --model-id pi05-cup-intersection-30000 \
  --port 18861 --evaluation-root "$intersection_root/source/evaluation" --openpi-root "$model_root/source/openpi"
