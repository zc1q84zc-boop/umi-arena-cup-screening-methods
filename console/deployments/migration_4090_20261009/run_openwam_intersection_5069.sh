#!/usr/bin/env bash
set -euo pipefail
[[ -z $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'Inference GPU1 occupied' >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
model_root=/home/claude/workspace/umi_cup_models_4090_20261009
intersection_root=/home/claude/workspace/umi_cup_intersection_models_4090_20261009
export PYTHONPATH="$model_root/openwam/runtime/site_packages:$model_root/runtime/lingbot_py312_site_packages:$intersection_root/source/OpenWAM:$model_root/openwam/scripts"
export TORCH_EXTENSIONS_DIR="$intersection_root/runtime/torch_extensions"
exec "$model_root/runtime/lingbot_venv/bin/python" "$intersection_root/scripts/intersection_4090_cpu_text_server.py" \
  --family openwam --checkpoint "$intersection_root/openwam/5069" --model-id openwam-cup-intersection-fullpass-5069 \
  --port 18862 --openwam-root "$intersection_root/source/OpenWAM"
