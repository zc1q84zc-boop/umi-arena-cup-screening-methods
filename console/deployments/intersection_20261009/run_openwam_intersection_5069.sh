#!/usr/bin/env bash
set -euo pipefail
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'GPU occupied; leave existing process running' >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/lrl/workspace/umi_cup_models_5090_20260928/openwam/runtime/site_packages:/home/lrl/workspace/umi_cup_models_5090_20260928/runtime/lingbot_py312_site_packages:/home/lrl/workspace/umi_cup_intersection_models_20261009/source/OpenWAM"
export TORCH_EXTENSIONS_DIR="/home/lrl/workspace/umi_cup_intersection_models_20261009/runtime/torch_extensions"
exec /home/lrl/workspace/umi_cup_models_5090_20260928/runtime/lingbot_venv/bin/python /home/lrl/workspace/umi_cup_intersection_models_20261009/scripts/intersection_server.py --family openwam --checkpoint /home/lrl/workspace/umi_cup_intersection_models_20261009/openwam/5069 --model-id openwam-cup-intersection-fullpass-5069 --port 18862 --openwam-root /home/lrl/workspace/umi_cup_intersection_models_20261009/source/OpenWAM
