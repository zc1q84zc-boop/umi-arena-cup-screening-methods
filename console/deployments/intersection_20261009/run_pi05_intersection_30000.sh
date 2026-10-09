#!/usr/bin/env bash
set -euo pipefail
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'GPU occupied; leave existing process running' >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="/home/lrl/workspace/umi_cup_models_5090_20260928/runtime/pi05_compat:/home/lrl/workspace/umi_cup_models_5090_20260928/runtime/pi05_py311_site_packages:/home/lrl/workspace/umi_cup_intersection_models_20261009/source/evaluation:/home/lrl/workspace/umi_cup_models_5090_20260928/source/openpi/src:/home/lrl/workspace/umi_cup_models_5090_20260928/source/openpi/packages/openpi-client/src"
exec /home/lrl/workspace/umi_cup_models_5090_20260928/runtime/pi05_venv/bin/python /home/lrl/workspace/umi_cup_intersection_models_20261009/scripts/intersection_server.py --family pi05 --checkpoint /home/lrl/workspace/umi_cup_intersection_models_20261009/pi05/30000 --model-id pi05-cup-intersection-30000 --port 18861 --evaluation-root /home/lrl/workspace/umi_cup_intersection_models_20261009/source/evaluation --openpi-root /home/lrl/workspace/umi_cup_models_5090_20260928/source/openpi
