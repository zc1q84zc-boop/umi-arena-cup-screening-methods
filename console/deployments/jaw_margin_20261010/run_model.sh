#!/usr/bin/env bash
set -euo pipefail
[[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader) ]] || exit 3
task_root=/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/deployment_diagnostics/jaw_margin_20261010
model_root=/home/claude/workspace/umi_cup_models_4090_20261009
intersection_root=/home/claude/workspace/umi_cup_intersection_models_4090_20261009
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME="$model_root/runtime/openpi_cache" HF_HOME="$model_root/runtime/huggingface_cache"
export PYTHONPATH="$model_root/runtime/pi05_compat:$model_root/runtime/pi05_py311_site_packages:$intersection_root/source/evaluation:$model_root/source/openpi/src:$model_root/source/openpi/packages/openpi-client/src"
export CUDA_ROOT="$model_root/runtime/pi05_py311_site_packages/nvidia/cuda_nvcc"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export TMPDIR="$task_root/model_tmp"
exec "$task_root/model_venv/bin/python" "$intersection_root/scripts/intersection_server.py" \
  --family pi05 --checkpoint "$intersection_root/pi05/30000" --model-id pi05-cup-intersection-30000 \
  --port 18863 --evaluation-root "$intersection_root/source/evaluation" --openpi-root "$model_root/source/openpi"
