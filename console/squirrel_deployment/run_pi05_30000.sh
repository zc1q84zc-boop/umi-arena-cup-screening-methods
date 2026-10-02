#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=/home/lrl/workspace/umi_cup_models_5090_20260928
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$MODEL_ROOT/runtime/pi05_compat:$MODEL_ROOT/runtime/pi05_py311_site_packages:$MODEL_ROOT/source/umi-arena-evaluation:$MODEL_ROOT/source/umi-arena-evaluation/adapters/openpi_pi05:$MODEL_ROOT/source/openpi/src:$MODEL_ROOT/source/openpi/packages/openpi-client/src"
export HF_HOME=/home/lrl/.cache/huggingface

exec "$MODEL_ROOT/runtime/pi05_venv/bin/python" \
  "$MODEL_ROOT/pi05/pi05_online_server.py" \
  --evaluation-root "$MODEL_ROOT/source/umi-arena-evaluation" \
  --openpi-root "$MODEL_ROOT/source/openpi" \
  --checkpoint "$MODEL_ROOT/pi05/30000/inference_export" \
  --model-id pi05-cup-clean-30000 \
  --port 18783
