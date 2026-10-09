#!/usr/bin/env bash
# Run on squirrel_5090 or openwam-a100. Each candidate is a fresh process; failed candidates
# cannot leave optimizer state or CUDA allocations in subsequent measurements.
set -euo pipefail
profile="${BENCHMARK_HOST_PROFILE:-5090}"
if [[ "$profile" == 5090 ]]; then
  model_root=/home/lrl/workspace/umi_cup_models_5090_20260928
  bench_root="$model_root/benchmarks/openwam_batch_20261007"
  py="$model_root/runtime/lingbot_venv/bin/python"
  repo="$model_root/openwam/source/OpenWAM"
  checkpoint="$model_root/openwam/10000"
  gpu="${BENCHMARK_GPU_INDEX:-0}"
  export PYTHONPATH="$model_root/openwam/runtime/site_packages:$model_root/runtime/lingbot_py312_site_packages:$repo:$bench_root/scripts"
  export CUDA_HOME=/usr/local/cuda
  # CPU Adam contains C++ sources only. The 5090 host has CUDA 13.2 toolkit
  # and the existing PyTorch CUDA 12.8 runtime; no CUDA kernel is JIT compiled
  # by CPU Adam. Keep this compatibility setting scoped to the benchmark.
  export DS_SKIP_CUDA_CHECK=1
elif [[ "$profile" == a100 ]]; then
  work=/mnt/data/benyun/workspace/umi_cup_multimodel_20260925
  old=/mnt/data/benyun/workspace/openwam_charger_full_20260922
  bench_root="$work/openwam/batch_benchmark_20261007"
  py="$old/.venv/bin/python"
  repo="$old/OpenWAM"
  checkpoint="$work/openwam/checkpoints/2026-09-25_18-55-32"
  gpu="${BENCHMARK_GPU_INDEX:?set BENCHMARK_GPU_INDEX to an available A100 index from 1 to 7}"
  [[ "$gpu" =~ ^[1-7]$ ]] || { echo 'A100 GPU 0 is reserved for other users' >&2; exit 1; }
  export PYTHONPATH="$bench_root/scripts:$repo"
  export CUDA_HOME=/usr/local/cuda-12.8
else
  echo 'BENCHMARK_HOST_PROFILE must be 5090 or a100' >&2
  exit 2
fi
batch="${1:?usage: run_openwam_batch_benchmark.sh BATCH [ACCUM [WARMUP [TIMED]]]}"
accum="${2:-1}"
warmup="${3:-3}"
timed="${4:-12}"
run_id="b${batch}_a${accum}_$(date +%Y%m%d_%H%M%S)"
exec 9>"$bench_root/gpu${gpu}.lock"
flock -n 9 || { echo 'another benchmark owns this GPU lock' >&2; exit 1; }
gpu_uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits)"
active="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F, -v uuid="$gpu_uuid" '$1 == uuid {print $2}')"
[[ -z "$active" ]] || { echo 'GPU has an existing compute process; refusing overlap' >&2; exit 1; }
if [[ "$profile" == a100 ]]; then
  "$py" "$work/scripts/gpu_guard.py" --gpu "$gpu"
  exec 8>"$work/openwam/training.lock"
  flock -n 8 || { echo 'an OpenWAM training launcher owns the lock' >&2; exit 1; }
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$bench_root/torch_extensions"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$bench_root/results"
printf '%s\n' "$run_id" > "$bench_root/current_run.txt"
cd "$bench_root"
exec "$py" -m torch.distributed.run \
  --standalone --nproc_per_node=1 "$bench_root/scripts/benchmark_openwam_cup.py" \
  --repo "$repo" --scripts "$bench_root/scripts" \
  --prepared "$bench_root/prepared" --checkpoint "$checkpoint" \
  --output "$bench_root/results/$run_id" --batch "$batch" --accum "$accum" \
  --warmup "$warmup" --timed "$timed"
