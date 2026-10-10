#!/usr/bin/env bash
set -euo pipefail
run_root=${1:?Fresh screening run root required}
dataset=${2:-/mnt/data/benyun/workspace/yubi-corl2026-umi-arena}
python_path=${SCREENING_PYTHON:-/home/benyun/.venvs/umi_arena_pi05/bin/python}
scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
cd "$scripts"
exec 9>"$run_root/quality.lock"
flock -n 9 || exit 2
if [[ ! -e "$run_root/selection/manifest.json" ]]; then
  "$python_path" prepare_selection.py --dataset-root "$dataset" --output "$run_root"
fi
"$python_path" scan_quality.py --dataset-root "$dataset" --output "$run_root" --workers 2
