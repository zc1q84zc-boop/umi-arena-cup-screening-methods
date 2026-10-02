#!/usr/bin/env bash
set -euo pipefail
deploy_root=/home/lrl/dual-franka-yubi-isaac-sim-deploy
gpu_id=${1:?GPU index required}
[[ "$gpu_id" =~ ^[0-9]+$ ]] || exit 2
shift
export CUDA_VISIBLE_DEVICES="$gpu_id" ISAAC_ACTIVE_GPU="$gpu_id"
export PATH="$deploy_root/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$deploy_root/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$deploy_root/.cache" XDG_CONFIG_HOME="$deploy_root/.config"
export XDG_DATA_HOME="$deploy_root/.local" TMPDIR="$deploy_root/.tmp"
cd "$deploy_root/repo"
exec "$deploy_root/.venv/bin/python" -m yubi_isaac_sim_env.run_reference_diagnostic --headless "$@"
