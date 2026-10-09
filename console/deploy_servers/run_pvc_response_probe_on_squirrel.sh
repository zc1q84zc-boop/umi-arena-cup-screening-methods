#!/usr/bin/env bash
set -euo pipefail
repo=/home/lrl/dual-franka-yubi-isaac-sim-console-tuned-v1
runtime=/home/lrl/dual-franka-yubi-isaac-sim-deploy
name=${1:?new diagnostic name}
youngs=${2:?1 3 or 6 GPa}
iterations=${3:?32 128 or 255}
hz=${4:?120 or 240}
thickness=${5:-1}
[[ $name =~ ^pvc_response_[a-z0-9_]{1,48}$ ]] || exit 2
[[ $youngs == 1 || $youngs == 3 || $youngs == 6 ]] || exit 2
[[ $iterations == 32 || $iterations == 128 || $iterations == 255 ]] || exit 2
[[ $hz == 120 || $hz == 240 ]] || exit 2
[[ $thickness == 1 || $thickness == 1.5 ]] || exit 2
[[ $(hostname) == slzl-System-Product-Name && $(id -un) == lrl ]] || exit 2
test ! -e "$runtime/runs/$name" || { echo 'Diagnostic output exists; no overwrite' >&2; exit 2; }
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || {
  echo 'GPU0 occupied; preserve existing jobs' >&2; exit 3;
}
unit="umi-${name//_/-}.service"
[[ $(systemctl --user show "$unit" -p MainPID --value) == 0 ]] || exit 3
systemd-run --user --unit="$unit" --property=RuntimeMaxSec=300 --property=TimeoutStopSec=10 \
  --working-directory="$repo" env CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0 \
  PATH="$runtime/.venv/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$runtime/.venv/lib" \
  XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config" \
  XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp" \
  flock -n "$runtime/console.lock" bash -c \
  'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3; exec "$1" -m yubi_isaac_sim_env.pvc_response_probe --output "$2" --youngs-gpa "$3" --iterations "$4" --hz "$5" --thickness-mm "$6"' \
  pvc-response-probe "$runtime/.venv/bin/python" "$runtime/runs/$name" "$youngs" "$iterations" "$hz" "$thickness"
