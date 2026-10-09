#!/usr/bin/env bash
set -euo pipefail
# Bounded stand-alone physics verification. No inference model is started.
repo=/home/lrl/dual-franka-yubi-isaac-sim-console-tuned-v1
runtime=/home/lrl/dual-franka-yubi-isaac-sim-deploy
name=${1:?new diagnostic name}
[[ $name =~ ^pvc_shell_platen_[a-z0-9_]{1,48}$ ]] || exit 2
[[ $(hostname) == slzl-System-Product-Name && $(id -un) == lrl ]] || exit 2
test ! -e "$runtime/runs/$name" || { echo 'Diagnostic output exists; do not overwrite' >&2; exit 2; }
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || {
  echo 'GPU0 is occupied; leave the existing job untouched' >&2; exit 3;
}
unit="umi-${name//_/-}.service"
[[ $(systemctl --user show "$unit" -p MainPID --value) == 0 ]] || exit 3
systemd-run --user --unit="$unit" --property=RuntimeMaxSec=600 --property=TimeoutStopSec=10 \
  --working-directory="$repo" env CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0 \
  PATH="$runtime/.venv/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$runtime/.venv/lib" \
  XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config" \
  XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp" \
  flock -n "$runtime/console.lock" bash -c \
  'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3; exec "$1" -m yubi_isaac_sim_env.pvc_shell_probe --output "$2"' \
  pvc-probe "$runtime/.venv/bin/python" "$runtime/runs/$name"
