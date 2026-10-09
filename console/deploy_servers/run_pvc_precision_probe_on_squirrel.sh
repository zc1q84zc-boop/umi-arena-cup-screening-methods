#!/usr/bin/env bash
set -euo pipefail
repo=/home/lrl/dual-franka-yubi-isaac-sim-console-tuned-v1
runtime=/home/lrl/dual-franka-yubi-isaac-sim-deploy
name=${1:?new diagnostic name}
profile=${2:?exact stiffness profile}
[[ $name =~ ^pvc_precision_[a-z0-9_]{1,48}$ ]] || exit 2
case "$profile" in
  pvc_shell_e3000mpa_i128_h240_v2) ;;
  *) exit 2 ;;
esac
[[ $(hostname) == slzl-System-Product-Name && $(id -un) == lrl ]] || exit 2
test ! -e "$runtime/runs/$name" || { echo 'Diagnostic output exists; do not overwrite' >&2; exit 2; }
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || {
  echo 'GPU0 is occupied; leave existing jobs untouched' >&2; exit 3;
}
unit="umi-${name//_/-}.service"
[[ $(systemctl --user show "$unit" -p MainPID --value) == 0 ]] || exit 3
systemd-run --user --unit="$unit" --property=RuntimeMaxSec=600 --property=TimeoutStopSec=10 \
  --working-directory="$repo" env CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0 \
  PATH="$runtime/.venv/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$runtime/.venv/lib" \
  XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config" \
  XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp" \
  flock -n "$runtime/console.lock" bash -c \
  'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3; exec "$1" -m yubi_isaac_sim_env.pvc_precision_probe --output "$2" --profile "$3"' \
  pvc-stiffness-probe "$runtime/.venv/bin/python" "$runtime/runs/$name" "$profile"
# This account has Linger=no. Keep the authorized SSH job alive rather than
# changing login policy; otherwise logind kills the user unit at last logout.
for ((check=0; check<330; check++)); do
  state=$(systemctl --user show "$unit" -p ActiveState --value)
  if [[ $state == inactive || $state == failed ]]; then
    test -f "$runtime/runs/$name/report.json" || exit 1
    "$runtime/.venv/bin/python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "completed"' "$runtime/runs/$name/report.json"
    test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3
    exit 0
  fi
  sleep 2
done
exit 124
