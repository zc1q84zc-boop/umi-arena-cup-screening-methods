#!/usr/bin/env bash
set -euo pipefail
kind=${1:?frozen or live}
margin=${2:?closure radians}
label=${3:?unique output label}
[[ $kind == frozen || $kind == live ]] || exit 2
[[ $label =~ ^[a-z0-9_]+$ ]] || exit 2
runtime=/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy
task_root=/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/deployment_diagnostics/jaw_margin_20261010
output="$runtime/runs/jaw_margin_20261010_$label"
test ! -e "$output"
if [[ $kind == live ]]; then
  model_unit=umi-jaw-margin-pi05-20261010.service
  model_pid=$(systemctl --user show "$model_unit" -p MainPID --value)
  [[ $model_pid =~ ^[1-9][0-9]*$ ]]
  "$task_root/runtime_venv/bin/python" - "$model_pid" <<'PY'
from pathlib import Path
import sys,json,urllib.request
cmd=Path('/proc/'+sys.argv[1]+'/cmdline').read_bytes().decode().split('\0')
assert '/home/claude/workspace/umi_cup_intersection_models_4090_20261009/scripts/intersection_server.py' in cmd
assert 'pi05-cup-intersection-30000' in cmd
health=json.load(urllib.request.urlopen('http://127.0.0.1:18863/health',timeout=5))
assert health['model']=='pi05-cup-intersection-30000'
PY
fi
export CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0
export PATH="$task_root/bin:$runtime/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/cpython-3.11.15-linux-x86_64-gnu/lib:$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$task_root/isolated_runtime_state_v2/cache" XDG_CONFIG_HOME="$task_root/isolated_runtime_state_v2/config" XDG_DATA_HOME="$task_root/isolated_runtime_state_v2/data" TMPDIR="$task_root/isolated_runtime_state_v2/tmp"
export UMI_CUP_MODEL=pvc_shell_e3000mpa_i128_h240_v2 UMI_AUDIT_CUP_CONTACTS=0 UMI_EXECUTE_30HZ=1 UMI_ONLINE_CALIBRATION=tuned_online_v1
export UMI_ONLINE_RESPONSE_GAIN=4 UMI_ONLINE_JAW_RESPONSE_GAIN=4
export INTERSECTION_ONLINE_URL=http://127.0.0.1:18863/infer
unset UMI_REPLAY_PATH UMI_JAW_BIAS_RAD UMI_REFERENCE_DIAGNOSTIC UMI_REFERENCE_FLANGE_PLUS90 UMI_PI05_RIGHT_TRANSLATION_GAIN
cd /home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1
if [[ $kind == frozen ]]; then
  exec "$task_root/runtime_venv/bin/python" "$task_root/frozen_margin.py" --gain 4 --jaw-gain 4 --output "$output" --phase-reset --margin "$margin"
else
  exec "$task_root/runtime_venv/bin/python" "$task_root/live_margin.py" --gain 4 --output "$output" --margin "$margin"
fi
