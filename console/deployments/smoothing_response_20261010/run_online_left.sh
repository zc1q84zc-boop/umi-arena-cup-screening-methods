#!/usr/bin/env bash
set -euo pipefail
runtime=/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy
task_root=/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/deployment_diagnostics/smoothing_response_20261010
output="$runtime/runs/smoothing_response_20261010_online_gain8"
test ! -e "$output"
model_unit=umi-intersection-pi05-30000-4090-console.service
model_pid=$(systemctl --user show "$model_unit" -p MainPID --value)
[[ $model_pid =~ ^[1-9][0-9]*$ ]]
"$runtime/.venv/bin/python" - "$model_pid" <<'PY'
from pathlib import Path
import sys,json,urllib.request
cmd=Path('/proc/'+sys.argv[1]+'/cmdline').read_bytes().decode().split('\0')
assert '/home/claude/workspace/umi_cup_intersection_models_4090_20261009/scripts/intersection_server.py' in cmd
assert 'pi05-cup-intersection-30000' in cmd
health=json.load(urllib.request.urlopen('http://127.0.0.1:18861/health',timeout=5))
assert health['model']=='pi05-cup-intersection-30000'
PY
export CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0
export PATH="$runtime/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config" XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp"
export UMI_CUP_MODEL=pvc_shell_e3000mpa_i128_h240_v2 UMI_AUDIT_CUP_CONTACTS=0 UMI_EXECUTE_30HZ=1 UMI_ONLINE_CALIBRATION=tuned_online_v1
export INTERSECTION_ONLINE_URL=http://127.0.0.1:18861/infer
unset UMI_REPLAY_PATH UMI_JAW_BIAS_RAD UMI_REFERENCE_DIAGNOSTIC UMI_REFERENCE_FLANGE_PLUS90 UMI_PI05_RIGHT_TRANSLATION_GAIN
cd /home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1
exec "$runtime/.venv/bin/python" "$task_root/online_left.py" --gain 8 --output "$output"
