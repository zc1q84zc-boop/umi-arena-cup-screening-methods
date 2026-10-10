#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "$0")" && pwd)
console_root=$(cd -- "$task_root/../.." && pwd)
runtime=$(cd -- "$console_root/../dual-franka-yubi-isaac-sim-deploy" && pwd)
previous_task="$console_root/deployment_diagnostics/jaw_margin_20261010"
repo="$console_root/simulator_profiles/tuned_v1"
output="$runtime/runs/jaw_margin005_sequential_home_20261010_v2"
test ! -e "$output"
export CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0
export PATH="$previous_task/bin:$runtime/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/cpython-3.11.15-linux-x86_64-gnu/lib:$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$previous_task/isolated_runtime_state_v2/cache"
export XDG_CONFIG_HOME="$previous_task/isolated_runtime_state_v2/config"
export XDG_DATA_HOME="$previous_task/isolated_runtime_state_v2/data"
export TMPDIR="$previous_task/isolated_runtime_state_v2/tmp"
export UMI_CUP_MODEL=pvc_shell_e3000mpa_i128_h240_v2 UMI_AUDIT_CUP_CONTACTS=0
export UMI_EXECUTE_30HZ=1 UMI_ONLINE_CALIBRATION=tuned_online_v1
export UMI_ONLINE_RESPONSE_GAIN=4 UMI_ONLINE_JAW_RESPONSE_GAIN=4
export INTERSECTION_ONLINE_URL=http://127.0.0.1:18861/infer
export SIM_ADAPTER_AUDIT_DIR="$output" PYTHONUNBUFFERED=1
unset UMI_REPLAY_PATH UMI_JAW_BIAS_RAD UMI_REFERENCE_DIAGNOSTIC UMI_REFERENCE_FLANGE_PLUS90
unset UMI_PI05_RIGHT_TRANSLATION_GAIN UMI_LEFT_SECOND_OFFSET_MM UMI_LEFT_SECOND_HEIGHT_MM
unset YUBI_CANONICAL_HAND_SIDES UMI_PREGRASP_APPROACH_M UMI_PREGRASP_ADDITIONAL_M
cd "$repo"
exec "$previous_task/runtime_venv/bin/python" "$task_root/full_task_sequential.py" \
  --headless --scene dual_franka_yubi_official_fingertip_friction_trial \
  --setup online_aligned_v2 --seed 42 --until-success --task-objective plate_return \
  --trajectory-policy-script "$repo/adapters/pi05_intersection_isaac_online_adapter.py" \
  --online-chunk-30hz --policy-images all --camera head --record-wrists \
  --head-camera-calibration "$repo/yubi_isaac_sim_env/head_camera_online_aligned_v2.json" \
  --stop-file "$task_root/stop" --record-run "$output"
