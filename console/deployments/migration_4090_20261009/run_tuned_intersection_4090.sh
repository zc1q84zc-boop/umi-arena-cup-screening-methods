#!/usr/bin/env bash
set -euo pipefail
# Registered model servers. No private demonstration data or oracle grasp offset.
repo=/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1
runtime=/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy
run_dir=${1:?new console output directory}
adapter=${2:?model adapter basename}
stop_file=${3:?dedicated console stop marker}
steps=${4:?positive request count or until-success}
task_objective=${5:-plate_return}
diagnostic_mode=${6:-baseline}
extra_closure_fraction=${7:-0}
contact_profile=${8:-baseline}
scene_name=dual_franka_yubi_cup40k_cupfriction_trial
cup_model=rigid
case "$contact_profile" in
  baseline) ;;
  official_fingertip_friction)
    [[ $diagnostic_mode == baseline && $steps =~ ^[1-9][0-9]*$ && $steps -le 600 ]] || exit 2
    scene_name=dual_franka_yubi_official_fingertip_friction_trial ;;
  pvc_shell_e3000mpa_i128_h240_v2)
    [[ $diagnostic_mode == baseline ]] || exit 2
    [[ $steps == until-success || ( $steps =~ ^[1-9][0-9]*$ && $steps -le 10000 ) ]] || exit 2
    scene_name=dual_franka_yubi_official_fingertip_friction_trial
    cup_model=$contact_profile ;;
  pvc_elastic_shell_v1|pvc_shell_e3000mpa_v1|pvc_shell_e2000mpa_v1|pvc_shell_e1000mpa_v1|pvc_shell_e0500mpa_v1|pvc_shell_e0200mpa_v1)
    [[ $diagnostic_mode == baseline && $steps =~ ^[1-9][0-9]*$ && $steps -le 600 ]] || exit 2
    scene_name=dual_franka_yubi_official_fingertip_friction_trial
    cup_model=$contact_profile ;;
  *) echo 'Unknown contact profile' >&2; exit 2 ;;
esac
[[ $extra_closure_fraction =~ ^0(\.[0-9]{1,6})?$ ]] || exit 2
"$runtime/.venv/bin/python" -c 'import sys; assert 0 <= float(sys.argv[1]) <= .05' "$extra_closure_fraction" || exit 2
setup_name=online_aligned_v2
diagnostic_args=()
case "$diagnostic_mode" in
  baseline) [[ $extra_closure_fraction == 0 ]] || exit 2 ;;
  left-return)
    [[ $adapter == pi05_isaac_online_adapter.py && ${UMI_MODEL_UNIT:-} == umi-squirrel-pi05-30000-console.service
       && $task_objective == plate_return && $steps =~ ^[1-9][0-9]*$ && $steps -le 600 ]] || exit 2
    setup_name=left_return_diagnostic
    diagnostic_args=(--left-return-diagnostic --left-extra-closure-fraction "$extra_closure_fraction")
    ;;
  *) exit 2 ;;
esac
[[ $task_objective == plate || $task_objective == plate_return ]] || exit 2
[[ $run_dir =~ ^/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/runs/console_[0-9a-f]{12}$ ]] || exit 2
[[ $stop_file =~ ^/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/runs/\.stop_[0-9a-f]{12}$ ]] || exit 2
[[ $adapter == pi05_intersection_10000_isaac_online_adapter.py || $adapter == pi05_intersection_20000_isaac_online_adapter.py || $adapter == pi05_intersection_isaac_online_adapter.py || $adapter == openwam_intersection_isaac_online_adapter.py ]] || exit 2
test ! -e "$run_dir" || { echo 'Output already exists' >&2; exit 2; }
case "${UMI_MODEL_UNIT:-}" in
  umi-intersection-pi05-10000-4090-console.service|umi-intersection-pi05-20000-4090-console.service|umi-intersection-pi05-30000-4090-console.service|umi-intersection-openwam-5069-4090-console.service) ;;
  *) echo 'Missing known dedicated model unit' >&2; exit 2 ;;
esac
model_pid=$(systemctl --user show "$UMI_MODEL_UNIT" -p MainPID --value)
[[ $model_pid =~ ^[1-9][0-9]*$ ]] || { echo 'Dedicated model server is not active' >&2; exit 2; }
command_line=$(tr '\0' ' ' < "/proc/$model_pid/cmdline")
[[ $command_line == *'/home/claude/workspace/umi_cup_intersection_models_4090_20261009/'* && $command_line == *'server.py'* ]] || {
  echo 'Model PID identity does not match private inference workspace' >&2; exit 2;
}
sim_pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
[[ -z $sim_pids ]] || { echo 'Simulation GPU0 is occupied; refusing launch' >&2; exit 2; }
compute_pids=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)
while IFS= read -r pid; do
  [[ -z $pid || $pid == "$model_pid" ]] || { echo "Inference GPU1 has another compute PID $pid" >&2; exit 2; }
done <<< "$compute_pids"
if [[ $steps == until-success ]]; then
  step_args=(--until-success)
  runtime_limits=()
else
  [[ $steps =~ ^[1-9][0-9]*$ ]] || exit 2
  step_args=(--steps "$steps")
  runtime_limits=(--property=RuntimeMaxSec=1800)
fi
umask 077
export CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0
export PATH="$runtime/.venv/bin:$PATH" LD_LIBRARY_PATH="$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config" XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp"
export SIM_ADAPTER_AUDIT_DIR="$run_dir" UMI_ONLINE_CALIBRATION=tuned_online_v1
export UMI_EXECUTE_30HZ=1 UMI_REQUIRE_CAUSAL_PI05=1
unset UMI_REPLAY_PATH UMI_JAW_BIAS_RAD UMI_LEFT_SECOND_OFFSET_MM UMI_LEFT_SECOND_HEIGHT_MM
unset UMI_REFERENCE_DIAGNOSTIC UMI_REFERENCE_FLANGE_PLUS90 YUBI_CANONICAL_HAND_SIDES
unset UMI_PREGRASP_APPROACH_M UMI_PREGRASP_ADDITIONAL_M

# A full precision-cup PI05 task uses the recorded right-home-before-left
# sequencing and the user's 0.005 rad left closure margin.
eval_id=''
case "$UMI_MODEL_UNIT" in
  umi-squirrel-pi05-10000-console.service) eval_id=pi05-cup-clean-10000 ;;
  umi-squirrel-pi05-20000-console.service) eval_id=pi05-cup-clean-20000 ;;
  umi-squirrel-pi05-30000-console.service) eval_id=pi05-cup-clean-30000 ;;
  umi-intersection-pi05-10000-4090-console.service) eval_id=pi05-cup-intersection-10000 ;;
  umi-intersection-pi05-20000-4090-console.service) eval_id=pi05-cup-intersection-20000 ;;
  umi-intersection-pi05-30000-4090-console.service) eval_id=pi05-cup-intersection-30000 ;;
esac
entry=("$runtime/.venv/bin/python" -m yubi_isaac_sim_env.run_visual_aligned)
response_gain=4; velocity=.8; acceleration=1.5; warmup=5
if [[ -n $eval_id && $cup_model == pvc_shell_e3000mpa_i128_h240_v2 && $diagnostic_mode == baseline && $task_objective == plate_return ]]; then
  entry=("$runtime/.venv/bin/python" "$repo/../../deployment_diagnostics/fast_parallel_pi05_20261010/full_task_sequential.py")
  response_gain=6; velocity=1.2; acceleration=2.4; warmup=1
fi
eval_url="${PI05_ONLINE_URL:-http://127.0.0.1:18783/infer}"
if [[ $eval_id == pi05-cup-intersection-* ]]; then eval_url="${INTERSECTION_ONLINE_URL:?intersection endpoint}"; fi

cd "$repo"
run_id=${run_dir##*console_}
runtime_unit="umi-tuned-runtime-$run_id.service"
set +e
systemd-run --user --wait --unit="$runtime_unit" --property=TimeoutStopSec=10 "${runtime_limits[@]}" \
  --working-directory="$repo" env \
  -u UMI_REPLAY_PATH -u UMI_JAW_BIAS_RAD -u UMI_LEFT_SECOND_OFFSET_MM -u UMI_LEFT_SECOND_HEIGHT_MM \
  -u UMI_REFERENCE_DIAGNOSTIC -u UMI_REFERENCE_FLANGE_PLUS90 -u YUBI_CANONICAL_HAND_SIDES \
  -u UMI_PREGRASP_APPROACH_M -u UMI_PREGRASP_ADDITIONAL_M -u UMI_PI05_RIGHT_TRANSLATION_GAIN \
  CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0 PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  XDG_CACHE_HOME="$XDG_CACHE_HOME" XDG_CONFIG_HOME="$XDG_CONFIG_HOME" XDG_DATA_HOME="$XDG_DATA_HOME" TMPDIR="$TMPDIR" \
  SIM_ADAPTER_AUDIT_DIR="$run_dir" UMI_ONLINE_CALIBRATION=tuned_online_v1 \
  UMI_CUP_MODEL="$cup_model" \
  UMI_ONLINE_RESPONSE_GAIN="$response_gain" UMI_ONLINE_JAW_RESPONSE_GAIN=4 \
  UMI_ONLINE_VELOCITY_RAD_S="$velocity" UMI_ONLINE_ACCELERATION_RAD_S2="$acceleration" \
  UMI_CAMERA_WARMUP_FRAMES="$warmup" UMI_SIM_CPU_THREADS=8 \
  UMI_EVAL_MODEL_ID="$eval_id" UMI_EVAL_MODEL_UNIT="$UMI_MODEL_UNIT" UMI_EVAL_MODEL_URL="$eval_url" UMI_EVAL_MODEL_GPU=1 \
  UMI_EVAL_CONSOLE_LEASE=1 UMI_EVAL_STOP_FILE="$stop_file" \
  UMI_EXECUTE_30HZ=1 UMI_REQUIRE_CAUSAL_PI05=1 UMI_AUDIT_CUP_CONTACTS=1 UMI_REFERENCE_LIGHT_INTENSITY=2200 UMI_REFERENCE_DARK_FINGERS=1 \
  PI05_ONLINE_URL="${PI05_ONLINE_URL:-http://127.0.0.1:18783/infer}" \
  LINGBOT_ONLINE_URL="${LINGBOT_ONLINE_URL:-http://127.0.0.1:18811/infer}" \
  OPENWAM_ONLINE_URL="${OPENWAM_ONLINE_URL:-http://127.0.0.1:18813/infer}" \
  INTERSECTION_ONLINE_URL="${INTERSECTION_ONLINE_URL:?new model endpoint}" \
  "${entry[@]}" --headless \
  --scene "$scene_name" --setup "$setup_name" --seed 42 \
  --trajectory-policy-script "$repo/adapters/$adapter" --online-chunk-30hz --policy-images all \
  --camera head --record-wrists --head-camera-calibration yubi_isaac_sim_env/head_camera_online_aligned_v2.json \
  --task-objective "$task_objective" --stop-file "$stop_file" --record-run "$run_dir" "${step_args[@]}" "${diagnostic_args[@]}"
runtime_status=$?
set -e
if [[ $runtime_status != 0 ]]; then
  # Kit's telemetry child can ignore TERM after Python exits. Do not confuse
  # bounded cleanup of this exact unit with model/simulator failure. Accept
  # only a successful main process, a flushed terminal report and an empty
  # cgroup; every other nonzero result still fails closed.
  main_status=$(systemctl --user show "$runtime_unit" -p ExecMainStatus --value)
  main_code=$(systemctl --user show "$runtime_unit" -p ExecMainCode --value)
  result=$(systemctl --user show "$runtime_unit" -p Result --value)
  group=$(systemctl --user show "$runtime_unit" -p ControlGroup --value)
  if [[ $main_status == 0 && $main_code == 1 && $result == timeout ]] && \
     [[ -z $group || ! -s "/sys/fs/cgroup$group/cgroup.procs" ]] && \
     "$runtime/.venv/bin/python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] in ("completed","stopped")' "$run_dir/report.json"; then
    echo 'Simulator main exited 0; task-owned telemetry cleanup completed within the unit timeout.'
    runtime_status=0
  fi
fi
systemctl --user reset-failed "$runtime_unit" >/dev/null 2>&1 || true
exit "$runtime_status"
