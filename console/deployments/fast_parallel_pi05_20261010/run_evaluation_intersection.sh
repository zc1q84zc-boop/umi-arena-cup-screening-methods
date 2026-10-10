#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "$0")" && pwd)
console_root=$(cd -- "$task_root/../.." && pwd)
project=$(cd -- "$console_root/.." && pwd)
runtime="$project/dual-franka-yubi-isaac-sim-deploy"
previous="$console_root/deployment_diagnostics/jaw_margin_20261010"
repo="$console_root/simulator_profiles/tuned_v1"
model=${1:?model identifier}; gpu=${2:?simulation physical GPU}; mode=${3:-full}
case "$model" in
  10000) id=pi05-cup-intersection-10000; port=18873; inference_gpu=0; unit=umi-fast-intersection-pi05-10000-20261010.service; adapter=pi05_intersection_10000_isaac_online_adapter.py ;;
  20000) id=pi05-cup-intersection-20000; port=18874; inference_gpu=1; unit=umi-fast-intersection-pi05-20000-20261010.service; adapter=pi05_intersection_20000_isaac_online_adapter.py ;;
  *) exit 2 ;;
esac
inference_gpu=${UMI_INTERSECTION_INFERENCE_GPU:-$inference_gpu}
unset CUDA_VISIBLE_DEVICES
export ISAAC_ACTIVE_GPU="$gpu" UMI_ISAAC_CUDA_INDEX="$gpu"
export PATH="$previous/bin:$runtime/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/cpython-3.11.15-linux-x86_64-gnu/lib:$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
state="$task_root/runtime_intersection_${model}_${mode}"; mkdir -p "$state"/{cache,config,data,tmp}
export XDG_CACHE_HOME="$state/cache" XDG_CONFIG_HOME="$state/config" XDG_DATA_HOME="$state/data" TMPDIR="$state/tmp"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 UMI_SIM_CPU_THREADS=8
export UMI_CUP_MODEL=pvc_shell_e3000mpa_i128_h240_v2 UMI_AUDIT_CUP_CONTACTS=0
export UMI_EXECUTE_30HZ=1 UMI_ONLINE_CALIBRATION=tuned_online_v1 UMI_REQUIRE_CAUSAL_PI05=1
export UMI_ONLINE_RESPONSE_GAIN=6 UMI_ONLINE_JAW_RESPONSE_GAIN=4
export UMI_ONLINE_VELOCITY_RAD_S=1.2 UMI_ONLINE_ACCELERATION_RAD_S2=2.4
export UMI_CAMERA_WARMUP_FRAMES=1 UMI_REFERENCE_LIGHT_INTENSITY=2200 UMI_REFERENCE_DARK_FINGERS=1
export PI05_ONLINE_URL="http://127.0.0.1:$port/infer" INTERSECTION_ONLINE_URL="http://127.0.0.1:$port/infer"
export UMI_EVAL_MODEL_ID="$id" UMI_EVAL_MODEL_UNIT="$unit" UMI_EVAL_MODEL_URL="$PI05_ONLINE_URL" UMI_EVAL_MODEL_GPU="$inference_gpu"
export UMI_EVAL_PARALLEL_PROTECTED=1 PYTHONUNBUFFERED=1
output="$runtime/runs/fast_housing_pi05_intersection_${model}_${mode}_20261010"
export SIM_ADAPTER_AUDIT_DIR="$output" UMI_EVAL_STOP_FILE="$task_root/stop_intersection_${model}"
test ! -e "$output"
unset UMI_REPLAY_PATH UMI_JAW_BIAS_RAD UMI_REFERENCE_DIAGNOSTIC UMI_REFERENCE_FLANGE_PLUS90
unset UMI_PI05_RIGHT_TRANSLATION_GAIN UMI_LEFT_SECOND_OFFSET_MM UMI_LEFT_SECOND_HEIGHT_MM
unset YUBI_CANONICAL_HAND_SIDES UMI_PREGRASP_APPROACH_M UMI_PREGRASP_ADDITIONAL_M
cd "$repo"
if [[ $mode == smoke* ]]; then
  export UMI_CAMERA_WARMUP_FRAMES=${UMI_SMOKE_WARMUP_FRAMES:-1}
  exec "$previous/runtime_venv/bin/python" -m yubi_isaac_sim_env.run_visual_aligned \
    --headless --scene dual_franka_yubi_official_fingertip_friction_trial --setup online_aligned_v2 --seed 42 --steps 3 \
    --trajectory-policy-script "$task_root/hold_three.py" --online-chunk-30hz --policy-images all --camera head --record-wrists \
    --head-camera-calibration "$repo/yubi_isaac_sim_env/head_camera_online_aligned_v2.json" --record-run "$output"
fi
exec "$previous/runtime_venv/bin/python" "$task_root/full_task_sequential.py" \
  --headless --scene dual_franka_yubi_official_fingertip_friction_trial --setup online_aligned_v2 --seed 42 \
  --until-success --task-objective plate_return --trajectory-policy-script "$repo/adapters/$adapter" \
  --online-chunk-30hz --policy-images all --camera head --record-wrists \
  --head-camera-calibration "$repo/yubi_isaac_sim_env/head_camera_online_aligned_v2.json" \
  --stop-file "$UMI_EVAL_STOP_FILE" --record-run "$output"
