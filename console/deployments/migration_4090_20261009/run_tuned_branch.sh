#!/usr/bin/env bash
set -euo pipefail

# Run the public 259632/259633 tuned replay in its isolated checkout. This is
# recorded-data replay, never an online π0.5/LingBot/OpenWAM policy result.
repo=/home/claude/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1
runtime=/home/claude/dual-franka-yubi-isaac-sim-deploy
record=$repo/c5f59021-416a-4da4-b057-7f3d1dc35ab5.json
expected_record_sha=b5d0d9551f17368b618f9884c16106c605245ff462d3d03c9d220ea82263226e
run_dir=${1:?Pass a new, absolute output directory under the private replay workspace}
run_dir=$(realpath -m -- "$run_dir")
stop_args=()
if [[ -n ${2:-} ]]; then
  [[ $2 =~ ^/home/claude/dual-franka-yubi-isaac-sim-deploy/runs/\.stop_[0-9a-f]{12}$ ]] || {
    echo 'Refusing stop marker outside dedicated console run namespace' >&2; exit 2;
  }
  stop_args=(--stop-file "$2")
fi

case "$run_dir" in
  /home/claude/dual-franka-yubi-isaac-sim-deploy/runs/console_????????????) ;;
  *) echo 'Refusing output outside private replay workspace' >&2; exit 2 ;;
esac
test -d "$repo/yubi_isaac_sim_env"
test -x "$runtime/.venv/bin/python"
test -f "$record"
test ! -e "$run_dir" || { echo "Output already exists: $run_dir" >&2; exit 2; }
actual_record_sha=$(sha256sum "$record")
test "${actual_record_sha%% *}" = "$expected_record_sha" || {
  echo 'Private paired source record hash mismatch' >&2; exit 2;
}

exec 9>"$repo/.replay.lock"
flock -n 9 || { echo 'Another tuned replay is already running' >&2; exit 2; }
compute_pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader) || {
  echo 'GPU driver/NVML unavailable; not starting Isaac' >&2
  exit 2
}
if grep -Eq '^[0-9]+' <<<"$compute_pids"; then
  echo 'GPU0 has a compute process; not starting Isaac' >&2
  exit 2
fi

umask 077
export CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0
export PATH="$runtime/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config"
export XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp"
export UMI_REPLAY_PATH="$record" UMI_JAW_BIAS_RAD=-0.02
export UMI_LEFT_SECOND_OFFSET_MM=70 UMI_LEFT_SECOND_HEIGHT_MM=10
export UMI_LEFT_SECOND_EXTRA_CLOSE_RAD=0.005
export UMI_LEFT_SECOND_EXTRA_CLOSE_START_FRAME=423
export UMI_LEFT_SECOND_EXTRA_CLOSE_END_FRAME=438

cd "$repo"
"$runtime/.venv/bin/python" -m yubi_isaac_sim_env.run \
  --headless --continue-after-success \
  --scene dual_franka_yubi_cup40k_cupfriction_trial \
  --setup replay_259632_259633_tuned --seed 42 \
  --trajectory-policy-script yubi_isaac_sim_env/policies/umi_left_second_height_replay.py \
  --trajectory-controller-profile panda-pose-replay \
  --joint-command-profile franka-panda-interface \
  --camera overview --record-wrists \
  --head-camera-calibration yubi_isaac_sim_env/head_camera_replay.json \
  --steps 215 --record-run "$run_dir" "${stop_args[@]}"
