#!/usr/bin/env bash
set -euo pipefail
task_root=/home/claude/umi-track1-console-4090-20261009/deployment_diagnostics/grasp_margin_20261009
tuned=/home/claude/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1
runtime=/home/claude/dual-franka-yubi-isaac-sim-deploy
source_run=$runtime/runs/console_c3b23931ada5
pair_root=${1:?unique pair directory}
start_mode=${2:-both}
[[ $pair_root =~ ^/home/claude/dual-franka-yubi-isaac-sim-deploy/runs/grasp_margin_pair_[0-9a-f]{12}$ ]] || exit 2
case "$start_mode" in
  both) test ! -e "$pair_root" || exit 2; modes=(baseline extra02) ;;
  resume-extra02)
    test -f "$pair_root/baseline/report.json" && test ! -e "$pair_root/extra02" || exit 2
    "$runtime/.venv/bin/python" - "$pair_root/baseline/report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['status']=='completed' and r['episodes'][0]['policy_steps']==360
d=r['grasp_margin_diagnostic'];assert d['source_run']=='c3b23931ada5' and d['max_right_extra_closure_fraction']==0
assert r['model_inference_requests']==0
PY
    modes=(extra02) ;;
  *) exit 2 ;;
esac
exec 9>"$runtime/console.lock"
flock -n 9 || { echo 'Public simulation lock occupied; no launch' >&2; exit 3; }
export PATH="$runtime/.venv/bin:$PATH" LD_LIBRARY_PATH="$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config" XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp"
unset UMI_REPLAY_PATH UMI_JAW_BIAS_RAD UMI_LEFT_SECOND_OFFSET_MM UMI_LEFT_SECOND_HEIGHT_MM
unset UMI_REFERENCE_DIAGNOSTIC UMI_REFERENCE_FLANGE_PLUS90 YUBI_CANONICAL_HAND_SIDES
unset UMI_PREGRASP_APPROACH_M UMI_PREGRASP_ADDITIONAL_M UMI_PI05_RIGHT_TRANSLATION_GAIN
"$runtime/.venv/bin/python" - "$source_run" <<'PY'
import hashlib,json,pathlib,sys,urllib.request
p=pathlib.Path(sys.argv[1]);m=json.loads((p/'manifest.json').read_text())
assert all(hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h for f,h in m['input_sha256'].items())
r=json.load(urllib.request.urlopen('http://127.0.0.1:8774/api/runs',timeout=5))
assert not [a for a in r if a['status'] in ('running','queued','starting','stopping')]
PY
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo 'Compute GPU occupied; no launch' >&2; exit 3; }
umask 077
[[ $start_mode != both ]] || mkdir "$pair_root"
pair_id=${pair_root##*pair_}
for mode in "${modes[@]}"; do
  extra=0
  [[ $mode == baseline ]] || extra=.02
  test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo 'Compute GPU occupied before next variant; no launch' >&2; exit 3; }
  out=$pair_root/$mode
  unit="umi-grasp-margin-$pair_id-$mode.service"
  echo "START $mode source=c3b23931ada5 extra=$extra requests=360 servo_actions=1080"
  set +e
  systemd-run --user --wait --unit="$unit" --property=TimeoutStopSec=10 --property=RuntimeMaxSec=1200 \
    --working-directory="$tuned" env \
    -u UMI_REPLAY_PATH -u UMI_JAW_BIAS_RAD -u UMI_LEFT_SECOND_OFFSET_MM -u UMI_LEFT_SECOND_HEIGHT_MM \
    -u UMI_REFERENCE_DIAGNOSTIC -u UMI_REFERENCE_FLANGE_PLUS90 -u YUBI_CANONICAL_HAND_SIDES \
    -u UMI_PREGRASP_APPROACH_M -u UMI_PREGRASP_ADDITIONAL_M -u UMI_PI05_RIGHT_TRANSLATION_GAIN \
    CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0 PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    XDG_CACHE_HOME="$XDG_CACHE_HOME" XDG_CONFIG_HOME="$XDG_CONFIG_HOME" XDG_DATA_HOME="$XDG_DATA_HOME" TMPDIR="$TMPDIR" \
    SIM_ADAPTER_AUDIT_DIR="$out" UMI_ONLINE_CALIBRATION=tuned_online_v1 UMI_CUP_MODEL=rigid \
    UMI_EXECUTE_30HZ=1 UMI_AUDIT_CUP_CONTACTS=1 UMI_REFERENCE_LIGHT_INTENSITY=2200 UMI_REFERENCE_DARK_FINGERS=1 \
    UMI_MARGIN_SOURCE="$source_run/online_adapter.jsonl" \
    UMI_MARGIN_SOURCE_SHA256=7bfba23843272f78ea8a2278fea976d75add22588143d42ba511928ee9e4608b \
    UMI_MARGIN_EXTRA_FRACTION="$extra" \
    "$runtime/.venv/bin/python" "$task_root/run_margin_probe.py" --headless \
    --scene dual_franka_yubi_cup40k_cupfriction_trial --setup online_aligned_v2 --seed 42 \
    --trajectory-policy-script "$task_root/replay_margin.py" --online-chunk-30hz --policy-images all \
    --camera head --record-wrists --head-camera-calibration yubi_isaac_sim_env/head_camera_online_aligned_v2.json \
    --task-objective plate_return --stop-file "$pair_root/stop" --record-run "$out" --steps 360 --continue-after-success
  result=$?
  set -e
  main_status=$(systemctl --user show "$unit" -p ExecMainStatus --value)
  main_code=$(systemctl --user show "$unit" -p ExecMainCode --value)
  main_pid=$(systemctl --user show "$unit" -p MainPID --value)
  group=$(systemctl --user show "$unit" -p ControlGroup --value)
  # Completed transient units may be collected immediately: ExecMainCode then
  # reads 0. The wait result and flushed report confirm execution; PID/cgroup
  # confirm cleanup. Never infer execution success from an absent unit alone.
  [[ $main_pid == 0 && $main_status == 0 && ( $result == 0 || $main_code == 1 ) && ( -z $group || ! -s "/sys/fs/cgroup$group/cgroup.procs" ) ]] || {
    echo "STOP $mode result=$result main_status=$main_status main_code=$main_code; no further variant" >&2; exit 4;
  }
  "$runtime/.venv/bin/python" - "$out/report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['status']=='completed' and r['episodes'][0]['policy_steps']==360
print('DONE requests=360 online_model_inference=0')
PY
  systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
done
echo "PAIR_DONE $pair_root"
