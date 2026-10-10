#!/usr/bin/env bash
set -euo pipefail
task_root=/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/deployment_diagnostics/grasp_pvc_20261009
tuned=/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1
runtime=/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy
source_run=$runtime/runs/console_c3b23931ada5
out=${1:?unique output directory}
gpa=${2:?3, 2 or 1 GPa}
[[ $out =~ ^/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/runs/grasp_pvc_[0-9a-f]{12}$ ]] || exit 2
case "$gpa" in 3|2|1) ;; *) exit 2;; esac
test ! -e "$out" || exit 2
exec 9>"$runtime/console.lock"
flock -n 9 || { echo 'Public simulation lock occupied; no launch' >&2; exit 3; }
export PATH="$runtime/.venv/bin:$PATH" LD_LIBRARY_PATH="$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config" XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp"
"$runtime/.venv/bin/python" - "$source_run" <<'PY'
import hashlib,json,pathlib,socket,sys,urllib.request
assert socket.gethostname()=='benyun-workstation'
p=pathlib.Path(sys.argv[1]);m=json.loads((p/'manifest.json').read_text())
assert all(hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h for f,h in m['input_sha256'].items())
r=json.load(urllib.request.urlopen('http://127.0.0.1:8774/api/runs',timeout=5))
assert not [a for a in r if a['status'] in ('running','queued','starting','stopping')]
PY
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo 'Compute GPU occupied; no launch' >&2; exit 3; }
umask 077
unit="umi-grasp-pvc-${out##*_}.service"
echo "START FEM ${gpa}GPa, 1mm, i128, h240, original frozen targets, requests=360"
set +e
systemd-run --user --wait --unit="$unit" --property=TimeoutStopSec=10 --property=RuntimeMaxSec=1500 \
  --working-directory="$tuned" env \
  -u UMI_REPLAY_PATH -u UMI_JAW_BIAS_RAD -u UMI_LEFT_SECOND_OFFSET_MM -u UMI_LEFT_SECOND_HEIGHT_MM \
  -u UMI_REFERENCE_DIAGNOSTIC -u UMI_REFERENCE_FLANGE_PLUS90 -u YUBI_CANONICAL_HAND_SIDES \
  -u UMI_PREGRASP_APPROACH_M -u UMI_PREGRASP_ADDITIONAL_M -u UMI_PI05_RIGHT_TRANSLATION_GAIN \
  CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0 PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  XDG_CACHE_HOME="$XDG_CACHE_HOME" XDG_CONFIG_HOME="$XDG_CONFIG_HOME" XDG_DATA_HOME="$XDG_DATA_HOME" TMPDIR="$TMPDIR" \
  SIM_ADAPTER_AUDIT_DIR="$out" UMI_ONLINE_CALIBRATION=tuned_online_v1 \
  UMI_CUP_MODEL="diagnostic_grasp_pvc_e${gpa}_i128_h240" UMI_PVC_GPA="$gpa" \
  UMI_EXECUTE_30HZ=1 UMI_AUDIT_CUP_CONTACTS=0 UMI_REFERENCE_LIGHT_INTENSITY=2200 UMI_REFERENCE_DARK_FINGERS=1 \
  UMI_MARGIN_SOURCE="$source_run/online_adapter.jsonl" \
  UMI_MARGIN_SOURCE_SHA256=7bfba23843272f78ea8a2278fea976d75add22588143d42ba511928ee9e4608b \
  UMI_MARGIN_EXTRA_FRACTION=0 \
  "$runtime/.venv/bin/python" "$task_root/run_pvc_probe.py" --headless \
  --scene dual_franka_yubi_cup40k_cupfriction_trial --setup online_aligned_v2 --seed 42 \
  --trajectory-policy-script "$task_root/../grasp_margin_20261009/replay_margin.py" --online-chunk-30hz --policy-images all \
  --camera head --record-wrists --head-camera-calibration yubi_isaac_sim_env/head_camera_online_aligned_v2.json \
  --task-objective plate_return --stop-file "$out/stop" --record-run "$out" --steps 360 --continue-after-success
result=$?
set -e
main_pid=$(systemctl --user show "$unit" -p MainPID --value)
group=$(systemctl --user show "$unit" -p ControlGroup --value)
[[ $main_pid == 0 && ( -z $group || ! -s "/sys/fs/cgroup$group/cgroup.procs" ) ]] || exit 4
"$runtime/.venv/bin/python" - "$out/report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print('TERMINAL',r['status'],r.get('error'))
assert r['model_inference_requests']==0
PY
systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || echo 'GPU compute processes present after our unit exit; inspect ownership'
exit "$result"
