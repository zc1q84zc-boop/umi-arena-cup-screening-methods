#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname) == benyun-workstation ]] || exit 2
models=/home/claude/workspace/umi_cup_models_4090_20261009
console=/home/claude/umi-track1-console-4090-20261009
runtime=/home/claude/dual-franka-yubi-isaac-sim-deploy
task_dir=/home/claude/workspace/umi_official_transfer_20261009
unit=lingbot-official-squirrel.service
checkpoint=$models/lingbot/official/hf_ckpt
audit=$console/deployment_validation/official_20261009
test -f "$checkpoint/official_manifest.json"
exec 9>"$runtime/console.lock"
flock -n 9 || { echo 'Runtime lock occupied; validation deferred'; exit 2; }
for gpu in 0 1; do
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
  [[ -z $pids ]] || { echo "GPU$gpu occupied; validation deferred"; exit 2; }
done
[[ $(systemctl --user show "$unit" -p MainPID --value) == 0 ]] || exit 2
owned_pid=0
cleanup() {
  current_pid=$(systemctl --user show "$unit" -p MainPID --value)
  if [[ $owned_pid != 0 && $current_pid == "$owned_pid" ]]; then
    cmd=$(tr '\0' ' ' <"/proc/$owned_pid/cmdline")
    [[ $cmd == *"$checkpoint"* && $cmd == *'lingbot_sim_server.py'* ]] || return 2
    systemctl --user stop "$unit"
  fi
}
trap cleanup EXIT
mkdir -p "$audit"
systemctl --user start "$unit"
owned_pid=$(systemctl --user show "$unit" -p MainPID --value)
[[ $owned_pid =~ ^[1-9][0-9]*$ ]] || exit 2
echo "OFFICIAL_GPU1_LOADING $owned_pid"
deadline=$((SECONDS + 600))
while ! curl -fsS --max-time 2 http://127.0.0.1:18814/health >"$audit/health.json" 2>/dev/null; do
  [[ $(systemctl --user show "$unit" -p MainPID --value) == "$owned_pid" ]] || {
    echo 'Official serving process exited'; exit 2;
  }
  (( SECONDS < deadline )) || { echo 'Official serving load timeout'; exit 2; }
  sleep 2
done
echo 'OFFICIAL_GPU1_HEALTH_READY'
"$models/runtime/lingbot_venv/bin/python" "$models/lingbot/probe_lingbot_causal_server.py" \
  --source-run "$console/sim_runs/364e3bbf1886" --url http://127.0.0.1:18814/infer \
  --expected-model lingbot-vla2-official-pretrained --steps 3 >"$audit/server_probe.json"
cleanup
owned_pid=0
"$models/runtime/lingbot_venv/bin/python" "$task_dir/record_official_validation.py"
echo 'OFFICIAL_TARGET_SERVING_VALIDATED_AND_STOPPED'
