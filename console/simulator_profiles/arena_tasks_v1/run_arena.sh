#!/usr/bin/env bash
set -euo pipefail
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runtime=${UMI_ISAAC_RUNTIME:-/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy}
task=${1:?Task pens/sps/cable/phone required}
output=${2:?Pass a fresh absolute recording directory}
seed=${3:-42}
index=${4:-0}
steps=${5:-60}
case "$task" in pens|sps|cable|phone) ;; *) echo 'Unsupported task' >&2; exit 2;; esac
[[ "$output" = /* && ! -e "$output" ]] || { echo 'Use a fresh absolute output directory' >&2; exit 2; }
for value in "$seed" "$index" "$steps"; do [[ $value =~ ^[0-9]+$ ]] || exit 2; done
[[ "$steps" -gt 0 ]] || exit 2
exec 9>"$runtime/console.lock"
flock -n 9 || { echo 'Another simulator owns the shared console lock' >&2; exit 2; }
pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
if [[ "$pids" =~ [0-9] ]]; then echo 'GPU0 is occupied; launch postponed' >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0
export PATH="$runtime/.venv/bin:$PATH" LD_LIBRARY_PATH="$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config" XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp"
unset UMI_CUP_MODEL UMI_AUDIT_CUP_CONTACTS UMI_REPLAY_PATH
cd "$profile"
"$runtime/.venv/bin/python" -m yubi_isaac_sim_env.arena_tasks.run --headless --task "$task" --record-run "$output" --seed "$seed" --index "$index" --steps "$steps" "${@:6}"
# Kit may return a zero process status after a Python failure; require the report.
"$runtime/.venv/bin/python" - "$output/report.json" <<'PY'
import json,sys
from pathlib import Path
report=json.loads(Path(sys.argv[1]).read_text())
if report.get('status')!='completed':raise SystemExit('Simulation report did not complete')
if report.get('video_frames',0)<2:raise SystemExit('No usable recorded frames')
print(report['task_id'],report['status'],report['video_frames'])
PY
