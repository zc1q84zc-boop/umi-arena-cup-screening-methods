#!/usr/bin/env bash
set -euo pipefail
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runtime=${UMI_ISAAC_RUNTIME:-/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy}
task=${1:?Task required}
output=${2:?Fresh absolute probe directory required}
case "$task" in pens|sps|phone) ;; *) exit 2;; esac
[[ "$output" = /* && ! -e "$output" ]] || exit 2
exec 9>"$runtime/console.lock"
flock -n 9 || exit 2
pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
[[ ! "$pids" =~ [0-9] ]] || { echo 'GPU0 occupied' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0
export PATH="$runtime/.venv/bin:$PATH" LD_LIBRARY_PATH="$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config" XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp"
unset UMI_CUP_MODEL UMI_AUDIT_CUP_CONTACTS SIM_ADAPTER_AUDIT_DIR UMI_REPLAY_PATH
cd "$profile"
"$runtime/.venv/bin/python" -m yubi_isaac_sim_env.arena_tasks.probe --task "$task" --output "$output"
"$runtime/.venv/bin/python" - "$output/report.json" <<'PY'
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text());assert value['status']=='ok',value.get('checks',value.get('error'))
print(value['task_id'],'goal_fit_ok')
PY
