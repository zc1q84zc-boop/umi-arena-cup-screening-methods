#!/usr/bin/env bash
# Offline USD authoring and composition checks; does not initialize a GPU.
set -euo pipefail
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runtime=${UMI_ISAAC_RUNTIME:-/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy}
export ISAAC_PYTHON="$runtime/.venv/bin/python"
cd "$profile"
if [[ ${1:-build} == validate ]]; then
  bash yubi_isaac_sim_env/pxr_python.sh -m yubi_isaac_sim_env.arena_tasks.validate
else
  bash yubi_isaac_sim_env/pxr_python.sh -m yubi_isaac_sim_env.arena_tasks.build "$@"
fi
