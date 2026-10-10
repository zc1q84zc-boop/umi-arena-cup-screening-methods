#!/usr/bin/env bash
set -euo pipefail
gain=${1:?4 or 8}
case "$gain" in 4|8) ;; *) exit 2;; esac
runtime=/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy
task_root=/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/deployment_diagnostics/smoothing_response_20261010
output="$runtime/runs/smoothing_response_20261010_phase_arm8_jaw4"
test ! -e "$output"
export CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0
export PATH="$runtime/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$runtime/.cache" XDG_CONFIG_HOME="$runtime/.config" XDG_DATA_HOME="$runtime/.local" TMPDIR="$runtime/.tmp"
export UMI_CUP_MODEL=pvc_shell_e3000mpa_i128_h240_v2 UMI_AUDIT_CUP_CONTACTS=0 UMI_EXECUTE_30HZ=1 UMI_ONLINE_CALIBRATION=tuned_online_v1
unset UMI_REPLAY_PATH UMI_JAW_BIAS_RAD UMI_REFERENCE_DIAGNOSTIC UMI_REFERENCE_FLANGE_PLUS90 UMI_PI05_RIGHT_TRANSLATION_GAIN
cd /home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1
exec "$runtime/.venv/bin/python" "$task_root/gpu_compare.py" --gain "$gain" --jaw-gain 4 --output "$output" --phase-reset
