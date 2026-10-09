#!/usr/bin/env bash
set -euo pipefail

deploy=/home/lrl/dual-franka-yubi-isaac-sim-deploy
repo="$deploy/repo_online_smooth_20260929"
cd "$repo"
export CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0
export PATH="$deploy/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$deploy/.venv/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$deploy/.cache" XDG_CONFIG_HOME="$deploy/.config"
export XDG_DATA_HOME="$deploy/.local" TMPDIR="$deploy/.tmp"
export YUBI_CANONICAL_HAND_SIDES=1 UMI_REFERENCE_FLANGE_PLUS90=1
export UMI_REFERENCE_DIAGNOSTIC="$deploy/adapters/reference_259632.json"
export UMI_ONLINE_SMOOTH=1 UMI_REPLAY_AUDIT_PATH="$repo/online_adapter.jsonl"

if (( $# )); then
    variants=("$@")
else
    variants=(legacy report_v1)
fi
for variant in "${variants[@]}"; do
    steps=${UMI_AB_STEPS:-40}
    [[ "$steps" =~ ^[0-9]+$ ]] && (( steps > 0 && steps <= 120 )) || exit 2
    profile="$variant"
    if [[ "$variant" == legacy_damped || "$variant" == legacy_preview_damped ]]; then
        profile=legacy
    fi
    export UMI_ONLINE_SMOOTH_PROFILE="$profile"
    unset UMI_TRAJECTORY_PREVIEW
    if [[ "$variant" == legacy_preview_damped ]]; then
        export UMI_TRAJECTORY_PREVIEW=1
    fi
    if [[ "$variant" == legacy ]]; then
        scene=yubi_isaac_sim_env/scenes/dual_franka_yubi_random_000_seed_20260924.usda
    else
        scene=yubi_isaac_sim_env/scenes/dual_franka_yubi_online_smooth.usda
    fi
    output="$deploy/runs/online_smooth_ab_${variant}_${steps}_20260929"
    mkdir -p "$output"
    export SIM_ADAPTER_AUDIT_DIR="$output"
    "$deploy/.venv/bin/python" -m yubi_isaac_sim_env.run_reference_diagnostic \
        --headless --scene "$scene" --setup "$deploy/adapters/reference_259632.json" \
        --seed 42 --camera head --online-chunk-30hz --steps "$steps" \
        --task-objective plate_return --policy-images all \
        --trajectory-policy-script "$repo/replay_recorded_online_waypoints.py" \
        --record-run "$output" >"$deploy/.tmp/online_smooth_ab_${variant}_${steps}.log" 2>&1
    python -c 'import json,sys; x=json.load(open(sys.argv[1])); print(sys.argv[1],x["status"],x["episodes"][0]["policy_steps"],flush=True)' "$output/report.json"
done
