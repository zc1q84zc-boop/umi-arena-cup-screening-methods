"""Import a verified private grasp probe into the simulation console.

The imported run is explicitly labelled as a non-model diagnostic. Raw model
weights and training data are never copied. Existing console runs are never
overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import uuid

from analyze_grasp_geometry import analyze
from sim_console import _stable_lift_streak


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "sim_runs"
FILES = ("report.json", "manifest.json", "video.mp4", "joints.csv", "grasp_calibration.jsonl")


def summarize(source: Path, scene_tuning: str) -> dict:
    episode = json.loads((source / "report.json").read_text())["episodes"][0]
    rows = [json.loads(line) for line in (source / "grasp_calibration.jsonl").read_text().splitlines()]
    reset = rows[0]
    steps = [row for row in rows if row.get("event") == "step"]
    if len(steps) != episode["policy_steps"] or [row["step"] for row in steps] != list(range(len(steps))):
        raise ValueError("trace and report policy steps do not align")
    if episode["video_frames"] != episode["joint_samples"]:
        raise ValueError("video/joint sample counts disagree")
    start_z = float(reset["cup_initial_position_m"][2])
    maximum_z = max(float(row["cup_position_m"][2]) for row in steps)
    final = episode["final_observation"]["objects"]["cup"]
    w, x, y, z = final["quaternion_wxyz"]
    vertical = max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))
    final_tilt_deg = math.degrees(math.acos(vertical))
    phases = {phase: [row["step"] for row in steps if row["phase"] == phase]
              for phase in ("align_above", "descend", "close", "lift", "hold", "abort")}
    close_steps = phases["close"]
    geometry = analyze(source / "grasp_calibration.jsonl", [close_steps[len(close_steps) // 2]])[0] if close_steps else None
    contact = geometry["meshes"] if geometry else None
    max_lift_m = maximum_z - start_z
    stable_lift_steps = _stable_lift_streak(steps, start_z)
    stable_lift_candidate = stable_lift_steps >= 10
    return {
        "classification": "stable_lift_candidate" if stable_lift_candidate else "contact_or_transient_lift_only",
        "scene_tuning": scene_tuning,
        "training_model_used": False,
        "episode": episode["scenario_id"],
        "policy_steps": episode["policy_steps"],
        "video_frames": episode["video_frames"],
        "joint_samples": episode["joint_samples"],
        "approach_reached": bool(close_steps),
        "lift_command_reached": bool(phases["lift"]),
        "phase_first_steps": {key: values[0] for key, values in phases.items() if values},
        "initial_cup_world_m": reset["cup_initial_position_m"],
        "initial_tool_world_pose": reset["tool_initial_pose"],
        "initial_yubi_base_world_pose": reset["yubi_base_initial_pose"],
        "target_grasp_world_m": reset["target_grasp_m"],
        "reference_jaw_angle_rad": steps[geometry["step"]]["driven_jaw_rad"] if geometry else None,
        "reference_jaw_mesh_contact": contact,
        "max_cup_lift_m": max_lift_m,
        "final_cup_lift_m": float(final["position_m"][2]) - start_z,
        "final_cup_tilt_deg": final_tilt_deg,
        "final_cup_xy_displacement_m": math.dist(
            final["position_m"][:2], reset["cup_initial_position_m"][:2]
        ),
        "stable_lift_hold_steps": stable_lift_steps,
        "stable_lift_candidate": stable_lift_candidate,
        "stable_grasp_verified": False,
        "success_flag": episode["success"],
        "caveat": ("Simulated table/base poses and camera extrinsics are provisional. "
                   "UMI hand-root to YUBI contact transform remains unmeasured; "
                   "nearest STL vertices are geometric proxies, not PhysX force sensors."),
    }


def import_run(source: Path, label: str, scene_tuning: str) -> Path:
    source = source.resolve()
    for name in FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    RUNS.mkdir(exist_ok=True)
    for existing in RUNS.glob("*/metadata.json"):
        if json.loads(existing.read_text()).get("calibration_source") == str(source):
            raise ValueError(f"already imported: {existing.parent}")
    summary = summarize(source, scene_tuning)
    run_id = uuid.uuid4().hex[:12]
    destination = RUNS / run_id
    destination.mkdir(mode=0o700)
    for name in FILES:
        shutil.copy2(source / name, destination / name)
    (destination / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    now = datetime.now(timezone.utc).isoformat()
    recorded_camera = json.loads((source / "report.json").read_text())["camera"]["name"]
    metadata = {
        "id": run_id, "status": "completed", "created_at": now, "finished_at": now,
        "policy": "calibration-grasp-probe", "policy_label": label,
        "setup_index": 0, "seed": 42, "steps": summary["policy_steps"],
        "task_objective": json.loads((source / "report.json").read_text()).get("task_objective", "plate"),
        "camera": recorded_camera, "gpu": 6, "inference_backend": None,
        "calibration_source": str(source),
        "calibration_result": {
            "scene_tuning": scene_tuning,
            "label": ("稳定提起候选，仍需双侧接触核验" if summary["stable_lift_candidate"] else
                      "夹爪未闭合；安全中止" if summary["phase_first_steps"].get("abort") is not None
                      and not summary["lift_command_reached"] else "接触后滑脱，稳定抓取未通过"),
            "max_cup_lift_mm": round(1000 * summary["max_cup_lift_m"], 1),
            "final_cup_tilt_deg": round(summary["final_cup_tilt_deg"], 1),
        },
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scene-tuning", required=True)
    args = parser.parse_args()
    print(import_run(args.source, args.label, args.scene_tuning))
