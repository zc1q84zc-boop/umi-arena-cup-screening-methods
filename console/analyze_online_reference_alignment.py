"""Matched-30-Hz diagnostic for a recorded reference and online Isaac run.

This compares trajectories and object image landmarks. A successful recorded
episode is a reference, not an exact target path a stochastic policy must
imitate. The provisional Reference259632 mapping is not hand-eye calibration.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset_replay.audit_right_wrist_landmarks import candidates, frame_at
from online_calibration import Reference259632


def run(source_json, source_video, run_dir, frames):
    source = json.loads(Path(source_json).read_text())
    rows = [json.loads(line) for line in (Path(run_dir) / "online_adapter.jsonl").read_text().splitlines()]
    real = cv2.VideoCapture(str(source_video))
    sim = cv2.VideoCapture(str(Path(run_dir) / "video_right_wrist.mp4"))
    if not real.isOpened() or not sim.isOpened():
        raise ValueError("both source and simulator right-wrist videos are required")
    if abs(real.get(cv2.CAP_PROP_FPS)-30) > 1e-3 or abs(sim.get(cv2.CAP_PROP_FPS)-30) > 1e-3:
        raise ValueError("both videos must be 30 Hz")
    calibration = Reference259632()
    comparison = []
    try:
        for frame in frames:
            if frame >= len(source["frames"]) or frame // 3 >= len(rows):
                continue
            sample = source["frames"][frame]["observation"]["poses_xyzw"][1]
            reference, _ = calibration.hand_to_world_tool(
                "right", sample[:3], [sample[6], *sample[3:6]])
            row = rows[frame // 3]
            if row["step"] != frame // 3:
                raise ValueError("online audit has a frame/step gap")
            current = np.asarray(row["world_tool_pose"]["right"]["position_m"])
            model_target = np.asarray(row["unassisted_model_waypoint"]["right"]["position_m"])
            source_gripper = float(source["frames"][frame]["observation"]["grippers_rad"][1])
            model_input_gripper = float(row["source_gripper_rad"][1])
            model_target_gripper = float(row["source_action_gripper_rad"][1])
            pads = (row.get("pregrasp_approach") or {}).get("pad_sidewall") or {}
            pad_gaps_mm = {
                name: round(float(pads[name]["sidewall_gap_m"]) * 1000, 1)
                for name in ("left_finger", "right_finger") if name in pads
            }
            real_cup = candidates(frame_at(real, frame), "cup", "real")
            sim_cup = candidates(frame_at(sim, 3 * (frame // 3)), "cup", "isaac")
            comparison.append({
                "source_frame": frame,
                "model_step": frame // 3,
                "sim_video_frame": 3 * (frame // 3),
                "reference_tool_xyz_m": reference.tolist(),
                "sim_tool_xyz_m": current.tolist(),
                "model_target_xyz_m": model_target.tolist(),
                "reference_to_sim_tool_mm": round(float(np.linalg.norm(reference-current))*1000, 1),
                "reference_to_model_target_mm": round(float(np.linalg.norm(reference-model_target))*1000, 1),
                "model_target_to_sim_tool_mm": round(float(np.linalg.norm(model_target-current))*1000, 1),
                "source_right_gripper_rad": round(source_gripper, 4),
                "model_input_right_gripper_rad": round(model_input_gripper, 4),
                "model_target_right_gripper_rad": round(model_target_gripper, 4),
                "sim_right_pad_sidewall_gap_mm": pad_gaps_mm or None,
                "sim_right_pad_azimuth_separation_deg": (row.get("pregrasp_approach") or {}).get(
                    "pad_azimuth_separation_deg"),
                "real_cup": real_cup[0] if real_cup else None,
                "sim_cup": sim_cup[0] if sim_cup else None,
            })
    finally:
        real.release()
        sim.release()
    return {
        "source_episode": source["episode_index"],
        "simulation_run": Path(run_dir).name,
        "method": "source frame f vs online observation at model step floor(f/3); "
                  "sim video frame 3*floor(f/3), so temporal offset is 0–2 frames at 30 Hz",
        "caveat": "Recorded demonstration is a diagnostic reference, not a required policy path. "
                  "Reference259632 is a provisional transform, not measured hand-eye calibration. "
                  "Source glove angle and simulator jaw angle have no measured aperture equivalence. "
                  "Color segmentation is approximate; verify landmarks visually.",
        "samples": comparison,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 30, 60, 87, 120, 180, 240])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.source_json, args.source_video, args.run_dir, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
