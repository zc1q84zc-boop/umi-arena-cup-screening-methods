#!/usr/bin/env python3
"""Fast, read-only per-frame triage of selected cup episodes.

This pass is deliberately not a robot feasibility certificate. It scans each
packed Parquet shard once, reads only motion columns, and emits a review queue.
Threshold crossings NEVER enter the imitation-training exclusion list.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time as wall_time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


CUP_TASK = "Place the cup on the plate, then put it back to its original position"
COLUMNS = [
    "episode_index",
    "frame_index",
    "timestamp",
    "observation.pose.left_hand_root.absolute",
    "observation.pose.right_hand_root.absolute",
    "observation.pose.left_hand_root_to_right_hand_root.absolute",
    "observation.joint_states",
    "action.joint_states",
    "action.pose.left_hand_root.relative",
    "action.pose.right_hand_root.relative",
]


def wait_for_idle_gpu(require_stable: bool) -> None:
    """Avoid bulk I/O while any user's GPU training is active."""
    idle_checks = 0
    needed_checks = 2 if require_stable else 1
    while idle_checks < needed_checks:
        probe = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        idle_checks = idle_checks + 1 if not probe.stdout.strip() else 0
        if idle_checks < needed_checks:
            wall_time.sleep(60)


def adjacent_shards(chunk: int, file_index: int):
    # LeRobot v3 uses 1,000 file slots per chunk. Metadata at an episode/file
    # boundary can point one file earlier than the actual packed Parquet rows.
    ordinal = chunk * 1000 + file_index
    for offset in (-1, 1):
        neighbor = ordinal + offset
        if neighbor >= 0:
            yield divmod(neighbor, 1000)


def recover_from_adjacent_shards(
    dataset_root: Path, chunk: int, file_index: int, episode_index: int,
    original_rows: list[dict], expected_length: int,
) -> tuple[list[dict], list[str]]:
    if len(original_rows) == expected_length:
        return original_rows, []
    rows = list(original_rows)
    extra_sources = []
    for nearby_chunk, nearby_file in adjacent_shards(chunk, file_index):
        path = dataset_root / f"data/chunk-{nearby_chunk:03d}/file-{nearby_file:03d}.parquet"
        if not path.exists():
            continue
        extra = pq.read_table(path, columns=COLUMNS)
        extra = extra.filter(pc.equal(extra["episode_index"], episode_index))
        if len(extra):
            rows.extend(extra.to_pylist())
            extra_sources.append(str(path))
        if len(rows) >= expected_length:
            break
    return rows, extra_sources


def episode_metrics(rows: list[dict], episode_index: int, expected_length: int) -> dict:
    frames = np.asarray([row["frame_index"] for row in rows], dtype=np.int64)
    time = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
    left = np.asarray([row[COLUMNS[3]] for row in rows], dtype=np.float64)
    right = np.asarray([row[COLUMNS[4]] for row in rows], dtype=np.float64)
    relative = np.asarray([row[COLUMNS[5]] for row in rows], dtype=np.float64)
    gripper = np.asarray([row[COLUMNS[6]] for row in rows], dtype=np.float64)
    action_gripper = np.asarray([row[COLUMNS[7]] for row in rows], dtype=np.float64)
    left_action = np.asarray([row[COLUMNS[8]] for row in rows], dtype=np.float64)
    right_action = np.asarray([row[COLUMNS[9]] for row in rows], dtype=np.float64)
    finite = all(np.isfinite(x).all() for x in (
        time, left, right, relative, gripper, action_gripper, left_action, right_action))
    flags: list[str] = []
    events: list[dict] = []
    if len(rows) != expected_length:
        flags.append("length_mismatch")
    if not np.array_equal(frames, np.arange(len(rows))):
        flags.append("frame_index_not_contiguous")
    dt = np.diff(time)
    if not np.isfinite(dt).all() or np.any(dt <= 0):
        flags.append("timestamp_not_increasing")
    if len(dt) and np.any(np.abs(dt - 1 / 30) > 0.002):
        flags.append("timestamp_not_30hz")
        for k in np.flatnonzero(np.abs(dt - 1 / 30) > 0.002):
            events.append({"frame_index": int(frames[k + 1]), "kind": "timestamp_gap", "dt_s": float(dt[k])})
    if not finite:
        flags.append("nonfinite_signal")
        return {
            "episode_index": episode_index,
            "frames": len(rows),
            "flags_for_review_only": flags,
            "candidate_events": events,
            "metrics_incomplete_due_to_nonfinite": True,
        }

    quat_norm = np.concatenate([
        np.linalg.norm(pose[:, 3:], axis=1)
        for pose in (left, right, relative, left_action, right_action)
    ])
    max_quat_norm_error = float(np.max(np.abs(quat_norm - 1)))
    if np.any(quat_norm < 1e-6):
        flags.append("zero_quaternion")
        return {
            "episode_index": episode_index,
            "frames": len(rows),
            "flags_for_review_only": flags,
            "candidate_events": events,
            "metrics_incomplete_due_to_zero_quaternion": True,
        }
    if max_quat_norm_error > 0.01:
        flags.append("quaternion_norm_error")
    rot_l = Rotation.from_quat(left[:, 3:])
    rot_r = Rotation.from_quat(right[:, 3:])
    predicted_rel_pos = rot_l.inv().apply(right[:, :3] - left[:, :3])
    predicted_rel_rot = rot_l.inv() * rot_r
    rel_pos_error = np.linalg.norm(predicted_rel_pos - relative[:, :3], axis=1)
    rel_rot_error = (
        Rotation.from_quat(relative[:, 3:]).inv() * predicted_rel_rot
    ).magnitude()
    max_rel_pos_error = float(np.max(rel_pos_error))
    max_rel_rot_error = float(np.max(rel_rot_error))
    if max_rel_pos_error > 0.001 or max_rel_rot_error > np.deg2rad(1):
        flags.append("interhand_frame_inconsistent")

    # These are Cartesian *recording* derivatives, not retargeted robot-joint
    # speed or acceleration. Use loose values strictly to prioritize review.
    rates = {}
    for label, pose, rot, action in (
        ("left", left, rot_l, left_action),
        ("right", right, rot_r, right_action),
    ):
        dp = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
        drot = (rot[:-1].inv() * rot[1:]).magnitude()
        good_dt = np.where(dt > 0, dt, np.nan)
        velocity = dp / good_dt
        angular_velocity = drot / good_dt
        linear_velocity_vectors = np.diff(pose[:, :3], axis=0) / good_dt[:, None]
        acceleration = (
            np.linalg.norm(
                np.diff(linear_velocity_vectors, axis=0) / good_dt[1:, None], axis=1
            ) if len(velocity) > 1 else np.array([])
        )
        # For any rigid hand-root -> TCP offset u with ||u|| <= 0.5 m,
        # ||delta TCP|| >= ||delta root|| - 2 ||u|| sin(delta angle / 2).
        # This bound is unchanged by any rigid world/base-frame transform.
        # FR3's published Cartesian translation limit is 3.0 m/s. The 0.5 m
        # offset is a generous *nominal assumption*, not measured calibration.
        fr3_tcp_speed_lower_bound = np.maximum(0.0, dp - np.sin(drot / 2.0)) / good_dt
        predicted_action_translation = rot[:-1].inv().apply(np.diff(pose[:, :3], axis=0))
        action_translation_error = np.linalg.norm(
            predicted_action_translation - action[1:, :3], axis=1)
        predicted_action_rotation = rot[:-1].inv() * rot[1:]
        action_rotation_error = (
            Rotation.from_quat(action[1:, 3:]).inv() * predicted_action_rotation
        ).magnitude()
        rates[label] = {
            "max_translation_per_tick_m": float(np.nanmax(dp)) if len(dp) else 0.0,
            "max_linear_speed_proxy_m_s": float(np.nanmax(velocity)) if len(velocity) else 0.0,
            "max_linear_acceleration_proxy_m_s2": float(np.nanmax(np.abs(acceleration))) if len(acceleration) else 0.0,
            "max_angular_speed_proxy_rad_s": float(np.nanmax(angular_velocity)) if len(angular_velocity) else 0.0,
            "max_fr3_tcp_speed_lower_bound_for_0p5m_offset_m_s": (
                float(np.nanmax(fr3_tcp_speed_lower_bound)) if len(dp) else 0.0),
            "max_action_observation_translation_error_m": (
                float(np.nanmax(action_translation_error)) if len(dp) else 0.0),
            "max_action_observation_rotation_error_rad": (
                float(np.nanmax(action_rotation_error)) if len(dp) else 0.0),
        }
        if rates[label]["max_translation_per_tick_m"] > 0.10:
            flags.append(f"{label}_large_one_tick_translation")
        if rates[label]["max_linear_speed_proxy_m_s"] > 2.0:
            flags.append(f"{label}_high_speed_proxy")
        if rates[label]["max_linear_acceleration_proxy_m_s2"] > 20.0:
            flags.append(f"{label}_high_acceleration_proxy")
        if rates[label]["max_angular_speed_proxy_rad_s"] > 2.55:
            flags.append(f"{label}_fr3_angular_speed_conflict")
        for k in np.flatnonzero((dp > 0.10) | (velocity > 2.0)):
            events.append({
                "frame_index": int(frames[k + 1]), "kind": f"{label}_translation_jump",
                "translation_m": float(dp[k]), "speed_m_s": float(velocity[k]),
                "angular_change_rad": float(drot[k]),
            })
        for k in np.flatnonzero(np.abs(acceleration) > 20.0):
            events.append({
                "frame_index": int(frames[k + 2]), "kind": f"{label}_acceleration_proxy",
                "magnitude_m_s2": float(np.abs(acceleration[k])),
            })
        for k in np.flatnonzero(fr3_tcp_speed_lower_bound > 3.05):
            flags.append(f"{label}_fr3_tcp_speed_infeasible_under_0p5m_offset")
            events.append({
                "frame_index": int(frames[k + 1]),
                "kind": f"{label}_fr3_tcp_speed_infeasible_under_0p5m_offset",
                "tcp_speed_lower_bound_m_s": float(fr3_tcp_speed_lower_bound[k]),
                "fr3_limit_m_s": 3.0,
                "max_hand_root_to_tcp_offset_m": 0.5,
                "recorded_dt_s": float(dt[k]),
            })
        for k in np.flatnonzero(angular_velocity > 2.55):
            events.append({
                "frame_index": int(frames[k + 1]),
                "kind": f"{label}_fr3_angular_speed_conflict",
                "angular_speed_rad_s": float(angular_velocity[k]),
                "fr3_limit_rad_s": 2.5,
                "recorded_dt_s": float(dt[k]),
            })
        for k in np.flatnonzero(
            (action_translation_error > 0.01) | (action_rotation_error > np.deg2rad(5))
        ):
            flags.append(f"{label}_action_observation_misaligned")
            events.append({
                "frame_index": int(frames[k + 1]),
                "kind": f"{label}_action_observation_misaligned",
                "translation_error_m": float(action_translation_error[k]),
                "rotation_error_rad": float(action_rotation_error[k]),
            })

    events.sort(key=lambda event: (event["frame_index"], event["kind"]))

    return {
        "episode_index": episode_index,
        "frames": len(rows),
        "max_quaternion_norm_error": max_quat_norm_error,
        "max_interhand_position_error_m": max_rel_pos_error,
        "max_interhand_rotation_error_rad": max_rel_rot_error,
        "gripper_angle_min_rad": np.min(gripper, axis=0).tolist(),
        "gripper_angle_max_rad": np.max(gripper, axis=0).tolist(),
        "action_gripper_angle_min_rad": np.min(action_gripper, axis=0).tolist(),
        "action_gripper_angle_max_rad": np.max(action_gripper, axis=0).tolist(),
        "hands": rates,
        "flags_for_review_only": sorted(set(flags)),
        "candidate_events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-limit", type=int, default=0)
    parser.add_argument("--only-shard", nargs=2, type=int, metavar=("CHUNK", "FILE"))
    parser.add_argument("--shard-path-override", type=Path)
    parser.add_argument("--progress-jsonl", type=Path,
                        help="append one completed shard at a time; safe to resume")
    parser.add_argument("--seconds-between-shards", type=float, default=0.0)
    parser.add_argument("--wait-for-idle-gpu", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selected = [row for row in manifest["episodes"] if row["task"] == CUP_TASK]
    if len(selected) != 3975:
        raise ValueError(f"expected 3,975 selected cup episodes, found {len(selected)}")
    by_shard: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for episode in selected:
        by_shard[(episode["data_chunk_index"], episode["data_file_index"])].append(episode)
    shard_keys = sorted(by_shard)
    if args.only_shard:
        shard_keys = [tuple(args.only_shard)]
        if shard_keys[0] not in by_shard:
            parser.error(f"selected cup task has no episodes in {shard_keys[0]}")
    if args.shard_path_override and len(shard_keys) != 1:
        parser.error("--shard-path-override requires --only-shard or one selected shard")
    if args.shard_limit:
        shard_keys = shard_keys[: args.shard_limit]
    results: list[dict] = []
    completed: set[tuple[int, int]] = set()
    if args.progress_jsonl and args.progress_jsonl.exists():
        for line in args.progress_jsonl.read_text().splitlines():
            record = json.loads(line)
            completed.add((record["chunk"], record["file_index"]))
            results.extend(record["episodes"])
    first_new_shard = True
    for chunk, file_index in shard_keys:
        if (chunk, file_index) in completed:
            continue
        if args.wait_for_idle_gpu:
            wait_for_idle_gpu(require_stable=first_new_shard)
        first_new_shard = False
        path = args.shard_path_override or (
            args.dataset_root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        )
        table = pq.read_table(path, columns=COLUMNS)
        ids = pa.array([int(row["episode_index"]) for row in by_shard[(chunk, file_index)]])
        table = table.filter(pc.is_in(table["episode_index"], value_set=ids))
        rows_by_episode: dict[int, list[dict]] = defaultdict(list)
        for row in table.to_pylist():
            rows_by_episode[int(row["episode_index"])].append(row)
        shard_results: list[dict] = []
        for episode in by_shard[(chunk, file_index)]:
            idx = int(episode["episode_index"])
            rows, extra_sources = recover_from_adjacent_shards(
                args.dataset_root, chunk, file_index, idx, rows_by_episode[idx],
                int(episode["length"]),
            )
            rows = sorted(rows, key=lambda row: row["frame_index"])
            if not rows:
                shard_results.append({
                    "episode_index": idx,
                    "frames": 0,
                    "flags_for_review_only": ["episode_missing_from_shard"],
                })
            else:
                result = episode_metrics(rows, idx, int(episode["length"]))
                if extra_sources:
                    result["corrected_source_shards"] = extra_sources
                shard_results.append(result)
        results.extend(shard_results)
        if args.progress_jsonl:
            args.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.progress_jsonl.open("a") as stream:
                stream.write(json.dumps({"chunk": chunk, "file_index": file_index,
                                         "episodes": shard_results}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        if args.seconds_between_shards:
            wall_time.sleep(args.seconds_between_shards)
    flagged = [row for row in results if row["flags_for_review_only"]]
    scanned_ids = {row["episode_index"] for row in results}
    report = {
        "scope": "selected successful cup episodes only",
        "source_manifest": str(args.manifest),
        "shards_scanned": len({(row["data_chunk_index"], row["data_file_index"])
                                for row in selected if row["episode_index"] in scanned_ids}),
        "episodes_scanned": len(results),
        "frames_scanned": sum(row["frames"] for row in results),
        "candidate_episodes_for_robot_and_human_review": len(flagged),
        "confirmed_robot_infeasible_episode_ids": [],
        "training_exclusion_applied": False,
        "reason": "recording-space flags do not prove failure on all calibrated target robots",
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "shards_scanned", "episodes_scanned", "frames_scanned",
        "candidate_episodes_for_robot_and_human_review",
        "confirmed_robot_infeasible_episode_ids", "training_exclusion_applied",
    )}, indent=2))


if __name__ == "__main__":
    main()
