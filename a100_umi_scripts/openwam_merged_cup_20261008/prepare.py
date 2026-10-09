#!/usr/bin/env python3
"""Build paired cup recordings using only the quality/IK intersection allowlist."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

PROMPTS = ["Pick up the cup with your right hand and set it on the plate",
           "Pick up the cup with your left hand and return the cup to its original position"]
CAMERAS = ["observation.image.left", "observation.image.right"]
INTERHAND = "observation.pose.left_hand_root_to_right_hand_root.absolute"
COLS = ["episode_index", "frame_index", "timestamp", INTERHAND,
        "observation.joint_states", "action.joint_states",
        "observation.pose.left_hand_root.absolute", "observation.pose.right_hand_root.absolute",
        "observation.pose.left_hand_root.relative", "observation.pose.right_hand_root.relative"]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def canonical_quat(array):
    q = np.asarray(array, dtype=np.float32).copy()
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-6) or not np.isfinite(q).all():
        raise ValueError("invalid quaternion")
    q /= norms
    q *= np.where(q[..., 3:] < 0, -1, 1)
    return q


def compose_deltas(deltas):
    """Compose local SE(3) per-frame deltas into one control-period delta."""
    position = np.zeros(3)
    rotation = Rotation.identity()
    for row in deltas:
        position += rotation.apply(row[:3])
        rotation = rotation * Rotation.from_quat(row[3:])
    return np.r_[position, canonical_quat(rotation.as_quat())].astype(np.float32)


def stats(values, quat_slices):
    out = {"min": values.min(0), "max": values.max(0),
           "mean": values.mean(0), "std": values.std(0),
           "q01": np.quantile(values, .01, axis=0), "q99": np.quantile(values, .99, axis=0)}
    for lo, hi in quat_slices:
        for key in ("min", "q01"): out[key][lo:hi] = -1
        for key in ("max", "q99", "std"): out[key][lo:hi] = 1
        out["mean"][lo:hi] = 0
    return {k: np.asarray(v, dtype=np.float32).tolist() for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--intersection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.1)
    # Reuse the quality review's existing 2 m/s and 2.55 rad/s bounds
    # at the contiguous 30 Hz recording boundary, rather than adding a
    # stricter arbitrary displacement cutoff to the intersection.
    parser.add_argument("--boundary-position-m", type=float, default=2/30)
    parser.add_argument("--boundary-angle-deg", type=float, default=float(np.rad2deg(2.55/30)))
    args = parser.parse_args()
    if args.output.exists(): parser.error("output already exists")
    if not 0 < args.val_fraction < 1: parser.error("invalid validation fraction")
    manifest_path = args.intersection / "manifest.json"
    original = json.loads(manifest_path.read_text())
    allowed = {int(r["episode_index"]): r for r in original["episodes"]}
    if len(allowed) != 2781 or sum(r["length"] for r in allowed.values()) != 600087:
        raise ValueError("unexpected intersection version")
    members = [json.loads(x) for x in (args.intersection / "record_membership.jsonl").read_text().splitlines()]
    decisions, candidates = [], []
    for member in members:
        ids = member["retained_episode_indices"]
        if len(ids) != 2:
            decisions.append({"uuid": member["uuid"], "episodes": ids, "decision": "exclude", "reason": "requires_two_retained_episodes"})
            continue
        source_ids = member["ik_episode_indices"]
        if abs(source_ids.index(ids[0]) - source_ids.index(ids[1])) != 1:
            decisions.append({"uuid": member["uuid"], "episodes": ids, "decision": "exclude", "reason": "intermediate_episode_not_retained"})
            continue
        candidates.append(member)
    candidate_ids = {i for m in candidates for i in m["retained_episode_indices"]}
    meta_cols = ["episode_index", "length", "tasks", "uuid", "task_success", "data/chunk_index", "data/file_index"]
    meta_cols += [f"videos/{c}/{field}" for c in CAMERAS for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    metadata = {}
    for path in sorted((args.dataset / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path, columns=meta_cols, filters=[("episode_index", "in", sorted(candidate_ids))]).to_pylist():
            idx = int(row["episode_index"])
            assert idx not in metadata and row["length"] == allowed[idx]["length"] and row["task_success"] is True
            metadata[idx] = row
    assert set(metadata) == candidate_ids
    phase_candidates = []
    for member in candidates:
        ids = member["retained_episode_indices"]
        prompts = [metadata[i]["tasks"] for i in ids]
        valid = all(metadata[i]["uuid"] == member["uuid"] for i in ids)
        valid &= all(len(p) == 1 and p[0].rstrip(".") == PROMPTS[n] for n, p in enumerate(prompts))
        if not valid:
            decisions.append({"uuid": member["uuid"], "episodes": ids, "decision": "exclude", "reason": "uuid_or_stage_order_mismatch", "tasks": prompts})
        else: phase_candidates.append(member)
    wanted_ids = {i for m in phase_candidates for i in m["retained_episode_indices"]}
    shards = defaultdict(list)
    for idx in wanted_ids:
        row = metadata[idx]
        shards[int(row["data/chunk_index"]), int(row["data/file_index"])].append(idx)
    motion = {}
    for (chunk, file_index), wanted in sorted(shards.items()):
        path = args.dataset / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        table = pq.read_table(path, columns=COLS, filters=[("episode_index", "in", wanted)])
        counts = Counter(table["episode_index"].to_pylist())
        short = [i for i in wanted if counts[i] < allowed[i]["length"]]
        if short:
            extra = pq.read_table(path.with_name(f"file-{file_index+1:03d}.parquet"), columns=COLS, filters=[("episode_index", "in", short)])
            table = pa.concat_tables([table, extra])
        indices = table["episode_index"].to_numpy()
        for idx in wanted:
            selected = table.take(pa.array(np.flatnonzero(indices == idx)))
            assert len(selected) == allowed[idx]["length"]
            data = {col: np.asarray(selected[col].to_pylist()) for col in COLS}
            assert np.array_equal(data["frame_index"], np.arange(len(selected)))
            assert np.allclose(np.diff(data["timestamp"]), 1 / 30, atol=.002)
            assert all(np.isfinite(v).all() for v in data.values())
            motion[idx] = data
        print(f"PREP_SHARD {chunk}/{file_index} episodes={len(wanted)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "records").mkdir()
    records, state_train, action_train = {}, [], []
    for member in phase_candidates:
        ids, uuid = member["retained_episode_indices"], member["uuid"]
        first, second = [motion[i] for i in ids]
        boundary = {}
        for side in ("left", "right"):
            key = f"observation.pose.{side}_hand_root.absolute"
            a, b = first[key][-1], second[key][0]
            boundary[side] = {"position_m": float(np.linalg.norm(a[:3] - b[:3])),
                              "angle_deg": float(np.rad2deg((Rotation.from_quat(a[3:]).inv() * Rotation.from_quat(b[3:])).magnitude()))}
        if any(b["position_m"] > args.boundary_position_m or b["angle_deg"] > args.boundary_angle_deg for b in boundary.values()):
            decisions.append({"uuid": uuid, "episodes": ids, "decision": "exclude", "reason": "discontinuous_boundary", "boundary": boundary})
            continue
        videos, gap_bad = [], False
        for idx in ids:
            entry = {}
            for camera in CAMERAS:
                m = metadata[idx]; prefix = f"videos/{camera}/"
                path = args.dataset / f"videos/{camera}/chunk-{int(m[prefix+'chunk_index']):03d}/file-{int(m[prefix+'file_index']):03d}.mp4"
                start, end = float(m[prefix+"from_timestamp"]), float(m[prefix+"to_timestamp"])
                assert path.is_file() and abs(end-start-m["length"]/30) < .1
                entry[camera] = {"path": str(path), "start_s": start, "end_s": end}
            videos.append(entry)
        for camera in CAMERAS:
            a, b = videos[0][camera], videos[1][camera]
            if a["path"] == b["path"] and abs(b["start_s"]-a["end_s"]) > 1/60:
                gap_bad = True
        if gap_bad:
            decisions.append({"uuid": uuid, "episodes": ids, "decision": "exclude", "reason": "video_boundary_gap"})
            continue
        raw = {key: np.concatenate([first[key], second[key]]) for key in COLS}
        cut, total = len(first["frame_index"]), len(raw["frame_index"])
        # Sampling restarts at each prompt boundary; no action is supervised
        # under the preceding subtask's instruction.
        starts = np.r_[np.arange(0, cut-3, 3), np.arange(cut, total-3, 3)].astype(np.int64)
        phase = (starts >= cut).astype(np.int64)
        state = np.c_[raw[INTERHAND][starts], raw["observation.joint_states"][starts]].astype(np.float32)
        state[:, 3:7] = canonical_quat(state[:, 3:7])
        action = np.empty((len(starts), 16), dtype=np.float32)
        for n, t in enumerate(starts):
            for side, offset in (("left", 0), ("right", 7)):
                deltas = raw[f"observation.pose.{side}_hand_root.relative"][t+1:t+4]
                action[n, offset:offset+7] = compose_deltas(deltas)
            action[n, 14:] = raw["action.joint_states"][t+2]
        assert state.shape[1] == 9 and action.shape[1] == 16
        assert np.isfinite(state).all() and np.isfinite(action).all()
        value = int(hashlib.sha256(f"{args.seed}:{uuid}".encode()).hexdigest()[:16], 16) / 2**64
        split = "val" if value < args.val_fraction else "train"
        if split == "train": state_train.append(state); action_train.append(action)
        np.savez_compressed(args.output / "records" / f"{uuid}.npz", state=state, action=action, raw_frame=starts, phase=phase)
        records[uuid] = {"episode_indices": ids, "raw_frames": total, "phase_boundary_raw_frame": cut,
                         "control_frames": len(starts), "split": split, "videos": videos,
                         "prompts": [metadata[i]["tasks"][0] for i in ids], "boundary": boundary}
        decisions.append({"uuid": uuid, "episodes": ids, "decision": "merge", "split": split, "boundary": boundary})
    assert records and state_train and any(r["split"] == "val" for r in records.values())
    norm = {"state": stats(np.concatenate(state_train), [(3,7)]),
            "action": stats(np.concatenate(action_train), [(3,7),(10,14)])}
    dump(args.output / "normalization.json", norm)
    np.save(args.output / "normalization_stats.npy", {k: {n: np.asarray(v,dtype=np.float32) for n,v in s.items()} for k,s in norm.items()})
    summary = {"intersection_episodes": len(allowed), "candidate_two_episode_records": len(candidates),
               "merged_records": len(records), "merged_source_episodes": len(records)*2,
               "merged_raw_frames": sum(r["raw_frames"] for r in records.values()),
               "train_records": sum(r["split"]=="train" for r in records.values()),
               "val_records": sum(r["split"]=="val" for r in records.values()),
               "control_samples": sum(r["control_frames"] for r in records.values()),
               "exclusion_reasons": dict(Counter(d["reason"] for d in decisions if d["decision"]=="exclude"))}
    dump(args.output / "manifest.json", {"version": "openwam_merged_official_yubi_v1", "records": records,
         "summary": summary, "source_fps":30, "control_fps":10, "action_horizon":32,
         "video_stride":4, "video_frames":9, "state_dim":9, "action_dim":16,
         "image_layout":"left wrist tile then right wrist tile; 640x256 RGB; native inputs 480x640",
         "action_semantics":"10Hz causal per-step local SE(3) delta composed from the next three official 30Hz relative-pose rows; gripper action at t+2",
         "prompt_boundary_policy":"merged recording, per-frame subtask prompt; target loss masks stop at prompt switch",
         "source_manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
         "split_seed":args.seed,"validation_fraction":args.val_fraction,
         "boundary_thresholds":{"position_m":args.boundary_position_m,"angle_deg":args.boundary_angle_deg}})
    (args.output / "pairing_decisions.jsonl").write_text("".join(json.dumps(d)+"\n" for d in decisions))
    dump(args.output / "summary.json", summary)
    print("PREPARED "+json.dumps(summary),flush=True)


if __name__ == "__main__": main()
