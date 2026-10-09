#!/usr/bin/env python3
"""Prepare only audited cup motion and causal video references for OpenWAM."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


SOURCE_COLUMNS = [
    "episode_index", "frame_index", "observation.pose.left_hand_root.absolute",
    "observation.pose.right_hand_root.absolute", "observation.joint_states",
    "action.joint_states",
]
CAMERA = "observation.image.center"
META_COLUMNS = ["episode_index", "length", "task_success", "data/chunk_index", "data/file_index", *[
    f"videos/{CAMERA}/{name}" for name in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
]]


def eef10(pose: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    if pose.shape[-1] != 7 or gripper.shape != pose.shape[:-1]:
        raise ValueError("expected xyz/quaternion pose and paired gripper angle")
    out = np.empty((*pose.shape[:-1], 10), dtype=np.float32)
    out[..., :3] = pose[..., :3]
    rotation = Rotation.from_quat(pose[..., 3:].reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)
    out[..., 3:6] = rotation[..., :, 0]
    out[..., 6:9] = rotation[..., :, 1]
    out[..., 9] = gripper
    return out


def stats(array: np.ndarray) -> dict[str, list[float]]:
    result = {
        "min": array.min(axis=0), "max": array.max(axis=0),
        "mean": array.mean(axis=0), "std": array.std(axis=0),
        "q01": np.quantile(array, 0.01, axis=0),
        "q99": np.quantile(array, 0.99, axis=0),
    }
    # Rot6D already lives in [-1, 1] and must not be dimension-wise stretched.
    for start in (3, 13):
        for offset in range(start, start + 6):
            for key, value in (("min", -1), ("max", 1), ("q01", -1),
                               ("q99", 1), ("mean", 0), ("std", 1)):
                result[key][offset] = value
    return {key: np.asarray(value, dtype=np.float32).tolist() for key, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())["episodes"]
    expected = {int(row["episode_index"]): int(row["length"]) for row in rows}
    if len(expected) != 3423 or sum(expected.values()) != 746914:
        raise ValueError("manifest does not match audited clean cup selection")
    if args.output.exists() and (args.output / "manifest.json").exists():
        raise FileExistsError("prepared OpenWAM dataset is already complete")
    args.output.mkdir(parents=True, exist_ok=True)

    metadata: dict[int, dict] = {}
    for path in sorted((args.dataset_root / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path, columns=META_COLUMNS,
                              filters=[("episode_index", "in", sorted(expected))])
        for row in table.to_pylist():
            episode_id = int(row["episode_index"])
            if (episode_id in metadata or int(row["length"]) != expected[episode_id]
                    or row["task_success"] is not True):
                raise ValueError(f"duplicate or inconsistent metadata: {episode_id}")
            metadata[episode_id] = row
    if set(metadata) != set(expected):
        raise ValueError("missing selected episode metadata")

    shards: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for episode_id, item in metadata.items():
        shards[int(item["data/chunk_index"]), int(item["data/file_index"])][episode_id] = expected[episode_id]
    seen: Counter[int] = Counter()
    action_stats, state_stats = [], []
    references: dict[str, dict] = {}
    for (chunk, file_index), wanted in sorted(shards.items()):
        path = args.dataset_root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        table = pq.read_table(path, columns=SOURCE_COLUMNS,
                              filters=[("episode_index", "in", sorted(wanted))])
        current = Counter(map(int, table["episode_index"].to_numpy()))
        missing = [episode_id for episode_id, length in wanted.items() if current[episode_id] < length]
        if missing:
            adjacent = args.dataset_root / f"data/chunk-{chunk:03d}/file-{file_index+1:03d}.parquet"
            extra = pq.read_table(adjacent, columns=SOURCE_COLUMNS,
                                  filters=[("episode_index", "in", missing)])
            table = pa.concat_tables([table, extra])
            print(f"CORRECTED_SHARD next_file={file_index+1} episodes={missing}", flush=True)
        ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        seen.update(map(int, ids))
        for episode_id in wanted:
            mask = np.flatnonzero(ids == episode_id)
            if len(mask) != wanted[episode_id]:
                raise ValueError(f"source rows differ from clean manifest: {episode_id}")
            selected = table.take(pa.array(mask))
            frame = np.asarray(selected["frame_index"].to_numpy(), dtype=np.int64)
            if not np.array_equal(frame, np.arange(len(mask))):
                raise ValueError(f"non-contiguous frame index: {episode_id}")
            left = np.asarray(selected["observation.pose.left_hand_root.absolute"].to_pylist(), dtype=np.float32)
            right = np.asarray(selected["observation.pose.right_hand_root.absolute"].to_pylist(), dtype=np.float32)
            grip = np.asarray(selected["observation.joint_states"].to_pylist(), dtype=np.float32)
            grip_action = np.asarray(selected["action.joint_states"].to_pylist(), dtype=np.float32)
            if not all(np.isfinite(x).all() for x in (left, right, grip, grip_action)):
                raise ValueError(f"non-finite motion in episode {episode_id}")
            poses = np.stack([left, right], axis=1)
            state = np.concatenate([eef10(left, grip[:, 0]), eef10(right, grip[:, 1])], axis=1)
            action = np.concatenate([eef10(left[1:], grip_action[:-1, 0]),
                                     eef10(right[1:], grip_action[:-1, 1])], axis=1)
            state_stats.append(state)
            action_stats.append(action)
            target = args.output / f"episode_{episode_id:06d}.npz"
            if target.exists():
                with np.load(target) as prior:
                    if prior["poses"].shape != poses.shape:
                        raise ValueError(f"incomplete prior episode file: {target}")
            else:
                temp = target.with_suffix(".tmp.npz")
                np.savez_compressed(temp, poses=poses, observation_gripper=grip,
                                    action_gripper=grip_action)
                temp.replace(target)
            meta = metadata[episode_id]
            prefix = f"videos/{CAMERA}/"
            video = (args.dataset_root / f"videos/{CAMERA}/chunk-{int(meta[prefix+'chunk_index']):03d}/"
                     f"file-{int(meta[prefix+'file_index']):03d}.mp4")
            start, stop = float(meta[prefix + "from_timestamp"]), float(meta[prefix + "to_timestamp"])
            if not video.is_file() or abs(stop - start - len(mask) / 30) > 0.1:
                raise ValueError(f"missing or mis-timed center video: {episode_id}")
            references[str(episode_id)] = {"frames": len(mask), "video": str(video), "video_start_s": start,
                                           "video_end_s": stop}
        print(f"OPENWAM_PREP_SHARD {chunk:03d}/{file_index:03d} episodes={len(wanted)}", flush=True)
    if seen != expected:
        raise ValueError("prepared episode rows differ from audited clean selection")
    normalization = {"action": stats(np.concatenate(action_stats)),
                     "state": stats(np.concatenate(state_stats))}
    (args.output / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    np.save(args.output / "normalization_stats.npy", {
        key: {name: np.asarray(value, dtype=np.float32) for name, value in record.items()}
        for key, record in normalization.items()
    })
    summary = {"train_episodes": sorted(expected), "episodes": references,
               "total_episodes": 3423, "total_frames": 746914, "fps": 30,
               "action_horizon": 32, "video_target_frames": 33, "video_stride": 4,
               "active_action_slots": [*range(10), *range(34, 44)],
               "action_semantics": "future absolute bimanual xyz+rotation6D+gripper target at t+1..t+32",
               "state_semantics": "current absolute bimanual xyz+rotation6D+gripper at t",
               "video_semantics": "center-camera t..t+32 target frames (stride 4); only t is an inference condition"}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("OPENWAM_PREP_VERIFIED 3423 episodes / 746914 frames", flush=True)


if __name__ == "__main__":
    main()
