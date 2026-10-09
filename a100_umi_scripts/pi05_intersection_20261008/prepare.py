#!/usr/bin/env python3
"""Prepare independent episodes; no pairing, concatenation, or pair filtering."""
from collections import Counter, defaultdict
import argparse
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
        "observation.joint_states", "action.joint_states"] + [
        f"observation.pose.{side}_hand_root.{kind}"
        for side in ("left", "right") for kind in ("absolute", "relative")]


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def quat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    assert np.all(norm > 1e-6) and np.isfinite(q).all()
    q = q / norm
    return (q * np.where(q[..., 3:] < 0, -1, 1)).astype(np.float32)


def compose(deltas):
    p, r = np.zeros(3), Rotation.identity()
    for delta in deltas:
        p += r.apply(delta[:3])
        r = r * Rotation.from_quat(delta[3:])
    return np.r_[p, quat(r.as_quat())].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--intersection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite prepared data"
    source = args.intersection / "manifest.json"
    allowed = {r["episode_index"]: r for r in json.loads(source.read_text())["episodes"]}
    assert len(allowed) == 2781 and sum(r["length"] for r in allowed.values()) == 600087
    columns = ["episode_index", "length", "uuid", "tasks", "task_success", "data/chunk_index", "data/file_index"]
    columns += [f"videos/{c}/{k}" for c in CAMERAS for k in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    metadata = {}
    for file in sorted((args.dataset / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(file, columns=columns, filters=[("episode_index", "in", sorted(allowed))]).to_pylist():
            idx = int(row["episode_index"])
            assert idx not in metadata and row["length"] == allowed[idx]["length"] and row["task_success"]
            assert len(row["tasks"]) == 1 and row["tasks"][0].rstrip(".") in PROMPTS
            metadata[idx] = row
    assert set(metadata) == set(allowed)
    shards = defaultdict(list)
    for idx, row in metadata.items():
        shards[int(row["data/chunk_index"]), int(row["data/file_index"])].append(idx)
    args.output.mkdir(parents=True)
    (args.output / "episodes").mkdir()
    records, states, actions, weights = {}, [], [], []
    max_position_error, max_rotation_error = 0., 0.
    for (chunk, file_id), ids in sorted(shards.items()):
        path = args.dataset / f"data/chunk-{chunk:03d}/file-{file_id:03d}.parquet"
        table = pq.read_table(path, columns=COLS, filters=[("episode_index", "in", ids)])
        counts = Counter(table["episode_index"].to_pylist())
        short = [idx for idx in ids if counts[idx] < allowed[idx]["length"]]
        if short:
            table = pa.concat_tables([table, pq.read_table(path.with_name(f"file-{file_id+1:03d}.parquet"), columns=COLS, filters=[("episode_index", "in", short)])])
        episode_column = table["episode_index"].to_numpy()
        for idx in sorted(ids):
            selected = table.take(pa.array(np.flatnonzero(episode_column == idx)))
            data = {c: np.asarray(selected[c].to_pylist()) for c in COLS}
            length, meta = len(selected), metadata[idx]
            assert length == allowed[idx]["length"] and length > 3
            assert np.array_equal(data["frame_index"], np.arange(length))
            assert np.allclose(np.diff(data["timestamp"]), 1/30, atol=.002)
            assert all(np.isfinite(v).all() for v in data.values())
            starts = np.arange(0, length, 3, dtype=np.int64)
            state = np.c_[data[INTERHAND][starts], data["observation.joint_states"][starts]].astype(np.float32)
            # Keep the contract's input quaternion representation unchanged;
            # the official inference transform does not canonicalize its sign.
            assert np.allclose(np.linalg.norm(state[:, 3:7], axis=1), 1, atol=.01)
            action = np.zeros((len(starts), 16), np.float32)
            action[:, [6, 13]] = 1
            for n, t in enumerate(starts):
                end = min(t+3, length-1)
                for side, offset in (("left", 0), ("right", 7)):
                    relative = data[f"observation.pose.{side}_hand_root.relative"][t+1:end+1]
                    target = compose(relative)
                    action[n, offset:offset+7] = target
                    # Audit labels against endpoint poses; absolute hand poses
                    # are never stored in the model's input state.
                    absolute = data[f"observation.pose.{side}_hand_root.absolute"]
                    r0 = Rotation.from_quat(absolute[t, 3:])
                    expected_p = r0.inv().apply(absolute[end, :3]-absolute[t, :3])
                    expected_r = r0.inv()*Rotation.from_quat(absolute[end, 3:])
                    pe = float(np.linalg.norm(target[:3]-expected_p))
                    re = float((Rotation.from_quat(target[3:]).inv()*expected_r).magnitude())
                    max_position_error, max_rotation_error = max(max_position_error, pe), max(max_rotation_error, re)
                    assert pe < .03 and re < np.deg2rad(15), (idx, t, side, pe, re)
                action[n, 14:] = data["action.joint_states"][end-1] if end > t else data["observation.joint_states"][t]
            hold = np.zeros(16, np.float32)
            hold[[6, 13]] = 1
            hold[14:] = data["observation.joint_states"][-1]
            value = int(hashlib.sha256(f"42:{meta['uuid']}".encode()).hexdigest()[:16], 16)/2**64
            split = "val" if value < .1 else "train"
            videos = {}
            for c in CAMERAS:
                prefix = f"videos/{c}/"
                file = args.dataset / f"videos/{c}/chunk-{int(meta[prefix+'chunk_index']):03d}/file-{int(meta[prefix+'file_index']):03d}.mp4"
                assert file.is_file()
                start, end = float(meta[prefix+"from_timestamp"]), float(meta[prefix+"to_timestamp"])
                assert abs(end-start-length/30) < .1
                videos[c] = {"path": str(file), "start_s": start, "end_s": end}
            np.savez_compressed(args.output / "episodes" / f"{idx}.npz", state=state, action=action, hold=hold, raw_frame=starts)
            records[str(idx)] = {"uuid": meta["uuid"], "split": split, "raw_frames": length,
                                 "samples": len(starts), "prompt": meta["tasks"][0], "videos": videos}
            if split == "train":
                states.append(state)
                actions.extend([action, hold[None]])
                # Exact weights of every target in all 32-step chunks. Beyond
                # this episode, supervise a stationary pose with final jaws.
                w = np.minimum(np.arange(1, len(action)+1), 32)
                pad_count = int(np.maximum(0, np.arange(len(action))+32-len(action)).sum())
                weights.extend([w, np.array([pad_count])])
        print(f"PREP_SHARD {chunk}/{file_id} episodes={len(ids)}", flush=True)
    # Reuse the audited streaming quantile accumulator from the first recipe.
    from compute_norm_stats import Accumulator, pad32
    accum_state, accum_action = Accumulator(32), Accumulator(32)
    accum_state.update(pad32(np.concatenate(states)))
    for action, weight in zip(actions, weights):
        accum_action.update(pad32(action), weight)
    norm = {"norm_stats": {"state": accum_state.finish(), "actions": accum_action.finish()}}
    dump(args.output / "norm_stats.json", norm)
    train = [r for r in records.values() if r["split"] == "train"]
    val = [r for r in records.values() if r["split"] == "val"]
    assert {r["uuid"] for r in train}.isdisjoint({r["uuid"] for r in val})
    summary = {"episodes": len(records), "raw_frames": sum(r["raw_frames"] for r in records.values()),
               "train_episodes": len(train), "val_episodes": len(val),
               "train_uuids": len({r["uuid"] for r in train}), "val_uuids": len({r["uuid"] for r in val}),
               "train_samples": sum(r["samples"] for r in train), "val_samples": sum(r["samples"] for r in val),
               "train_prompts": dict(Counter(r["prompt"] for r in train)), "val_prompts": dict(Counter(r["prompt"] for r in val)),
               "max_composition_position_error_m": max_position_error,
               "max_composition_angle_error_deg": float(np.rad2deg(max_rotation_error))}
    assert summary["episodes"] == 2781 and summary["raw_frames"] == 600087
    dump(args.output / "manifest.json", {"version": "pi05_intersection_independent_v2", "episodes": records,
         "summary": summary, "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
         "source_fps": 30, "action_hz": 10, "horizon": 32, "split_seed": 42, "val_fraction": .1,
         "action_semantics": "causal local SE(3) composition of source rows t+1 through min(t+3,end); absolute jaws at endpoint",
         "tail_policy": "partial final interval then stationary local pose / final jaw hold; never cross episode",
         "no_pairing": True, "normalization": "training split only; exact horizon weights; 5000-bin q01/q99"})
    dump(args.output / "summary.json", summary)
    print("PREPARED " + json.dumps(summary), flush=True)


if __name__ == "__main__": main()
