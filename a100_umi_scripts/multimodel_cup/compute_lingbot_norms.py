#!/usr/bin/env python3
"""Compute LingBot normalization only from the verified clean cup frames."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FIELDS = {
    "observation.state.end.position": (
        "observation.pose.left_hand_root.absolute", "observation.pose.right_hand_root.absolute"
    ),
    "observation.state.effector.position": ("observation.joint_states",),
    "action.end.position": (
        "observation.pose.left_hand_root.relative", "observation.pose.right_hand_root.relative"
    ),
    "action.effector.position": ("action.joint_states",),
}


def statistics(value: np.ndarray) -> dict[str, list[float]]:
    result = {
        "mean": value.mean(axis=0),
        "std": value.std(axis=0),
        "min": value.min(axis=0),
        "max": value.max(axis=0),
    }
    for percentile in (1, 2, 98, 99):
        result[f"q{percentile:02d}"] = np.percentile(value, percentile, axis=0)
    return {key: np.asarray(item, dtype=np.float32).tolist() for key, item in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace norm stats: {args.output}")
    rows = json.loads(args.manifest.read_text())["episodes"]
    if len(rows) != 3423 or sum(int(r["length"]) for r in rows) != 746914:
        raise ValueError("not the audited clean cup manifest")
    expected = {int(row["episode_index"]): int(row["length"]) for row in rows}
    metadata: dict[int, dict] = {}
    for path in sorted((args.dataset_root / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path,
                              columns=["episode_index", "length", "task_success", "data/chunk_index", "data/file_index"],
                              filters=[("episode_index", "in", sorted(expected))])
        for item in table.to_pylist():
            episode_id = int(item["episode_index"])
            if (episode_id in metadata or int(item["length"]) != expected[episode_id]
                    or item["task_success"] is not True):
                raise ValueError(f"source metadata disagrees with clean manifest: {episode_id}")
            metadata[episode_id] = item
    if set(metadata) != set(expected):
        raise ValueError("clean episodes absent from source metadata")
    shards: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for episode_id, item in metadata.items():
        shard = int(item["data/chunk_index"]), int(item["data/file_index"])
        shards[shard][episode_id] = expected[episode_id]
    keys = sorted({key for pair in FIELDS.values() for key in pair})
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    seen: Counter[int] = Counter()
    for (chunk, file_index), wanted in sorted(shards.items()):
        path = args.dataset_root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        table = pq.read_table(path, columns=["episode_index", *keys],
                              filters=[("episode_index", "in", sorted(wanted))])
        current = Counter(map(int, table["episode_index"].to_numpy()))
        missing = [episode_id for episode_id, length in wanted.items() if current[episode_id] < length]
        if missing:
            adjacent = args.dataset_root / f"data/chunk-{chunk:03d}/file-{file_index+1:03d}.parquet"
            extra = pq.read_table(adjacent, columns=["episode_index", *keys],
                                  filters=[("episode_index", "in", missing)])
            table = pa.concat_tables([table, extra])
            print(f"CORRECTED_SHARD next_file={file_index+1} episodes={missing}", flush=True)
        episode_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        seen.update(map(int, episode_ids))
        source = {key: np.asarray(table[key].to_pylist(), dtype=np.float32) for key in keys}
        for output_key, input_keys in FIELDS.items():
            array = np.concatenate([source[key] for key in input_keys], axis=1)
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite values in {output_key} from {path}")
            values[output_key].append(array)
        print(f"NORM_SHARD {chunk:03d}/{file_index:03d} rows={len(table)}", flush=True)
    if seen != expected:
        missing = sorted(set(expected) - set(seen))
        raise ValueError(f"normalization rows differ from clean manifest: {len(seen)} episodes, missing={missing}")
    norms = {key: statistics(np.concatenate(parts, axis=0)) for key, parts in values.items()}
    result = {"norm_stats": norms, "count": 746914}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("LINGBOT_NORMS_VERIFIED count=746914", flush=True)


if __name__ == "__main__":
    main()
