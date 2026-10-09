#!/usr/bin/env python3
"""Fail closed before either model can read the UMI cup training selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK = "Place the cup on the plate, then put it back to its original position"
EXPECTED_EPISODES = 3423
EXPECTED_FRAMES = 746914
EXPECTED_QUARANTINED = 552


def inspect(manifest_path: Path, quarantine_path: Path, info_path: Path) -> dict:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    info = json.loads(info_path.read_text())
    rows = manifest["episodes"]
    summary = manifest["summary"]
    if int(info["fps"]) != 30 or int(summary["fps"]) != 30:
        raise ValueError("dataset and clean manifest must both be 30 Hz")
    if len(rows) != EXPECTED_EPISODES or int(summary["total_episodes"]) != EXPECTED_EPISODES:
        raise ValueError("clean episode count differs from audited selection")
    ids: set[int] = set()
    total = 0
    for row in rows:
        episode_id = int(row["episode_index"])
        length = int(row["length"])
        if episode_id in ids or row["task"] != TASK or length < 1:
            raise ValueError(f"invalid clean row: episode {episode_id}")
        if int(row["dataset_to_index"]) - int(row["dataset_from_index"]) != length:
            raise ValueError(f"frame span mismatch: episode {episode_id}")
        ids.add(episode_id)
        total += length
    if total != EXPECTED_FRAMES or int(summary["total_frames"]) != EXPECTED_FRAMES:
        raise ValueError("clean frame count differs from audited selection")
    excluded: set[int] = set()
    for line in quarantine_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        episode_id = int(item["source_manifest_row"]["episode_index"])
        if episode_id in excluded or episode_id in ids:
            raise ValueError(f"duplicate or selected quarantine episode {episode_id}")
        excluded.add(episode_id)
    if len(excluded) != EXPECTED_QUARANTINED:
        raise ValueError("quarantine count differs from audited selection")
    return {
        "task": TASK,
        "episodes": len(ids),
        "frames": total,
        "quarantined_episodes": len(excluded),
        "fps": 30,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "episode_ids": sorted(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--quarantine", required=True, type=Path)
    parser.add_argument("--dataset-info", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = inspect(args.manifest, args.quarantine, args.dataset_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"will not overwrite an existing selection audit: {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"VERIFIED clean cup: {result['episodes']} episodes / {result['frames']} frames")


if __name__ == "__main__":
    main()
