#!/usr/bin/env python3
"""Prove LeRobot's episode-filtered rows match the audited clean manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--allowlist", required=True, type=Path)
    args = parser.parse_args()
    allowlist = json.loads(args.allowlist.read_text())
    expected = {int(r["episode_index"]): int(r["length"]) for r in json.loads(
        (args.dataset_root / "manifest.json").read_text()
    )["episodes"]}
    if set(allowlist["episode_ids"]) != set(expected) or sum(expected.values()) != 746914:
        raise ValueError("manifest and verified allowlist disagree")
    data = LeRobotDataset(
        args.dataset_root.name,
        root=args.dataset_root,
        episodes=allowlist["episode_ids"],
        video_backend="pyav",
    )
    actual = Counter(int(x) for x in data.hf_dataset["episode_index"])
    if actual != expected or len(data) != 746914:
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        raise ValueError(f"filtered rows mismatch: missing={len(missing)}, extra={len(extra)}")
    print(f"LEROBOT_FILTER_VERIFIED {len(actual)} episodes / {len(data)} frames", flush=True)


if __name__ == "__main__":
    main()
