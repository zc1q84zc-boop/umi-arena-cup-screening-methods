#!/usr/bin/env python3
"""Build the replay branch's private input from the audited official export.

The public replay branch expects one ``poses[N,16]``/``angles[N,2]`` record.
This conversion uses *observation* poses and jaw states, never future actions.
It does not modify either official episode or publish the resulting record.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


UUID = "c5f59021-416a-4da4-b057-7f3d1dc35ab5"
EPISODES = (259632, 259633)


def build_record(source: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text())
    matching = [item for item in manifest["recordings"] if item["uuid"] == UUID]
    if len(matching) != 1 or tuple(item["episode_index"] for item in matching[0]["episodes"]) != EPISODES:
        raise ValueError("official manifest does not contain the expected paired recording")
    poses: list[list[float]] = []
    angles: list[list[float]] = []
    for expected in matching[0]["episodes"]:
        episode_id = expected["episode_index"]
        episode = json.loads((source / f"episode-{episode_id}.json").read_text())
        if (episode["recording_uuid"] != UUID or episode["episode_index"] != episode_id
                or episode["dataset_revision"] != manifest["dataset_revision"]
                or episode["fps"] != 30 or len(episode["frames"]) != expected["frames"]):
            raise ValueError(f"episode {episode_id} does not match the official manifest")
        for index, frame in enumerate(episode["frames"]):
            if frame["frame_index"] != index or abs(frame["timestamp_s"] - index / 30) > 0.003:
                raise ValueError(f"episode {episode_id}: noncontiguous 30 Hz frame {index}")
            hands = frame["observation"]["poses_xyzw"]
            jaws = frame["observation"]["grippers_rad"]
            if (len(hands) != 2 or any(len(pose) != 7 for pose in hands)
                    or len(jaws) != 2 or not all(math.isfinite(float(value))
                    for pose in hands for value in pose)
                    or not all(math.isfinite(float(value)) for value in jaws)):
                raise ValueError(f"episode {episode_id}: invalid pose or jaw at frame {index}")
            poses.append([episode_id, index, *hands[0], *hands[1]])
            angles.append([float(value) for value in jaws])
    return {"poses": poses, "angles": angles}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "official_cup_5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = build_record(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(record, destination, separators=(",", ":"))
    print(f"Prepared {len(record['poses'])} private 30 Hz observation rows: {args.output}")


if __name__ == "__main__":
    main()
