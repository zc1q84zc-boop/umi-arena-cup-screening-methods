#!/usr/bin/env python3
"""Export the official practice suite's recorded EE poses and gripper joints.

Run where the gated LeRobot v3 dataset and umi-arena-evaluation are available.
Only numeric motion data is exported; wrist videos are not copied.
"""
import argparse
import json
from pathlib import Path

from umi_arena.replay_data import Dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    suite = json.loads(args.suite.read_text())
    dataset = Dataset(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = []
    for task in suite["tasks"]:
        for recording in task["recordings"]:
            for step, index in enumerate(recording["episodes"]):
                episode = dataset.episode(index)
                entry = {
                    "episode_index": index,
                    "task_id": task["id"],
                    "task_name": task["name"],
                    "step": step + 1,
                    "step_name": task["steps"][step],
                    "recording_uuid": recording["uuid"],
                    "frames": episode.length,
                    "duration_s": round(float(episode.times[-1] - episode.times[0]), 3),
                }
                data = {
                    **entry,
                    "fps": dataset.fps,
                    "times": (episode.times - episode.times[0]).round(6).tolist(),
                    "poses": episode.hand_poses.round(7).tolist(),
                    "grippers": episode.joints.round(7).tolist(),
                }
                (args.output / f"episode-{index}.json").write_text(json.dumps(data, separators=(",", ":")))
                catalog.append(entry)
                print(f"exported {index}: {episode.length} frames", flush=True)
    (args.output / "catalog.json").write_text(json.dumps({
        "suite": suite["name"], "dataset_repo": suite["dataset_repo"],
        "dataset_revision": suite["dataset_revision"], "fps": dataset.fps,
        "episodes": catalog,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
