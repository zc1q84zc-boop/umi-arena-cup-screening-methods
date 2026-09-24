#!/usr/bin/env python3
"""Replay one trained cup checkpoint on clean practice episodes, without Docker.

This produces the trajectory fields consumed by the Track 1 console. It uses
the evaluation repository's Dataset, Settings and compare() implementation,
but it is a visualization replay, not an official evaluation run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback


CUP_TASK = "Place the cup on the plate, then put it back to its original position"


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint_id.startswith("pi05-cup-clean-"):
        parser.error("checkpoint ID must identify the clean cup run")
    if not all((args.checkpoint / name).exists() for name in ("params", "assets", "train_state", "_CHECKPOINT_METADATA")):
        parser.error("checkpoint is incomplete")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")

    sys.path.insert(0, str(args.evaluation_root))
    sys.path.insert(0, str(args.openpi_root / "src"))
    sys.path.insert(0, str(args.evaluation_root / "adapters/openpi_pi05"))
    from umi_arena.replay_data import Dataset
    from umi_arena.replay_metrics import Settings, compare, summarize
    from policy import Policy

    suite = json.loads(args.suite.read_text())
    clean = json.loads(args.train_manifest.read_text())
    clean_ids = {int(row["episode_index"]) for row in clean["episodes"]}
    cup_tasks = [task for task in suite["tasks"] if task["dataset_task"] == CUP_TASK]
    if len(cup_tasks) != 1:
        raise ValueError("practice suite must have exactly one cup task")
    suite_ids = [int(index) for recording in cup_tasks[0]["recordings"] for index in recording["episodes"]]
    selected = [index for index in suite_ids if index in clean_ids]
    excluded = [index for index in suite_ids if index not in clean_ids]
    if args.max_episodes:
        selected = selected[:args.max_episodes]
    if not selected:
        raise ValueError("no clean practice cup episodes")

    settings = Settings(30, "body", "previous")
    if args.output.exists():
        if not args.resume:
            parser.error(f"output already exists: {args.output}; use --resume only after inspection")
        report = json.loads((args.output / "report.json").read_text())
        if report.get("checkpoint_id") != args.checkpoint_id or report.get("requested_episodes") != selected:
            raise ValueError("existing report does not match this checkpoint and selection")
    else:
        args.output.mkdir(parents=True)
        report = {
            "version": 3,
            "status": "running",
            "baseline": None,
            "checkpoint_id": args.checkpoint_id,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "dataset_revision": suite["dataset_revision"],
            "requested_episodes": selected,
            "excluded_practice_episodes": excluded,
            "settings": asdict(settings),
            "scope": "Clean cup training examples replayed for visualization; not held-out or official evaluation",
            "prompt": CUP_TASK,
            "episodes": [],
        }
        write_report(args.output / "report.json", report)

    dataset = Dataset(args.dataset)
    policy = Policy(str(args.checkpoint))
    done = {int(entry["episode_index"]) for entry in report["episodes"] if entry.get("status") == "complete"}
    for index in selected:
        if index in done:
            continue
        print(f"REPLAY_START checkpoint={args.checkpoint_id} episode={index}", flush=True)
        entry = {"episode_index": index, "repeat": 0, "status": "running", "training_overlap": True}
        try:
            episode = dataset.episode(index)
            episode.audit()
            windows_for_summary = []
            windows_for_console = []
            for start, anchors, reference, grippers in episode.windows(settings):
                observation = episode.observation(start)
                # The clean training manifest used the full cup task prompt for
                # every selected episode, not each practice step description.
                observation["prompt"] = CUP_TASK
                begin = time.monotonic()
                actions = policy.infer(observation)
                latency_ms = (time.monotonic() - begin) * 1000
                window = compare(actions, anchors, reference, grippers, settings)
                window.update(frame=start, latency_ms=latency_ms)
                windows_for_summary.append(window)
                windows_for_console.append({key: window[key] for key in (
                    "frame", "predicted_poses", "predicted_grippers",
                    "position_cm", "rotation_deg", "gripper_rad",
                )})
            entry.update(status="complete", frames=episode.length,
                         summary=summarize(windows_for_summary, settings), windows=windows_for_console)
            print(f"REPLAY_COMPLETE checkpoint={args.checkpoint_id} episode={index} windows={len(windows_for_console)}", flush=True)
        except Exception as exc:
            entry.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            print(f"REPLAY_FAILED checkpoint={args.checkpoint_id} episode={index}: {exc}", flush=True)
        report["episodes"] = [old for old in report["episodes"] if old["episode_index"] != index] + [entry]
        write_report(args.output / "report.json", report)
    report["status"] = "complete" if all(
        any(e["episode_index"] == index and e["status"] == "complete" for e in report["episodes"])
        for index in selected
    ) else "partial"
    write_report(args.output / "report.json", report)
    print(f"REPLAY_RESULT checkpoint={args.checkpoint_id} status={report['status']} complete={sum(e['status'] == 'complete' for e in report['episodes'])}/{len(selected)}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
