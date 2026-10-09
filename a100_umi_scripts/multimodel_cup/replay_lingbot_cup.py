#!/usr/bin/env python3
"""Offline LingBot replay on the same clean practice windows as π0.5.

This is a training-overlap visualization, not a held-out or robot evaluation.
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

import numpy as np


TASK = "Place the cup on the plate, then put it back to its original position"
ACTION_POSE_KEYS = (
    "observation.pose.left_hand_root.relative",
    "observation.pose.right_hand_root.relative",
)


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--lingbot-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--hf-checkpoint", required=True, type=Path)
    parser.add_argument("--robot-config-root", required=True, type=Path)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=None,
                        help="process at most this many new episodes before exiting (for smoke validation)")
    args = parser.parse_args()
    if not args.checkpoint_id.startswith("lingbot-cup-clean-"):
        parser.error("checkpoint ID must identify the clean-cup LingBot run")
    if not list(args.hf_checkpoint.glob("*.safetensors")):
        parser.error("no completed HF-format LingBot weights")
    if args.robot_config_root.name != "configs" or not (args.robot_config_root / "robot_configs/umi_cup_clean.yaml").is_file():
        parser.error("workspace robot config is absent")

    sys.path[:0] = [str(args.evaluation_root), str(args.lingbot_root),
                    str(args.lingbot_root / "deploy")]
    from umi_arena.replay_data import Dataset
    from umi_arena.replay_metrics import Settings, compare, summarize
    from lingbot_vla_v2_policy import LingbotVLAv2Server

    suite = json.loads(args.suite.read_text())
    clean = json.loads(args.train_manifest.read_text())
    clean_ids = {int(row["episode_index"]) for row in clean["episodes"]}
    if len(clean_ids) != 3423:
        raise ValueError("training manifest is not the audited clean cup selection")
    task = [x for x in suite["tasks"] if x["dataset_task"] == TASK]
    if len(task) != 1:
        raise ValueError("practice suite must contain exactly one cup task")
    suite_ids = [int(i) for recording in task[0]["recordings"] for i in recording["episodes"]]
    selected = [i for i in suite_ids if i in clean_ids]
    excluded = [i for i in suite_ids if i not in clean_ids]
    if not selected:
        raise ValueError("no clean practice cup episodes")
    # The policy only sees the recorded observation at `start`. Its pose label
    # row zero is observation.pose.*.relative at that same frame (past motion),
    # while action.joint_states row zero targets the next frame.
    settings = Settings(30, "body", "next")
    causal_alignment = "observation at frame t; pose rows 1..16 and gripper rows 0..15 target frames t+1..t+16"

    if args.output.exists():
        if not args.resume:
            parser.error(f"output already exists: {args.output}")
        report = json.loads((args.output / "report.json").read_text())
        if report.get("checkpoint_id") != args.checkpoint_id or report.get("requested_episodes") != selected:
            raise ValueError("existing report does not match checkpoint and selection")
        if report.get("settings", {}).get("pose_timing") != "next" or report.get("causal_alignment") != causal_alignment:
            raise ValueError("existing report does not use causal next-frame alignment")
    else:
        args.output.mkdir(parents=True)
        report = {
            "version": 4, "status": "running", "baseline": None,
            "checkpoint_id": args.checkpoint_id,
            "checkpoint_path": str(args.hf_checkpoint.resolve()),
            "dataset_revision": suite["dataset_revision"],
            "requested_episodes": selected,
            "excluded_practice_episodes": excluded,
            "settings": asdict(settings),
            "causal_alignment": causal_alignment,
            "scope": "Clean cup training examples for visualization; not held-out or official evaluation",
            "suite_task": TASK, "prompt_source": "dataset task_index (primitive action)",
            "episodes": [],
        }
        atomic_json(args.output / "report.json", report)

    # The deployment wrapper resolves configs/robot_configs relative to cwd.
    os.chdir(args.robot_config_root.parent)
    policy = LingbotVLAv2Server(path_to_pi_model=str(args.hf_checkpoint),
                                robot_norm_path=None, chunk_ret=True,
                                use_length=32, use_compile=False)
    policy.infer({"reset": True, "robo_name": "umi_cup_clean"})
    dataset = Dataset(args.dataset)
    done = {int(e["episode_index"]) for e in report["episodes"] if e.get("status") == "complete"}
    processed = 0
    for index in selected:
        if index in done:
            continue
        if args.limit_episodes is not None and processed >= args.limit_episodes:
            break
        print(f"LINGBOT_REPLAY_START checkpoint={args.checkpoint_id} episode={index}", flush=True)
        entry = {"episode_index": index, "repeat": 0, "status": "running", "training_overlap": True}
        try:
            episode = dataset.episode(index)
            episode.audit()
            windows_for_summary, windows_for_console = [], []
            for start, anchors, reference, grippers in episode.windows(settings):
                observation = episode.observation(start)
                for hand in ("left", "right"):
                    key = f"observation.pose.{hand}_hand_root.absolute"
                    observation[key] = np.asarray(episode.data[key][start], dtype=np.float32)
                begin = time.monotonic()
                prediction = policy.infer(observation)
                latency_ms = (time.monotonic() - begin) * 1000
                parts = [np.asarray(prediction[key], dtype=np.float32) for key in ACTION_POSE_KEYS]
                grip = np.asarray(prediction["action.joint_states"], dtype=np.float32)
                if any(part.ndim != 2 or part.shape[1] != 7 for part in parts) or grip.ndim != 2 or grip.shape[1] != 2:
                    raise ValueError("LingBot action outputs are not left7/right7/gripper2 chunks")
                if (len(parts[0]) < len(reference) + 1 or
                        len(parts[1]) < len(reference) + 1 or
                        len(grip) < len(reference)):
                    raise ValueError("LingBot chunk is too short for causal pose/gripper alignment")
                actions = np.concatenate([parts[0][1:len(reference)+1],
                                          parts[1][1:len(reference)+1],
                                          grip[:len(reference)]], axis=1)
                if not np.isfinite(actions).all():
                    raise ValueError("LingBot action chunk is short or non-finite")
                window = compare(actions, anchors, reference, grippers, settings)
                window.update(frame=start, latency_ms=latency_ms)
                windows_for_summary.append(window)
                windows_for_console.append({key: window[key] for key in (
                    "frame", "predicted_poses", "predicted_grippers",
                    "position_cm", "rotation_deg", "gripper_rad",
                )})
            entry.update(status="complete", frames=episode.length,
                         summary=summarize(windows_for_summary, settings), windows=windows_for_console)
            print(f"LINGBOT_REPLAY_COMPLETE checkpoint={args.checkpoint_id} episode={index} windows={len(windows_for_console)}", flush=True)
        except Exception as exc:
            entry.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            print(f"LINGBOT_REPLAY_FAILED checkpoint={args.checkpoint_id} episode={index}: {exc}", flush=True)
        report["episodes"] = [old for old in report["episodes"] if old["episode_index"] != index] + [entry]
        atomic_json(args.output / "report.json", report)
        processed += 1
        if entry["status"] == "failed" and args.limit_episodes is not None:
            break
    report["status"] = "complete" if all(
        any(e["episode_index"] == i and e["status"] == "complete" for e in report["episodes"])
        for i in selected
    ) else "partial"
    atomic_json(args.output / "report.json", report)
    print(f"LINGBOT_REPLAY_RESULT {report['status']} complete={sum(e['status'] == 'complete' for e in report['episodes'])}/{len(selected)}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
