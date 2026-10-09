#!/usr/bin/env python3
"""Compare identical commanded waypoint streams at fixed simulation time."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(directory: Path) -> dict:
    with (directory / "joints.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    commands = [json.loads(line) for line in (directory / "continuous_targets.jsonl").read_text().splitlines()]
    q_command = np.asarray([row["q"] for row in commands], dtype=float)
    v_command = np.asarray([row["v"] for row in commands], dtype=float)
    if len(q_command) % 2 or not np.isfinite(q_command).all():
        raise ValueError("missing or nonfinite 60 Hz joint references")
    observed_q = []
    for side in ("left", "right"):
        joints = []
        for index in range(1, 8):
            samples = [float(row["position"]) for row in rows
                       if row["side"] == side and row["joint_name"] == f"panda_joint{index}"]
            joints.append(samples)
        observed_q.append(np.asarray(joints).T)
    observed_q = np.asarray(observed_q).transpose(1, 0, 2)
    target_30hz = q_command[1::2, :, :7]
    if observed_q.shape[0] != len(target_30hz) + 1:
        raise ValueError("joint samples and command references are not aligned")
    observed_v = np.diff(observed_q, axis=0) * 30.0
    observed_a = np.diff(observed_v, axis=0) * 30.0
    reversals = ((observed_v[1:] * observed_v[:-1] < 0)
                 & (np.abs(observed_v[1:]) > .02)
                 & (np.abs(observed_v[:-1]) > .02))
    report = json.loads((directory / "report.json").read_text())["episodes"][0]
    return {
        "run": directory.name,
        "steps_10hz": report["policy_steps"],
        "actual_velocity_p95_rad_s": float(np.percentile(np.abs(observed_v), 95)),
        "actual_velocity_max_rad_s": float(np.abs(observed_v).max()),
        "actual_acceleration_p95_rad_s2": float(np.percentile(np.abs(observed_a), 95)),
        "actual_acceleration_max_rad_s2": float(np.abs(observed_a).max()),
        "velocity_reversals_gt_0_02_rad_s": int(reversals.sum()),
        "target_tracking_error_p95_rad": float(np.percentile(
            np.abs(observed_q[1:] - target_30hz), 95)),
        "command_velocity_max_rad_s": float(np.abs(v_command[:, :, :7]).max()),
        "cup_displacement_mm": 1000 * float(report["object_displacement_m"]["cup"]),
        "task_success": bool(report["success"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.runs], indent=2))
