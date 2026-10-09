#!/usr/bin/env python3
"""Confirm the two-step OpenWAM smoke run is complete before formal training."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from safetensors import safe_open


WORK = Path("/mnt/data/benyun/workspace/umi_cup_multimodel_20260925")
LOG = WORK / "openwam/smoke_retry3.log"
METRICS = WORK / "openwam/gradient_metrics.jsonl"
SMOKE = WORK / "openwam/smoke"
OUTPUT = WORK / "openwam/smoke_verified.json"


def verify() -> dict:
    if OUTPUT.exists():
        raise RuntimeError("smoke already verified; refusing to overwrite")
    runs = [p for p in SMOKE.iterdir() if p.is_dir()]
    if len(runs) != 1:
        raise RuntimeError(f"expected one smoke run, found {len(runs)}")
    run = runs[0]
    weight = run / "checkpoint_step_2.safetensors"
    if not weight.is_file() or weight.stat().st_size < 10_000_000_000:
        raise RuntimeError("step-2 weights are absent or not fully written")
    if not (run / "config.yaml").is_file():
        raise RuntimeError("deploy config absent")
    if not (run / "normalization_stats.npy").is_file():
        raise RuntimeError("action normalization absent")
    log = LOG.read_text(errors="replace")
    if "Saving step 2" not in log or "[checkpoint] Saved:" not in log:
        raise RuntimeError("trainer has not reported a completed smoke checkpoint")
    entries = [json.loads(line) for line in METRICS.read_text().splitlines() if line.strip()]
    smoke_entries = [entry for entry in entries if entry["step"] in (1, 2)]
    if len(smoke_entries) != 2 or [entry["step"] for entry in smoke_entries] != [1, 2]:
        raise RuntimeError("both smoke optimizer steps were not recorded exactly once")
    for entry in smoke_entries:
        metrics = entry["metrics"]
        if entry["optimizer_step"] != entry["step"]:
            raise RuntimeError("optimizer did not update on each smoke step")
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("non-finite smoke metrics")
        if not (metrics["loss_action"] > 0 and metrics["grad_norm"] > 0):
            raise RuntimeError("no evidence of an action-gradient update")
    with safe_open(weight, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
    if len(keys) < 100:
        raise RuntimeError("smoke weight file has too few tensors")
    return {
        "status": "verified",
        "steps": [entry["step"] for entry in smoke_entries],
        "metrics": [entry["metrics"] for entry in smoke_entries],
        "checkpoint": str(weight),
        "checkpoint_bytes": weight.stat().st_size,
        "checkpoint_tensor_count": len(keys),
        "training_data_episodes": 3423,
        "training_data_frames": 746914,
        "trainable_components": ["action_backbone", "proprio_encoder"],
        "frozen_components": ["video_backbone"],
    }


if __name__ == "__main__":
    result = verify()
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
