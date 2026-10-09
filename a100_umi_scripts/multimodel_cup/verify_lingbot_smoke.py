#!/usr/bin/env python3
"""Gate a longer LingBot run on one finite optimizer step and complete DCP."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re


root = Path("/mnt/data/benyun/workspace/umi_cup_multimodel_20260925")
log_path = root / "lingbot/smoke_retry2.log"
checkpoint = root / "lingbot/smoke/checkpoints/global_step_1"
marker = root / "lingbot/smoke_verified.json"


def main() -> None:
    if marker.exists():
        raise FileExistsError(marker)
    text = log_path.read_text()
    match = re.search(r"Step 1/23341, Epoch 1, Loss ([0-9.]+).*?GradNorm ([0-9.]+)", text)
    if not match or not all(math.isfinite(float(value)) and float(value) > 0 for value in match.groups()):
        raise ValueError("LingBot smoke lacks one finite nonzero gradient update")
    if "Reached max_steps=1, stopping training." not in text or "Distributed checkpoint saved" not in text:
        raise ValueError("LingBot smoke did not finish after its checkpoint")
    for part in ("model", "optimizer"):
        directory = checkpoint / part
        if not (directory / ".metadata").is_file() or not list(directory.glob("*.distcp")):
            raise ValueError(f"incomplete LingBot DCP component: {directory}")
    if (root / "lingbot/checkpoints").exists():
        raise FileExistsError("formal LingBot output already exists")
    result = {
        "status": "verified", "optimizer_steps": 1,
        "loss": float(match.group(1)), "gradient_norm": float(match.group(2)),
        "checkpoint": str(checkpoint),
        "clean_episodes": 3423, "clean_frames": 746914,
    }
    temp = marker.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, marker)
    print("LINGBOT_SMOKE_VERIFIED", json.dumps(result))


if __name__ == "__main__":
    main()
