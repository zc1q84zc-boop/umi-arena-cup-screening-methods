#!/usr/bin/env python3
"""Correct the OpenWAM video-target description before any training occurs.

The prepared motion NPZ and video references are unchanged. This is a
versioned metadata migration, not a rewrite of the audited source data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


ROOT = Path("/mnt/data/benyun/workspace/umi_cup_multimodel_20260925")
manifest_path = ROOT / "openwam/prepared/manifest.json"
backup_path = ROOT / "openwam/prepared/manifest.before_temporal_alignment.json"
temp_path = ROOT / "openwam/prepared/manifest.json.tmp"


def main() -> None:
    if (ROOT / "openwam/checkpoints").exists() or (ROOT / "openwam/smoke").exists():
        raise RuntimeError("refusing to change dataset metadata after OpenWAM model output exists")
    if backup_path.exists() or temp_path.exists():
        raise FileExistsError("prior migration artifact exists; inspect before retrying")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["total_episodes"], manifest["total_frames"], manifest["video_history_frames"],
        manifest["action_horizon"], manifest["video_stride"]) != (3423, 746914, 33, 32, 4):
        raise ValueError("unexpected prepared dataset manifest")
    if (len(manifest["train_episodes"]), len(manifest["episodes"]),
            sum(item["frames"] for item in manifest["episodes"].values())) != (3423, 3423, 746914):
        raise ValueError("prepared episode inventory is not complete")
    if manifest["video_semantics"] != "center-camera history t-32..t only; no future video input":
        raise ValueError("unexpected prior video semantics")
    shutil.copy2(manifest_path, backup_path)
    manifest.pop("video_history_frames")
    manifest["video_target_frames"] = 33
    manifest["video_semantics"] = (
        "center-camera t..t+32 target frames (stride 4); only t is an inference condition"
    )
    temp_path.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temp_path, manifest_path)
    print("OPENWAM_TEMPORAL_MANIFEST_MIGRATED 3423 episodes / 746914 frames")


if __name__ == "__main__":
    main()
