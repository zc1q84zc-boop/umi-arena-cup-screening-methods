#!/usr/bin/env python3
"""Version the prepared manifest's OpenWAM bimanual slot map."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


root = Path("/mnt/data/benyun/workspace/umi_cup_multimodel_20260925")
manifest_path = root / "openwam/prepared/manifest.json"
backup_path = root / "openwam/prepared/manifest.before_bimanual_slots.json"
temp_path = root / "openwam/prepared/manifest.json.tmp"


def main() -> None:
    if (root / "openwam/checkpoints").exists() or (root / "openwam/smoke").exists():
        raise RuntimeError("refusing slot migration after OpenWAM output exists")
    if backup_path.exists() or temp_path.exists():
        raise FileExistsError("prior slot migration artifact exists")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["total_episodes"], manifest["total_frames"],
            manifest["video_target_frames"], manifest["action_horizon"],
            manifest["active_action_slots"]) != (3423, 746914, 33, 32, list(range(20))):
        raise ValueError("unexpected prepared manifest before slot migration")
    shutil.copy2(manifest_path, backup_path)
    manifest["active_action_slots"] = [*range(10), *range(34, 44)]
    temp_path.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temp_path, manifest_path)
    print("OPENWAM_SLOTS_MIGRATED left=0..9 right=34..43")


if __name__ == "__main__":
    main()
