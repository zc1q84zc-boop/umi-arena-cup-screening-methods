#!/usr/bin/env python3
"""Guard the official LingBot trainer with the audited cup episode allowlist."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy


WORK = Path("/mnt/data/benyun/workspace/umi_cup_multimodel_20260925")
DATASET = Path("/mnt/data/benyun/workspace/umi_cup_review_20260923/clean_cup_v1/clean_dataset")
REPO = Path("/mnt/data/benyun/workspace/projects/lingbot-vla-v2")


def install_clean_filter() -> None:
    import lingbotvla.data.vla_data.base_dataset as base

    verified = json.loads((WORK / "selection/verified_allowlist.json").read_text())
    raw = (WORK / "selection/manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != verified["manifest_sha256"]:
        raise RuntimeError("clean manifest changed after verification")
    if (verified["episodes"], verified["frames"], verified["quarantined_episodes"]) != (3423, 746914, 552):
        raise RuntimeError("clean-selection audit count changed")
    selected = list(verified["episode_ids"])
    original = base.LeRobotDataset

    class AuditedLeRobotDataset(original):
        def __init__(self, repo_id: str, load_image: bool = True, **kwargs):
            if Path(repo_id).resolve() != DATASET.resolve():
                raise RuntimeError(f"refusing a non-clean dataset: {repo_id}")
            if "episodes" in kwargs:
                raise RuntimeError("episode selection must come from the verified allowlist")
            kwargs["episodes"] = selected
            kwargs["video_backend"] = "pyav"
            super().__init__(repo_id, load_image=load_image, **kwargs)
            if len(self) != 746914 or set(self.episodes) != set(selected):
                raise RuntimeError("LingBot loader escaped the clean episode selection")

    base.LeRobotDataset = AuditedLeRobotDataset


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    install_clean_filter()
    runpy.run_path(str(REPO / "tasks/vla/train_lingbotvla.py"), run_name="__main__")
