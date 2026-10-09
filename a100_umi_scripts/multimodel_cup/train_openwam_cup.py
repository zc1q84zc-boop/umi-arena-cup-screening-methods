#!/usr/bin/env python3
"""Register the isolated clean-cup reader, then invoke the official OpenWAM trainer."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

import setuptools  # noqa: F401 -- Python 3.12/DeepSpeed distutils compatibility


WORK = Path("/mnt/data/benyun/workspace/umi_cup_multimodel_20260925")
REPO = Path("/mnt/data/benyun/workspace/openwam_charger_full_20260922/OpenWAM")
PREPARED = WORK / "openwam/prepared"
sys.path[:0] = [str(WORK / "scripts"), str(REPO)]

manifest = json.loads((PREPARED / "manifest.json").read_text())
if (manifest["total_episodes"], manifest["total_frames"]) != (3423, 746914):
    raise RuntimeError("refusing OpenWAM training on an unaudited dataset")

from openwam_cup_dataset import CupDataset  # noqa: E402
from openwam.dataloader.registry import register_dataset  # noqa: E402

register_dataset("umi_cup_clean")(CupDataset)

# DeepSpeed's NVML helper may not accept a GPU UUID in CUDA_VISIBLE_DEVICES.
from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator  # noqa: E402

original_nvml_id = CUDA_Accelerator._get_nvml_gpu_id


def nvml_id(self, device_index=None):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if devices and devices[0].startswith("GPU-"):
        import pynvml
        pynvml.nvmlInit()
        index = self.current_device() if device_index is None else device_index
        handle = pynvml.nvmlDeviceGetHandleByUUID(devices[index])
        return pynvml.nvmlDeviceGetIndex(handle)
    return original_nvml_id(self, device_index)


CUDA_Accelerator._get_nvml_gpu_id = nvml_id

import openwam.train.openwam_trainer as trainer_module  # noqa: E402

# Preserve complete states for readback and safe resume; do not delete saves.
trainer_module.finalize_keep_weights_only = lambda output_path, keep_last_k: None
original_log_step = trainer_module.OpenWAMTrainer.log_step


def audited_log_step(self, **kwargs):
    metrics = {key: float(value) for key, value in kwargs["metrics"].items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite OpenWAM training metrics: {metrics}")
    if self.accelerator.is_main_process:
        with (WORK / "openwam/gradient_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": time.time(), "step": kwargs["global_step"],
                                     "optimizer_step": kwargs["opt_step"], "metrics": metrics}) + "\n")
    return original_log_step(self, **kwargs)


trainer_module.OpenWAMTrainer.log_step = audited_log_step
runpy.run_path(str(REPO / "scripts/train.py"), run_name="__main__")
