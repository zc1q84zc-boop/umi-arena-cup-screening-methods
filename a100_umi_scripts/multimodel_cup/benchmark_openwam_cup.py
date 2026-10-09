#!/usr/bin/env python3
"""Bounded real-data OpenWAM training benchmark; never save trained weights.

Uses the checkpoint's audited recipe and the upstream training loop, including
backward, gradient clipping, DeepSpeed CPU Adam and real optimizer updates.
The prepared manifest retains the complete audited allowlist, while an explicit
benchmark selection limits reads to files copied to this private test directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "scripts", "prepared", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed", type=int, default=12)
    args = parser.parse_args()
    if min(args.batch, args.accum, args.timed) < 1 or args.warmup < 0:
        parser.error("batch, accum and timed must be positive")
    if (args.warmup + args.timed) % args.accum or args.warmup % args.accum:
        parser.error("warmup and timed must contain complete accumulation cycles")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.scripts), str(args.repo)]
    # Python 3.12/DeepSpeed compatibility, as in the original training wrapper.
    import setuptools  # noqa: F401
    import torch
    import pynvml
    from omegaconf import OmegaConf
    from openwam_cup_dataset import CupDataset
    from openwam.train.openwam_trainer import OpenWAMTrainer

    selection = json.loads((args.prepared / "benchmark_selection.json").read_text())
    dataset = CupDataset(args.prepared)
    selected = set(selection["episode_ids"])
    dataset.samples = [(episode, frame) for episode, frame in dataset.samples if episode in selected]
    if not dataset.samples or set(episode for episode, _ in dataset.samples) != selected:
        raise RuntimeError("benchmark selection differs from the audited source manifest")
    manifest = dataset.manifest
    cfg = OmegaConf.load(args.checkpoint / "config.yaml")
    cfg.dataloader.prepared_dir = str(args.prepared)
    cfg.dataloader.dataset_dir = str(args.prepared)
    cfg.training.batch_size = args.batch
    cfg.training.gradient_accumulation_steps = args.accum
    cfg.training.max_steps = args.warmup + args.timed
    cfg.training.num_epochs = None
    cfg.training.finetune_ckpt_path = str(args.checkpoint)
    cfg.training.resume_ckpt_path = None
    cfg.training.save_steps = 0
    cfg.training.save_full_states_for_resume = False
    cfg.training.output_path = str(args.output / "metadata")
    cfg.project.output_dir = str(args.output / "metadata")
    cfg.project.wandb.project = None
    cfg.project.seed = 42
    OmegaConf.save(cfg, args.output / "benchmark_config.yaml")
    metadata = {
        "host": os.uname().nodename, "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint": str(args.checkpoint), "micro_batch": args.batch,
        "gradient_accumulation": args.accum, "warmup_micro_steps": args.warmup,
        "timed_micro_steps": args.timed, "benchmark_windows": len(dataset),
        "full_audited_windows": sum(manifest["episodes"][str(i)]["frames"] - 32
                                    for i in manifest["train_episodes"]),
        "selection": selection, "records": [], "started_at": time.time(),
        "scope": "real training; warm-start for throughput only; local subset I/O; no quality evaluation",
    }
    started = time.monotonic()
    phase = "accelerator_initialization"
    pynvml.nvmlInit()
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    nvml_handle = (pynvml.nvmlDeviceGetHandleByUUID(visible_device)
                   if visible_device.startswith("GPU-") else
                   pynvml.nvmlDeviceGetHandleByIndex(int(visible_device)))
    metadata["gpu_total_memory_mib"] = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).total / 2**20

    spec = importlib.util.spec_from_file_location("openwam_benchmark_entry", args.repo / "scripts/train.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    class BenchmarkTrainer(OpenWAMTrainer):
        def prepare_accelerate(self, *arguments):
            result = super().prepare_accelerate(*arguments)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._last_end = time.monotonic()
            self._current_batch = 0
            metadata["setup_seconds"] = self._last_end - started
            metadata["gpu"] = torch.cuda.get_device_name()
            return result

        def compute_loss(self, batch):
            torch.cuda.synchronize()
            self._compute_start = time.monotonic()
            self._current_batch = len(batch) if isinstance(batch, list) else 1
            # Same preparation and loss calls as the upstream trainer, with a
            # cardinality check so a nominal batch cannot silently train only
            # its first sample.
            if not isinstance(batch, list):
                batch = [batch]
            inputs = self.architecture.prepare_inputs(batch)
            actions = inputs.get("actions")
            if actions is None or tuple(actions.shape) != (len(batch), 32, 80):
                raise RuntimeError(f"unexpected real model action batch: {None if actions is None else actions.shape}")
            self._input_action_shape = list(actions.shape)
            loss_result = self.architecture.compute_loss(
                **inputs, lambda_video=self.lambda_video, lambda_action=self.lambda_action)
            result = {"total": loss_result["loss"],
                      "video": loss_result.get("loss_video", torch.tensor(0.0)),
                      "action": loss_result.get("loss_action", torch.tensor(0.0))}
            torch.cuda.synchronize()
            self._forward_end = time.monotonic()
            return result

        def log_step(self, **kwargs):
            torch.cuda.synchronize()
            now = time.monotonic()
            metrics = {k: float(v) for k, v in kwargs["metrics"].items()}
            if not all(math.isfinite(v) for v in metrics.values()):
                raise RuntimeError(f"non-finite training metric: {metrics}")
            row = {
                "micro_step": kwargs["global_step"], "optimizer_step": kwargs["opt_step"],
                "samples": self._current_batch * self.accelerator.num_processes,
                "model_action_batch_shape": self._input_action_shape,
                "seconds": now - self._last_end,
                "data_wait_seconds": self._compute_start - self._last_end,
                "forward_seconds": self._forward_end - self._compute_start,
                "backward_update_metric_seconds": now - self._forward_end,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "device_used_mib": pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / 2**20,
                "metrics": metrics, "warmup": kwargs["global_step"] <= args.warmup,
            }
            metadata["records"].append(row)
            with (args.output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print("BENCH_STEP " + json.dumps(row), flush=True)
            self._last_end = now
            kwargs["pbar"].update(1)

    try:
        accelerator = entry._build_accelerator(cfg)
        phase = "model_initialization"
        trainer = BenchmarkTrainer(cfg, accelerator=accelerator, dataset=dataset)
        metadata["trainable_parameters"] = sum(p.numel() for p in trainer.architecture.parameters()
                                               if p.requires_grad)
        phase = "training"
        trainer.train()
        timed = [r for r in metadata["records"] if not r["warmup"]]
        seconds = sum(r["seconds"] for r in timed)
        samples = sum(r["samples"] for r in timed)
        previous_opt = metadata["records"][args.warmup - 1]["optimizer_step"] if args.warmup else 0
        updates = metadata["records"][-1]["optimizer_step"] - previous_opt
        if len(timed) != args.timed or updates != args.timed // args.accum:
            raise RuntimeError("optimizer/micro-step count does not match the requested benchmark")
        metadata.update(status="ok", timed_samples=samples, timed_seconds=seconds,
                        timed_optimizer_updates=updates, samples_per_second=samples / seconds,
                        median_micro_step_seconds=statistics.median(r["seconds"] for r in timed),
                        peak_allocated_mib=max(r["peak_allocated_mib"] for r in metadata["records"]),
                        peak_reserved_mib=max(r["peak_reserved_mib"] for r in metadata["records"]),
                        peak_sampled_device_used_mib=max(r["device_used_mib"] for r in metadata["records"]))
    except BaseException as error:
        metadata.update(status="oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error",
                        phase=phase, error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            metadata.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        raise
    finally:
        metadata["wall_seconds"] = time.monotonic() - started
        (args.output / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("BENCH_RESULT " + json.dumps({k: v for k, v in metadata.items()
                                          if k not in ("records", "traceback", "selection")}), flush=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
