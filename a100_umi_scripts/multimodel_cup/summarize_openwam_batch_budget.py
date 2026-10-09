#!/usr/bin/env python3
"""Convert measured training throughput into explicit sample/time budgets."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def budget(samples: int, batch: int, accum: int, world: int, rate: float) -> dict:
    updates = math.ceil(samples / (batch * accum * world))
    micro_steps = updates * accum
    delivered = micro_steps * batch * world
    return {"requested_sample_presentations": samples,
            "scheduled_sample_presentations": delivered,
            "micro_steps_per_rank": micro_steps, "optimizer_updates": updates,
            "estimated_wall_hours": delivered / rate / 3600,
            "estimated_gpu_hours": delivered / rate / 3600 * world,
            "full_window_pool_ratio": delivered / 637378}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--gpu-memory-mib", type=float)
    parser.add_argument("--search-complete", action="store_true",
                        help="set only after batch search and selected-candidate repeat finish")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.world != 1:
        parser.error("this benchmark measures a single GPU; do not extrapolate unmeasured multi-GPU throughput")
    output = {
        "status": "awaiting_new_measurements", "candidate_results": [],
        "batch_search_complete": args.search_complete,
        "effective_batch_formula": "micro_batch_per_gpu * world_size * gradient_accumulation",
        "samples_formula": "micro_steps_per_rank * micro_batch_per_gpu * world_size",
        "openwam_max_steps_unit": "micro-steps, not optimizer updates",
        "dataset": {"episodes": 3423, "frames": 746914, "valid_32_action_windows": 637378},
        "selection_rule": "choose highest measured samples/s among stable candidates with >=10% VRAM headroom; rerun the selected candidate before long training",
        "sample_milestones": [50000, 160000, 320000, 637378, 960000],
        "time_policy": {
            "initial_gpu_hours_cap": 24,
            "evaluation_interval": "every 8 GPU-hours or 50000 sample presentations, whichever is earlier",
            "extensions_gpu_hours": [48, 72],
            "extension_condition": "independent task success improves beyond evaluation uncertainty; otherwise inspect stage transition, labels and input/prompt consistency before spending more compute",
            "note": "pilot caps are planning choices; they are not claims that a model needs these amounts",
        },
        "validation": {
            "split_unit": "complete recording_uuid; both ordered subtask episodes stay in the same split",
            "existing_checkpoint_limitation": "all selected recordings were in the old training dataset; they cannot be labelled an independent validation split retroactively",
            "independent_data_options": ["new recordings not used in old training", "fresh training from the foundation checkpoint with a recording-level split and normalization fitted on training only"],
            "offline_metrics": ["next-step aligned active-arm translation/rotation error", "gripper error", "inactive-arm drift", "metrics separately for right pickup, left return and transition"],
            "primary_metric": "closed-loop ordered full-task success: right-hand cup-to-plate then left-hand cup-to-original-location",
            "evaluation_conditions": "same observations, action timing, environment, task definitions and control wrapper; report any external stage selector or assistance",
            "current_live_evaluation_status": "Isaac startup failure must be resolved before new full-task success scores can be obtained",
        },
    }
    candidates = []
    for path in args.result:
        row = json.loads(path.read_text())
        brief = {k: row.get(k) for k in ("status", "host", "gpu", "micro_batch", "gradient_accumulation",
                                       "samples_per_second", "peak_reserved_mib", "timed_samples",
                                       "peak_sampled_device_used_mib", "timed_seconds", "timed_optimizer_updates", "phase", "error")}
        brief["path"] = str(path)
        output["candidate_results"].append(brief)
        if row.get("status") == "ok":
            reserved = row.get("peak_sampled_device_used_mib", row["peak_reserved_mib"])
            if args.gpu_memory_mib is None or reserved <= args.gpu_memory_mib * 0.9:
                candidates.append(row)
    if candidates:
        chosen = max(candidates, key=lambda row: row["samples_per_second"])
        output["status"] = "measured_budget_available" if args.search_complete else "partial_measurements"
        output["budget_is_provisional"] = not args.search_complete
        output["chosen_micro_batch"] = chosen["micro_batch"]
        output["chosen_accumulation"] = chosen["gradient_accumulation"]
        output["measured_samples_per_second"] = chosen["samples_per_second"]
        output["measured_budget"] = [budget(n, chosen["micro_batch"], chosen["gradient_accumulation"],
                                             args.world, chosen["samples_per_second"])
                                       for n in output["sample_milestones"]]
    if args.historical:
        history = json.loads(args.historical.read_text())
        rate = history["batch"] / history["mean_seconds_per_micro_step"]
        output["historical_baseline"] = history
        output["historical_baseline"]["label"] = "old A100 batch1 timing; not a new batch benchmark or a 5090 speed estimate"
        output["historical_budget_only"] = [budget(n, 1, 1, 1, rate) for n in output["sample_milestones"]]
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: output[k] for k in ("status", "candidate_results", "sample_milestones")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
