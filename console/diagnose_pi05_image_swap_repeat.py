#!/usr/bin/env python3
"""Interleaved repeated image-swap probe; read-only on recorded data."""

import json
import random
import statistics

from diagnose_pi05_real_observation import infer, summarize


def probe(frame, conditions, repeats):
    jobs = [(name, repetition) for repetition in range(repeats) for name in conditions]
    random.Random(20260929 + frame).shuffle(jobs)
    values = {name: [] for name in conditions}
    for name, _ in jobs:
        response, request = infer(frame, name)
        row = summarize(frame, response, request, name)
        values[name].append(1000 * row["predicted_right_source_world_delta_xyz"][2])
    return {name: {"z_mm": [round(x, 3) for x in series],
                   "mean_z_mm": round(statistics.mean(series), 3),
                   "sd_z_mm": round(statistics.stdev(series), 3)}
            for name, series in values.items()}


if __name__ == "__main__":
    print(json.dumps({"frame": 0, "repeats": 6,
                      "conditions": probe(0, ("recorded", "replay_aligned_sim",
                                              "replay_aligned_sim_light500",
                                              "replay_aligned_sim_emissive",
                                              "simulated", "real_left_sim_right",
                                              "sim_left_real_right"), 6)},
                     separators=(",", ":")), flush=True)
    print(json.dumps({"frame": 30, "repeats": 6,
                      "conditions": probe(30, ("recorded", "replay_aligned_sim", "simulated"), 6)},
                     separators=(",", ":")), flush=True)
