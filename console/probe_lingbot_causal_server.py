#!/usr/bin/env python3
"""Probe a private LingBot server with sequential, previously rendered sim observations.

This checks serving and action timing only. It is not a live or task-success test.
"""

import argparse
import base64
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18811/infer")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--expected-model")
    args = parser.parse_args()
    if not 1 <= args.steps <= 3:
        raise ValueError("probe is limited to three saved current-frame observations")
    rows = [json.loads(line) for line in (args.source_run / "online_adapter.jsonl").read_text().splitlines()]
    results = []
    provenance = None
    for step, row in enumerate(rows[: args.steps]):
        if row["step"] != step or row["observation_origin"] != "current_simulator_render_and_robot_state":
            raise ValueError("saved observation sequence is not causal")
        pose = row["model_input_pose"]
        payload = {
            "episode": 0,
            "step": step,
            "pose_frame": "source_hand_reference_259632_v1",
            "relative_pose_xyzw": row["relative_pose_xyzw"],
            "gripper_rad": row["source_gripper_rad"],
            "prompt": "Pick up the cup with your right hand and set it on the plate",
        }
        for side in ("left", "right"):
            payload[f"{side}_pose_wxyz"] = [*pose[side]["position_m"], *pose[side]["quaternion_wxyz"]]
            payload[f"{side}_jpeg"] = base64.b64encode(
                (args.source_run / f"input_{side}_{step:04d}.jpg").read_bytes()
            ).decode("ascii")
        request = Request(args.url, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=120) as response:
            prediction = json.load(response)
        actions = prediction["actions"]
        if args.expected_model:
            provenance = prediction.get('model_provenance')
            if (prediction.get('model_id') != args.expected_model or not provenance
                    or provenance.get('fine_tuned') is not False
                    or not provenance.get('all_files_sha256_verified')):
                raise ValueError('Expected verified official pretrained model identity')
        if (prediction["step"] != step or prediction["episode"] != 0
                or prediction["action_timing"] != {"pose_rows": [1, 2, 3], "gripper_rows": [0, 1, 2]}
                or prediction["pose_mapping"] != "source_hand_reference_259632_v1"
                or len(actions) != 3 or any(len(action) != 16 for action in actions)
                or not all(math.isfinite(value) for action in actions for value in action)):
            raise ValueError("server returned non-causal, non-finite, or malformed actions")
        results.append({"step": step, "action_rows": len(actions),
                        "latency_ms": prediction["latency_ms"]})
    if len(results) != args.steps:
        raise ValueError("fewer saved observations than requested")
    print(json.dumps({"status": "causal_server_probe_passed", "saved_sim_steps": results,
                      "expected_model": args.expected_model, "model_provenance": provenance,
                      "live_simulation": False, "task_success": False}))


if __name__ == "__main__":
    main()
