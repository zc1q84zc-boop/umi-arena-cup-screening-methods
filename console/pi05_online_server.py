#!/usr/bin/env python3
"""Loopback-only π0.5 inference bridge for live Isaac Sim wrist observations.

Run on a private GPU host with CUDA_VISIBLE_DEVICES restricted to a verified
free GPU. A weights-only inference export is accepted with an explicit marker.
Only the two current wrist RGB frames and current robot state are accepted.
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


CUP_TASK = "Place the cup on the plate, then put it back to its original position"
MAX_BODY = 4_000_000
ACTION_TIMING = {
    "pose_rows": [1, 2, 3],
    "gripper_rows": [0, 1, 2],
    "basis": "pose[t] is previous-to-current; gripper[t] targets next frame",
}


def decode_image(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    if len(raw) > 2_000_000:
        raise ValueError("image too large")
    with Image.open(BytesIO(raw)) as image:
        result = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if result.shape != (480, 640, 3):
        raise ValueError(f"wrist image must be 480x640 RGB, got {result.shape}")
    return result


def build_observation(payload: dict) -> dict:
    if set(payload) != {"left_jpeg", "right_jpeg", "relative_pose_xyzw", "gripper_rad", "step"}:
        raise ValueError("unexpected observation fields")
    pose = np.asarray(payload["relative_pose_xyzw"], dtype=np.float32)
    gripper = np.asarray(payload["gripper_rad"], dtype=np.float32)
    if pose.shape != (7,) or gripper.shape != (2,) or not np.isfinite(pose).all() or not np.isfinite(gripper).all():
        raise ValueError("invalid pose or gripper state")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("invalid step")
    return {
        "observation.image.left": decode_image(payload["left_jpeg"]),
        "observation.image.right": decode_image(payload["right_jpeg"]),
        "observation.pose.left_hand_root_to_right_hand_root.absolute": pose,
        "observation.joint_states": gripper,
        "prompt": CUP_TASK,
    }


def next_three_causal_actions(raw_actions: np.ndarray) -> np.ndarray:
    """Align differently timed pose and gripper labels to three future slots.

    Training starts the 30 Hz pose sequence at observation t, but the stored
    `observation.pose.*.relative[t]` is motion from t-1 to t. Conversely,
    `action.joint_states[t]` is the jaw target at t+1. Thus the first future
    pose is row 1 and the first future jaw target is row 0.
    """
    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.shape != (32, 16) or not np.isfinite(actions).all():
        raise ValueError(f"invalid action shape or values: {actions.shape}")
    return np.concatenate((actions[1:4, :14], actions[:3, 14:16]), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", default="pi05-cup-clean-30000")
    parser.add_argument("--port", type=int, default=18781)
    args = parser.parse_args()
    allowed_models = {f"pi05-cup-clean-{step}": step for step in (10000, 20000, 30000)}
    if args.model_id not in allowed_models:
        parser.error("unsupported clean-cup π0.5 checkpoint")
    required = ("params", "assets", "_CHECKPOINT_METADATA")
    if not all((args.checkpoint / key).exists() for key in required):
        parser.error("incomplete π0.5 checkpoint")
    if not (args.checkpoint / "train_state").exists():
        marker = args.checkpoint / "inference_export.json"
        if not marker.is_file():
            parser.error("weights-only export requires inference_export.json")
        provenance = json.loads(marker.read_text())
        if (provenance.get("model") != args.model_id
                or provenance.get("source_step") != allowed_models[args.model_id]
                or provenance.get("export_type") != "params_and_assets_only"):
            parser.error("weights-only export provenance mismatch")
    sys.path[:0] = [str(args.evaluation_root), str(args.openpi_root / "src"), str(args.evaluation_root / "adapters/openpi_pi05")]
    from policy import Policy

    policy = Policy(str(args.checkpoint))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self._reply(200, {"ready": True, "model": args.model_id, "checkpoint": str(args.checkpoint)})

        def do_POST(self):
            if self.path != "/infer":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size))
                observation = build_observation(payload)
                started = time.monotonic()
                actions = next_three_causal_actions(policy.infer(observation))
                elapsed_ms = (time.monotonic() - started) * 1000
                self._reply(200, {"step": payload["step"], "actions": actions.tolist(),
                                  "action_timing": ACTION_TIMING, "latency_ms": elapsed_ms})
            except Exception as exc:
                self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})

        def _reply(self, status: int, data: dict) -> None:
            body = json.dumps(data, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PI05_ONLINE_READY port={args.port} checkpoint={args.checkpoint}", flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
