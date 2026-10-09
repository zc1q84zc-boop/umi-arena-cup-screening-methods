#!/usr/bin/env python3
"""Loopback-only π0.5 inference bridge for live Isaac Sim wrist observations.

Run on the A100 with CUDA_VISIBLE_DEVICES restricted to a verified free GPU.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18781)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    if args.model_id not in ("pi05-cup-clean-10000", "pi05-cup-clean-20000"):
        parser.error("only audited clean-cup 10k/20k checkpoints are supported")
    if args.checkpoint.name != args.model_id.rsplit("-", 1)[-1]:
        parser.error("model ID does not match checkpoint step")
    if not all((args.checkpoint / key).exists() for key in ("params", "assets", "train_state", "_CHECKPOINT_METADATA")):
        parser.error("incomplete π0.5 checkpoint")
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
                actions = np.asarray(policy.infer(observation), dtype=np.float32)
                elapsed_ms = (time.monotonic() - started) * 1000
                if actions.shape != (32, 16) or not np.isfinite(actions).all():
                    raise ValueError(f"invalid action shape or values: {actions.shape}")
                self._reply(200, {"step": payload["step"], "actions": actions[:3].tolist(), "latency_ms": elapsed_ms})
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
