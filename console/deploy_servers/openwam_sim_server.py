#!/usr/bin/env python3
"""Private causal OpenWAM Alpha / Isaac Sim inference bridge.

Consumes only the current simulator head RGB frame and current dual-tool
state. Per-arm first-frame SE(3) anchors are simulation hypotheses, not a
shared physical VR/camera calibration. Generated future video is never fed back as observed
ground truth. Returns body-relative 30 Hz actions for the bounded sim adapter.
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

from lingbot_sim_server import REFERENCE_LEFT, pose7, qconj, qmul, qnorm, rotate


TASK = "Place the cup on the plate, then put it back to its original position"
REFERENCE_RIGHT = np.array([0.19422552, 0.13807945, 0.22340487,
                            0.31853813, -0.03337900, -0.09657992,
                            0.94238615], dtype=np.float64)  # same episode/frame, p + wxyz
POSE_MAPPING = "per_arm_first_live_tool_to_training_episode_61164_frame0_SE3_provisional"
CALIBRATED_POSE_FRAMES = frozenset(("source_hand_mirrored_replay_prior_v1",
                                   "source_hand_reference_259632_v1"))


def validated_pose_frame(payload, required):
    pose_frame = payload.get("pose_frame")
    if pose_frame is not None and pose_frame not in CALIBRATED_POSE_FRAMES:
        raise ValueError("unsupported calibrated pose frame")
    if set(payload) != required | ({"pose_frame"} if pose_frame is not None else set()):
        raise ValueError("unexpected request fields")
    return pose_frame


def head_image(encoded):
    from PIL import Image, ImageOps

    if not isinstance(encoded, str) or len(encoded) > 3_000_000:
        raise ValueError("invalid head JPEG")
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(BytesIO(raw)) as image:
        if image.size != (640, 480):
            raise ValueError("head image dimensions changed")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if int(rgb.max()) - int(rgb.min()) < 2:
            raise ValueError("head image is blank")
        return ImageOps.pad(image.convert("RGB"), (320, 384),
                            method=Image.Resampling.BILINEAR, color=(0, 0, 0))


def raw_state20(mapped, grippers):
    from scipy.spatial.transform import Rotation

    result = np.empty(20, dtype=np.float32)
    for hand, side in enumerate(("left", "right")):
        pose = mapped[side]
        q = pose[3:]
        matrix = Rotation.from_quat((q[1], q[2], q[3], q[0])).as_matrix()
        offset = hand * 10
        result[offset:offset + 3] = pose[:3]
        result[offset + 3:offset + 6] = matrix[:, 0]
        result[offset + 6:offset + 9] = matrix[:, 1]
        result[offset + 9] = grippers[hand]
    return result


def absolute_to_body_actions(predicted, grippers, current):
    """Convert OpenWAM world absolute pose predictions to causal body deltas."""
    actions = np.empty((3, 16), dtype=np.float64)
    for hand, side in enumerate(("left", "right")):
        previous_p, previous_q = current[side][:3], current[side][3:]
        offset = hand * 7
        for index in range(3):
            pose = predicted[index, hand]
            position = np.asarray(pose[:3], dtype=np.float64)
            quaternion = qnorm([pose[6], *pose[3:6]])  # decoded xyzw -> wxyz
            delta_p = rotate(qconj(previous_q), position - previous_p)
            delta_q = qnorm(qmul(qconj(previous_q), quaternion))
            actions[index, offset:offset + 3] = delta_p
            actions[index, offset + 3:offset + 7] = [*delta_q[1:], delta_q[0]]
            actions[index, 14 + hand] = grippers[index, hand]
            previous_p, previous_q = position, quaternion
    if not np.isfinite(actions).all():
        raise ValueError("OpenWAM converted action is non-finite")
    return actions


class Engine:
    def __init__(self, args):
        checkpoint_dir = args.checkpoint_dir.resolve()
        weight = checkpoint_dir / args.checkpoint_name
        if not weight.is_file() or weight.stat().st_size < 1_000_000_000:
            raise ValueError("OpenWAM checkpoint missing or incomplete")
        prepared = args.prepared.resolve()
        sys.path[:0] = [str(args.openwam_root), str(Path(__file__).resolve().parent)]
        import setuptools  # noqa: F401 -- DeepSpeed/Python distutils shim
        from replay_openwam_cup import install_cup_deploy_normalizer, decode_actions20
        from openwam.deploy.server import build_server_from_config
        from omegaconf import OmegaConf

        install_cup_deploy_normalizer(prepared, checkpoint_dir)
        config = OmegaConf.load(args.openwam_root / "configs/deploy.yaml")
        if int(config.inference.denoise_steps) != 10 or config.inference.denoise_mode != "sync":
            raise ValueError("OpenWAM deploy schedule changed")
        OmegaConf.update(config, "optimization.compile.enabled", False, merge=False)
        self.model = build_server_from_config(config, str(checkpoint_dir),
                                               ckpt_name=args.checkpoint_name,
                                               device="cuda").engine
        self.decode_actions20 = decode_actions20
        self.checkpoint = str(weight)
        self.lock = threading.Lock()
        self.episode = None
        self.next_step = 0
        self.rotations = {}
        self.translations = {}
        self.pose_frame = None

    def infer(self, payload):
        required = {"head_jpeg", "left_pose_wxyz", "right_pose_wxyz",
                    "gripper_rad", "episode", "step"}
        pose_frame = validated_pose_frame(payload, required)
        calibrated = pose_frame is not None
        step, episode = payload["step"], payload["episode"]
        if type(step) is not int or type(episode) is not int or step < 0 or episode < 0:
            raise ValueError("invalid episode or step")
        live = {side: pose7(payload, f"{side}_pose_wxyz")
                for side in ("left", "right")}
        grips = np.asarray(payload["gripper_rad"], dtype=np.float32)
        if grips.shape != (2,) or not np.isfinite(grips).all():
            raise ValueError("invalid gripper state")
        frame = head_image(payload["head_jpeg"])
        with self.lock:
            if step == 0:
                for side, reference in (("left", REFERENCE_LEFT),
                                        ("right", REFERENCE_RIGHT)):
                    rotation = qnorm(qmul(reference[3:], qconj(live[side][3:])))
                    self.rotations[side] = rotation
                    self.translations[side] = reference[:3] - rotate(rotation, live[side][:3])
                    if calibrated:
                        self.rotations[side] = np.array([1., 0., 0., 0.])
                        self.translations[side] = np.zeros(3)
                self.calibrated = calibrated
                self.pose_frame = pose_frame
                self.episode, self.next_step = episode, 0
            if (self.episode != episode or step != self.next_step
                    or self.calibrated != calibrated or self.pose_frame != pose_frame):
                raise ValueError("non-causal or missing simulator step")
            mapped = {
                side: np.concatenate((rotate(self.rotations[side], pose[:3]) + self.translations[side],
                                      qnorm(qmul(self.rotations[side], pose[3:]))))
                for side, pose in live.items()
            }
            state = raw_state20(mapped, grips)
            start = time.monotonic()
            generated = self.model.generate({
                "first_frame_image": [frame], "prompt": TASK, "proprio": state,
                "num_frames": 33, "video_num_frames": 9,
                "height": 384, "width": 320, "seed": 42,
            })
            raw = np.asarray(generated["actions"], dtype=np.float32)
            if raw.ndim != 2 or raw.shape[0] < 3 or raw.shape[1] != 20:
                raise ValueError("OpenWAM generated an invalid action chunk")
            predicted, grippers = self.decode_actions20(raw[:3])
            actions = absolute_to_body_actions(predicted, grippers, mapped)
            self.next_step += 1
            return {"episode": episode, "step": step, "actions": actions.tolist(),
                    "latency_ms": (time.monotonic() - start) * 1000,
                    "pose_mapping": pose_frame if calibrated else POSE_MAPPING,
                    "future_observation_used": False}


def serve(args):
    engine = Engine(args)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self._json(200, {"model": args.model_id, "checkpoint": engine.checkpoint,
                             "status": "ready", "mapping": POSE_MAPPING})

        def do_POST(self):
            if self.path != "/infer":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4_000_000:
                    raise ValueError("request size out of range")
                self._json(200, engine.infer(json.loads(self.rfile.read(length))))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def _json(self, status, value):
            raw = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--openwam-root", type=Path, required=True)
    parser.add_argument("--model-id", default="openwam-cup-clean-10000")
    parser.add_argument("--port", type=int, default=18787)
    args = parser.parse_args()
    if not 1024 <= args.port < 65536:
        parser.error("port out of range")
    serve(args)


if __name__ == "__main__":
    main()
