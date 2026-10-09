#!/usr/bin/env python3
"""Private causal LingBot/Isaac bridge for one clean-cup checkpoint.

The reference pose anchors the simulator's *first observed* left tool pose to
one clean training pose.  This is a simulation-only alignment hypothesis, not
a measured VR-to-robot calibration.  No future or recorded image is supplied.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
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

try:
    from .official_cup_prompts import (RIGHT_PLACE, PROMPT_SOURCE, PROMPT_PROTOCOL,
                                       validate_instruction)
except ImportError:  # deployed together in the private scripts directory
    from official_cup_prompts import (RIGHT_PLACE, PROMPT_SOURCE, PROMPT_PROTOCOL,
                                      validate_instruction)

# Compatibility name for training-label tests; never a fixed inference prompt.
TASK = RIGHT_PLACE
INTER_HAND = "observation.pose.left_hand_root_to_right_hand_root.absolute"
POSE_KEYS = ("observation.pose.left_hand_root.relative",
             "observation.pose.right_hand_root.relative")


def align_causal_action_chunk(parts, gripper):
    """Match pose deltas at t+1..t+3 to next-frame gripper rows t..t+2.

    The UMI training config reads ``observation.pose.*.relative`` as its
    action pose. Row zero is the already observed t-1→t delta, while
    ``action.joint_states`` row zero targets frame t+1.
    """
    parts = [np.asarray(part, dtype=np.float64) for part in parts]
    gripper = np.asarray(gripper, dtype=np.float64)
    if (len(parts) != 2 or any(part.ndim != 2 or part.shape[0] < 4 or part.shape[1] != 7
                            for part in parts)
            or gripper.ndim != 2 or gripper.shape[0] < 3 or gripper.shape[1] != 2):
        raise ValueError("LingBot returned a short or malformed action chunk")
    actions = np.concatenate((parts[0][1:4], parts[1][1:4], gripper[:3]), axis=1)
    if not np.isfinite(actions).all():
        raise ValueError("LingBot returned non-finite causal actions")
    return actions
CALIBRATED_POSE_FRAMES = frozenset(("source_hand_mirrored_replay_prior_v1",
                                   "source_hand_reference_259632_v1"))


def validated_pose_frame(payload, required):
    pose_frame = payload.get("pose_frame")
    if pose_frame is not None and pose_frame not in CALIBRATED_POSE_FRAMES:
        raise ValueError("unsupported calibrated pose frame")
    if set(payload) != required | ({"pose_frame"} if pose_frame is not None else set()):
        raise ValueError("unexpected request fields")
    return pose_frame
# Frame 0 of audited successful training episode 61164.  Used ONLY as a
# provisional pose-frame anchor, never as an image or future observation.
REFERENCE_LEFT = np.array([-0.2967015727, 0.1179096300, 0.3183151206,
                           0.8879379794, -0.0200879912, -0.2024016271,
                           0.4125484197], dtype=np.float64)  # p + wxyz


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw))


def qnorm(q):
    q = np.asarray(q, dtype=np.float64)
    length = float(np.linalg.norm(q))
    if q.shape != (4,) or not math.isfinite(length) or length < 1e-8:
        raise ValueError("invalid quaternion")
    return q / length


def qconj(q):
    return np.array((q[0], -q[1], -q[2], -q[3]))


def rotate(q, point):
    return qmul(qmul(q, np.array((0.0, *point))), qconj(q))[1:]


def pose7(payload, key):
    pose = np.asarray(payload[key], dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{key} must be finite p3+wxyz4")
    pose[3:] = qnorm(pose[3:])
    return pose


def xyzw_pose(position, quaternion):
    return np.asarray([*position, *quaternion[1:], quaternion[0]], dtype=np.float32)


def image(payload, side):
    from PIL import Image
    encoded = payload[f"{side}_jpeg"]
    if not isinstance(encoded, str) or len(encoded) > 3_000_000:
        raise ValueError(f"invalid {side} JPEG")
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(BytesIO(raw)) as img:
        if img.size != (640, 480):
            raise ValueError(f"{side} image dimensions changed")
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    if int(rgb.max()) - int(rgb.min()) < 2:
        raise ValueError(f"{side} image is blank")
    return rgb


class Engine:
    def __init__(self, args):
        checkpoint = args.checkpoint.resolve()
        if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors.index.json").is_file():
            raise ValueError("incomplete LingBot HF checkpoint")
        index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
        shards = set(index['weight_map'].values())
        if not shards or any(Path(name).name != name or not (checkpoint / name).is_file()
                             for name in shards):
            raise ValueError("LingBot checkpoint has missing or invalid model shards")
        self.provenance = {'weights_origin': 'cup_fine_tuned', 'fine_tuned': True}
        if args.official_manifest:
            from official_lingbot_weights import verify_official_checkpoint
            self.provenance = verify_official_checkpoint(checkpoint, args.official_manifest)
        config_root = args.config_root.resolve()
        if not (config_root / "configs/robot_configs/umi_cup_clean.yaml").is_file():
            raise ValueError("clean-cup robot config missing")
        # Checkpoint data_config retains the original host's absolute path.
        # Relocate the same verified stats file, never rewrite checkpoint config.
        normalization_path = config_root / 'lingbot/norm_stats.json'
        if not normalization_path.is_file():
            raise ValueError('relocated clean-cup normalization file missing')
        self.provenance['normalization_path'] = str(normalization_path)
        self.provenance['normalization_sha256'] = hashlib.sha256(normalization_path.read_bytes()).hexdigest()
        sys.path[:0] = [str(args.lingbot_root), str(args.lingbot_root / "deploy")]
        os.chdir(config_root)
        from lingbot_vla_v2_policy import LingbotVLAv2Server
        server_class = LingbotVLAv2Server
        if args.official_manifest:
            from official_lingbot_weights import official_server_class
            server_class = official_server_class(server_class)
        self.model = server_class(path_to_pi_model=str(checkpoint),
                                        robot_norm_path=str(normalization_path), chunk_ret=True,
                                        use_length=32, use_compile=False)
        self.model.infer({"reset": True, "robo_name": "umi_cup_clean"})
        if args.official_manifest:
            self.provenance['robot_config_sha256'] = hashlib.sha256(
                (config_root / 'configs/robot_configs/umi_cup_clean.yaml').read_bytes()).hexdigest()
            self.provenance['normalization_sha256'] = hashlib.sha256(
                Path(self.model.robot_norm_path).read_bytes()).hexdigest()
        self.checkpoint = str(checkpoint)
        self.model_id = args.model_id
        self.lock = threading.Lock()
        self.episode = None
        self.next_step = 0
        self.rotation = None
        self.translation = None
        self.pose_frame = None

    def infer(self, payload):
        required = {"left_jpeg", "right_jpeg", "left_pose_wxyz", "right_pose_wxyz",
                    "relative_pose_xyzw", "gripper_rad", "episode", "step", "prompt"}
        pose_frame = validated_pose_frame(payload, required)
        prompt = validate_instruction(payload["prompt"])
        calibrated = pose_frame is not None
        step, episode = payload["step"], payload["episode"]
        if type(step) is not int or type(episode) is not int or step < 0 or episode < 0:
            raise ValueError("invalid episode or step")
        left, right = (pose7(payload, f"{side}_pose_wxyz") for side in ("left", "right"))
        relative = np.asarray(payload["relative_pose_xyzw"], dtype=np.float32)
        grips = np.asarray(payload["gripper_rad"], dtype=np.float32)
        if relative.shape != (7,) or grips.shape != (2,) or not np.isfinite(relative).all() or not np.isfinite(grips).all():
            raise ValueError("invalid relative pose or gripper")
        left_img, right_img = image(payload, "left"), image(payload, "right")
        with self.lock:
            if step == 0:
                self.model.infer({"reset": True, "robo_name": "umi_cup_clean"})
                self.rotation = qnorm(qmul(REFERENCE_LEFT[3:], qconj(left[3:])))
                self.translation = REFERENCE_LEFT[:3] - rotate(self.rotation, left[:3])
                if calibrated:
                    self.rotation = np.array([1., 0., 0., 0.])
                    self.translation = np.zeros(3)
                self.calibrated = calibrated
                self.pose_frame = pose_frame
                self.episode, self.next_step = episode, 0
            if (self.episode != episode or step != self.next_step
                    or self.calibrated != calibrated or self.pose_frame != pose_frame):
                raise ValueError("non-causal or missing simulator step")
            mapped = {}
            for side, pose in (("left", left), ("right", right)):
                mapped[side] = xyzw_pose(rotate(self.rotation, pose[:3]) + self.translation,
                                         qnorm(qmul(self.rotation, pose[3:])))
            observation = {
                "observation.image.left": left_img,
                "observation.image.right": right_img,
                INTER_HAND: relative,
                "observation.joint_states": grips,
                "observation.pose.left_hand_root.absolute": mapped["left"],
                "observation.pose.right_hand_root.absolute": mapped["right"],
                "prompt": prompt, "task": prompt,
            }
            start = time.monotonic()
            prediction = self.model.infer(observation)
            parts = [np.asarray(prediction[key], dtype=np.float64) for key in POSE_KEYS]
            gripper = np.asarray(prediction["action.joint_states"], dtype=np.float64)
            actions = align_causal_action_chunk(parts, gripper)
            self.next_step += 1
            return {"episode": episode, "step": step, "actions": actions.tolist(),
                    "model_id": self.model_id, "model_provenance": self.provenance,
                    "prompt": prompt, "prompt_source": PROMPT_SOURCE,
                    "prompt_protocol": PROMPT_PROTOCOL,
                    "inference_mode": "native_lingbot_vla_v2_chunk",
                    "latency_ms": (time.monotonic() - start) * 1000,
                    "pose_mapping": pose_frame if calibrated else "first_live_left_tool_to_training_episode_61164_frame0_SE3_provisional",
                    "action_timing": {"pose_rows": [1, 2, 3], "gripper_rows": [0, 1, 2]}}


def serve(args):
    engine = Engine(args)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self._json(200, {"model": args.model_id, "checkpoint": engine.checkpoint,
                             "model_provenance": engine.provenance,
                             "status": "ready", "mapping": "provisional_SE3",
                             "prompt_protocol": PROMPT_PROTOCOL})

        def do_POST(self):
            if self.path != "/infer":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 7_000_000:
                    raise ValueError("request size out of range")
                value = engine.infer(json.loads(self.rfile.read(length)))
                self._json(200, value)
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--model-id", default="lingbot-cup-clean-5000")
    parser.add_argument("--official-manifest", type=Path)
    parser.add_argument("--port", type=int, default=18784)
    args = parser.parse_args()
    if not 1024 <= args.port < 65536:
        parser.error("port out of range")
    serve(args)


if __name__ == "__main__":
    main()
