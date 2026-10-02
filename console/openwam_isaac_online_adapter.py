"""Causal OpenWAM Alpha -> dual-YUBI trajectory adapter (provisional).

Only the current head render and current robot state condition OpenWAM.  Its
absolute UMI poses are converted to body deltas by the private server; this
client applies bounded 30 Hz-to-10 Hz integration. The separate first-frame
arm alignments are simulation hypotheses, not a shared measured extrinsic.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi05_isaac_online_adapter import (  # noqa: E402
    _source_from_fraction, _waypoint, current_robot, encode_wrist, CALIBRATION, POSE_FRAME,
)


URL = os.environ.get("OPENWAM_ONLINE_URL", "http://127.0.0.1:18789/infer")
AUDIT_DIR = Path(os.environ.get("SIM_ADAPTER_AUDIT_DIR", "/tmp/openwam_online_audit"))


def predict(observation, step, episode):
    start = time.monotonic()
    if "images" not in observation or "head" not in observation["images"]:
        raise ValueError("simulator did not provide a live head image")
    current = {side: current_robot(observation, side) for side in ("left", "right")}
    head_jpg = encode_wrist(observation["images"]["head"], "head")
    # Retain the simultaneous wrist renders for audit, but do not feed them to
    # the OpenWAM center-camera model as if they were its training view.
    wrist_jpg = {side: encode_wrist(observation["images"][f"{side}_wrist"], side)
                 for side in ("left", "right")}
    payload = {
        "head_jpeg": base64.b64encode(head_jpg).decode("ascii"),
        "left_pose_wxyz": [*current["left"][0].tolist(), *current["left"][1].tolist()],
        "right_pose_wxyz": [*current["right"][0].tolist(), *current["right"][1].tolist()],
        "gripper_rad": [_source_from_fraction(current[side][2]) for side in ("left", "right")],
        "episode": int(episode), "step": int(step),
    }
    if CALIBRATION:
        payload["pose_frame"] = POSE_FRAME
    request = Request(URL, data=json.dumps(payload, separators=(",", ":")).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:
        prediction = json.load(response)
    if prediction.get("episode") != episode or prediction.get("step") != step:
        raise ValueError("OpenWAM response episode/step mismatch")
    if prediction.get("future_observation_used") is not False:
        raise ValueError("OpenWAM inference causality is unverified")
    expected_mapping = POSE_FRAME if CALIBRATION else "per_arm_first_live_tool_to_training_episode_61164_frame0_SE3_provisional"
    if prediction.get("pose_mapping") != expected_mapping:
        raise ValueError("OpenWAM pose mapping changed")
    actions = np.asarray(prediction["actions"], dtype=np.float64)
    if actions.shape != (3, 16) or not np.isfinite(actions).all():
        raise ValueError("OpenWAM action must be 3x16 finite")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve bounded diagnostics even when the actuator safety gate rejects
    # a model prediction. This does not execute or clip the rejected action.
    (AUDIT_DIR / f"prediction_{step:04d}.json").write_text(json.dumps({
        "episode": int(episode), "step": int(step),
        "actions": actions.tolist(),
        "current_model_input_pose": {side: {
            "position_m": current[side][0].tolist(),
            "quaternion_wxyz": current[side][1].tolist(),
        } for side in ("left", "right")},
        "source_gripper_rad": payload["gripper_rad"],
        "pose_mapping": prediction["pose_mapping"],
        "calibration": CALIBRATION.audit() if CALIBRATION else None,
    }, separators=(",", ":")))
    use_30hz = os.environ.get("UMI_EXECUTE_30HZ") == "1"
    if use_30hz:
        from pi05_isaac_online_adapter import _waypoints_30hz
        waypoints, diagnostics = _waypoints_30hz(current, actions)
        waypoint = waypoints[-1]
    else:
        waypoint, diagnostics = _waypoint(current, actions)
    if step < 3:
        (AUDIT_DIR / f"input_head_{step:04d}.jpg").write_bytes(head_jpg)
        for side in ("left", "right"):
            (AUDIT_DIR / f"input_{side}_{step:04d}.jpg").write_bytes(wrist_jpg[side])
    audit = {
        "episode": int(episode), "step": int(step), "model": "openwam_alpha_clean_cup",
        "observation_origin": "current_simulator_render_and_robot_state",
        "image_metadata": observation.get("image_metadata", {}),
        "model_input_views": ["head"],
        "image_shapes": {side: [480, 640, 3] for side in ("left", "right")},
        "head_image_shape": [480, 640, 3],
        "head_image_sha256": hashlib.sha256(head_jpg).hexdigest(),
        "wrist_image_sha256": {side: hashlib.sha256(wrist_jpg[side]).hexdigest()
                               for side in ("left", "right")},
        "model_input_pose": {side: {"position_m": current[side][0].tolist(),
                                   "quaternion_wxyz": current[side][1].tolist()}
                            for side in ("left", "right")},
        "source_gripper_rad": payload["gripper_rad"],
        "source_action_gripper_rad": [float(actions[-1, 14]), float(actions[-1, 15])],
        "waypoint": waypoint, "mapping": diagnostics,
        "execution_hz": 30 if use_30hz else 10,
        "substep_waypoints": waypoints if use_30hz else None,
        "model_latency_ms": prediction["latency_ms"],
        "total_adapter_latency_ms": (time.monotonic()-start)*1000,
        "endpoint_assumption": prediction["pose_mapping"],
        "calibration": CALIBRATION.audit() if CALIBRATION else None,
        "world_tool_pose": {s: observation["robots"][s]["tool_pose"] for s in ("left", "right")},
        "future_observation_used": False,
        "gripper_calibration": {"source_closed_rad": -0.45, "source_open_rad": 0.78,
                                "sim_closed_rad": 0.0, "sim_open_rad": 0.6},
    }
    if CALIBRATION:
        audit["gripper_calibration"] = {**CALIBRATION.gripper, "sim_closed_rad": 0., "sim_open_rad": .6}
    with (AUDIT_DIR / "online_adapter.jsonl").open("a") as output:
        output.write(json.dumps(audit, separators=(",", ":")) + "\n")
    if use_30hz:
        return {"action_dt_s": 1 / 30, "waypoints": waypoints, "execute_steps": 3}
    return {"action_dt_s": 0.1, "waypoints": [waypoint], "execute_steps": 1}
