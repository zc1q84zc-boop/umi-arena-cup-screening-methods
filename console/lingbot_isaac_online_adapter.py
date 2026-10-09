"""Causal LingBot VLA2 -> dual-YUBI trajectory adapter (provisional).

The private server anchors the first live tool pose into a clean-cup UMI
reference frame.  That alignment is a simulation hypothesis, not measured
camera/VR calibration.  Only current rendered images and robot state are sent.
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

# Isaac loads policy scripts via importlib, which does not add their parent
# directory to sys.path (unlike running a Python file directly).
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from deploy_servers.official_cup_prompts import cup_instruction, PROMPT_SOURCE, PROMPT_PROTOCOL
except ImportError:  # the same file is deployed beside this adapter
    from official_cup_prompts import cup_instruction, PROMPT_SOURCE, PROMPT_PROTOCOL

from pi05_isaac_online_adapter import (  # noqa: E402
    _source_from_fraction, _waypoint, conj, current_robot, encode_wrist,
    mul, normalize, rotate, CALIBRATION, POSE_FRAME,
)


URL = os.environ.get("LINGBOT_ONLINE_URL", "http://127.0.0.1:18786/infer")
AUDIT_DIR = Path(os.environ.get("SIM_ADAPTER_AUDIT_DIR", "/tmp/lingbot_online_audit"))


def predict(observation, step, episode):
    start = time.monotonic()
    if "images" not in observation:
        raise ValueError("simulator did not provide live wrist images")
    # Only the evaluator's discrete instruction changes. Cup coordinates,
    # contact forces and demonstration actions are not model inputs.
    task_stage = observation["task_stage"]
    prompt = cup_instruction(task_stage)
    current = {side: current_robot(observation, side) for side in ("left", "right")}
    left_p, left_q, left_grip = current["left"]
    right_p, right_q, right_grip = current["right"]
    relative_p = rotate(conj(left_q), right_p-left_p)
    relative_q = normalize(mul(conj(left_q), right_q))
    relative_pose = [*relative_p.tolist(), *relative_q[1:].tolist(), float(relative_q[0])]
    jpg = {side: encode_wrist(observation["images"][f"{side}_wrist"], side)
           for side in ("left", "right")}
    payload = {
        "left_jpeg": base64.b64encode(jpg["left"]).decode("ascii"),
        "right_jpeg": base64.b64encode(jpg["right"]).decode("ascii"),
        "left_pose_wxyz": [*left_p.tolist(), *left_q.tolist()],
        "right_pose_wxyz": [*right_p.tolist(), *right_q.tolist()],
        "relative_pose_xyzw": relative_pose,
        "gripper_rad": [_source_from_fraction(left_grip), _source_from_fraction(right_grip)],
        "episode": int(episode), "step": int(step),
        "prompt": prompt,
    }
    if CALIBRATION:
        payload["pose_frame"] = POSE_FRAME
    request = Request(URL, data=json.dumps(payload, separators=(",", ":")).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=120) as response:
        prediction = json.load(response)
    expected_model = os.environ.get('LINGBOT_EXPECTED_MODEL_ID')
    if expected_model:
        provenance = prediction.get('model_provenance', {})
        if (prediction.get('model_id') != expected_model
                or provenance.get('fine_tuned') is not False
                or provenance.get('all_files_sha256_verified') is not True):
            raise ValueError('Official model adapter rejected unverified or fine-tuned weights')
    if prediction.get("episode") != episode or prediction.get("step") != step:
        raise ValueError("LingBot response episode/step mismatch")
    if (prediction.get("prompt") != prompt or prediction.get("prompt_source") != PROMPT_SOURCE
            or prediction.get("prompt_protocol") != PROMPT_PROTOCOL
            or prediction.get("inference_mode") != "native_lingbot_vla_v2_chunk"):
        raise ValueError("LingBot did not confirm the official primitive prompt/native inference")
    expected_mapping = POSE_FRAME if CALIBRATION else "first_live_left_tool_to_training_episode_61164_frame0_SE3_provisional"
    if prediction.get("pose_mapping") != expected_mapping:
        raise ValueError("LingBot response pose mapping changed")
    expected_timing = {"pose_rows": [1, 2, 3], "gripper_rows": [0, 1, 2]}
    if prediction.get("action_timing") != expected_timing:
        raise ValueError("LingBot response does not confirm causal pose/gripper alignment")
    actions = np.asarray(prediction["actions"], dtype=np.float64)
    if actions.shape != (3, 16) or not np.isfinite(actions).all():
        raise ValueError("LingBot action must be 3x16 finite")
    use_30hz = os.environ.get("UMI_EXECUTE_30HZ") == "1"
    if use_30hz:
        from pi05_isaac_online_adapter import _waypoints_30hz
        waypoints, diagnostic = _waypoints_30hz(current, actions)
        waypoint = waypoints[-1]
    else:
        waypoint, diagnostic = _waypoint(current, actions)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if step < 3:
        for side in ("left", "right"):
            (AUDIT_DIR / f"input_{side}_{step:04d}.jpg").write_bytes(jpg[side])
    audit = {
        "episode": int(episode), "step": int(step),
        "model": prediction.get('model_id', 'lingbot_vla2_clean_cup'),
        "model_provenance": prediction.get('model_provenance'),
        "task_stage": task_stage, "prompt": prompt, "server_prompt": prediction["prompt"],
        "prompt_source": PROMPT_SOURCE, "prompt_protocol": PROMPT_PROTOCOL,
        "inference_mode": prediction["inference_mode"],
        "observation_origin": "current_simulator_render_and_robot_state",
        "image_metadata": observation.get("image_metadata", {}),
        "image_shapes": {side: [480, 640, 3] for side in ("left", "right")},
        "image_sha256": {side: hashlib.sha256(jpg[side]).hexdigest() for side in ("left", "right")},
        "relative_pose_xyzw": relative_pose,
        "source_gripper_rad": payload["gripper_rad"],
        "model_input_pose": {side: {"position_m": current[side][0].tolist(),
                                   "quaternion_wxyz": current[side][1].tolist()}
                            for side in ("left", "right")},
        "source_action_gripper_rad": [float(actions[-1, 14]), float(actions[-1, 15])],
        "waypoint": waypoint, "mapping": diagnostic,
        "execution_hz": 30 if use_30hz else 10,
        "substep_waypoints": waypoints if use_30hz else None,
        "model_latency_ms": prediction["latency_ms"],
        "total_adapter_latency_ms": (time.monotonic()-start)*1000,
        "endpoint_assumption": prediction["pose_mapping"],
        "calibration": CALIBRATION.audit() if CALIBRATION else None,
        'future_observation_used': False,
        'action_timing': prediction.get('action_timing'),
        "world_tool_pose": {s: observation["robots"][s]["tool_pose"] for s in ("left", "right")},
        "gripper_calibration": {"source_closed_rad": -0.45, "source_open_rad": 0.78,
                                "sim_closed_rad": 0.0, "sim_open_rad": 0.6},
    }
    if CALIBRATION:
        from pi05_isaac_online_adapter import SIM_GRIPPER_CLOSED_RAD, SIM_GRIPPER_OPEN_RAD
        audit["gripper_calibration"] = {**CALIBRATION.gripper, "sim_closed_rad": SIM_GRIPPER_CLOSED_RAD,
                                      "sim_open_rad": SIM_GRIPPER_OPEN_RAD}
    with (AUDIT_DIR / "online_adapter.jsonl").open("a") as output:
        output.write(json.dumps(audit, separators=(",", ":")) + "\n")
    if use_30hz:
        return {"action_dt_s": 1 / 30, "waypoints": waypoints, "execute_steps": 3}
    return {"action_dt_s": 0.1, "waypoints": [waypoint], "execute_steps": 1}
