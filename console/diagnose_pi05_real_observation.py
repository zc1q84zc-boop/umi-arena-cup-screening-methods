#!/usr/bin/env python3
"""Read-only real-observation probe for the private squirrel π0.5 service.

Uses matched current left/right wrist video frames and robot state from one
recorded episode. It does not replay actions or write to the source dataset.
Run only after checking the dedicated model service and GPU ownership.
"""

import base64
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EPISODE = json.loads((ROOT / "data/episode-259632.json").read_text())
FRAMES = (0, 30, 60)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def quat_rotate(q, v):
    return quat_mul(quat_mul(q, (0., *v)), quat_conj(q))[1:]


def pose_quat(pose):
    q = pose[3:]
    return (q[3], *q[:3])


def current_image(side, frame, *, simulated=False):
    video = (ROOT / f"sim_runs/06d9b4cbc1eb/video_{side}_wrist.mp4" if simulated
             else ROOT / f"videos/episode-259632-{side}.mp4")
    cmd = ("ffmpeg", "-hide_banner", "-loglevel", "error", "-i",
           str(video),
           "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=True)
    if not result.stdout.startswith(b"\xff\xd8"):
        raise RuntimeError(f"missing {side} frame {frame}")
    return base64.b64encode(result.stdout).decode("ascii")


def infer(frame, image_source="recorded"):
    left, right = EPISODE["poses"][frame]
    lq, rq = pose_quat(left), pose_quat(right)
    rel_p = quat_rotate(quat_conj(lq), [right[i] - left[i] for i in range(3)])
    rel_q = quat_mul(quat_conj(lq), rq)
    selections = {
        "recorded": {"left": "recorded", "right": "recorded"},
        "simulated": {"left": "simulated", "right": "simulated"},
        "replay_aligned_sim": {"left": "replay", "right": "replay"},
        "replay_aligned_sim_light500": {"left": "light500", "right": "light500"},
        "replay_aligned_sim_emissive": {"left": "emissive", "right": "emissive"},
        "replay_aligned_sim_darkjaws": {"left": "darkjaws", "right": "darkjaws"},
        "real_left_sim_right": {"left": "recorded", "right": "simulated"},
        "sim_left_real_right": {"left": "simulated", "right": "recorded"},
    }
    if image_source not in selections:
        raise ValueError("unsupported diagnostic image source")
    images = {}
    for side, origin in selections[image_source].items():
        if origin == "recorded":
            images[side] = current_image(side, frame)
        elif origin == "replay":
            path = ROOT / f"sim_validation/reference259632_camera_motion_a/{side}_wrist_source_{frame:03d}.png"
            images[side] = base64.b64encode(path.read_bytes()).decode("ascii")
        elif origin == "light500":
            if frame != 0:
                raise ValueError("the bounded light-500 probe captured frame 0 only")
            path = ROOT / f"sim_validation/reference259632_light500_probe_20260929/{side}_wrist_source_000.png"
            images[side] = base64.b64encode(path.read_bytes()).decode("ascii")
        elif origin == "emissive":
            if frame != 0:
                raise ValueError("the bounded emissive probe captured frame 0 only")
            path = ROOT / f"sim_validation/reference259632_light500_emission020030036_probe_20260929/{side}_wrist_source_000.png"
            images[side] = base64.b64encode(path.read_bytes()).decode("ascii")
        elif origin == "darkjaws":
            if frame != 0:
                raise ValueError("the bounded dark-jaws probe captured frame 0 only")
            path = ROOT / f"sim_validation/reference259632_light500_darkjaws_probe_20260929/{side}_wrist_source_000.png"
            images[side] = base64.b64encode(path.read_bytes()).decode("ascii")
        else:
            if frame == 0:
                path = ROOT / f"sim_runs/06d9b4cbc1eb/input_{side}_0000.jpg"
                images[side] = base64.b64encode(path.read_bytes()).decode("ascii")
            else:
                images[side] = current_image(side, frame, simulated=True)
    payload = {
        "left_jpeg": images["left"],
        "right_jpeg": images["right"],
        "relative_pose_xyzw": [*rel_p, *rel_q[1:], rel_q[0]],
        "gripper_rad": EPISODE["grippers"][frame],
        "step": frame,
    }
    cmd = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
           "squirrel_5090", "curl -fsS --max-time 120 -H 'Content-Type: application/json' "
           "--data-binary @- http://127.0.0.1:18783/infer")
    response = subprocess.run(cmd, input=json.dumps(payload).encode(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=160, check=True)
    result = json.loads(response.stdout)
    if result["step"] != frame or len(result["actions"]) != 3:
        raise RuntimeError("unexpected model response")
    return result, payload


def summarize(frame, result, payload, image_source):
    current = EPISODE["poses"][frame][1]
    gt = EPISODE["poses"][frame + 3][1]
    q = pose_quat(current)
    predicted = list(current[:3])
    raw = []
    for action in result["actions"]:
        dp = action[7:10]
        raw.append([round(x, 5) for x in dp])
        world_dp = quat_rotate(q, dp)
        predicted = [predicted[i] + world_dp[i] for i in range(3)]
        dq = action[10:14]
        dq_wxyz = (dq[3], *dq[:3])
        norm = math.sqrt(sum(x*x for x in dq_wxyz))
        q = quat_mul(q, tuple(x/norm for x in dq_wxyz))
    return {
        "frame": frame,
        "image_source": image_source,
        "source_right_hand_xyz": [round(x, 5) for x in current[:3]],
        "source_gripper_rad": round(payload["gripper_rad"][1], 5),
        "relative_state_xyzw": [round(x, 5) for x in payload["relative_pose_xyzw"]],
        "predicted_right_local_deltas_xyz": raw,
        "predicted_right_source_world_delta_xyz": [round(predicted[i]-current[i], 5) for i in range(3)],
        "recorded_right_source_world_delta_xyz_t_to_t3": [round(gt[i]-current[i], 5) for i in range(3)],
        "predicted_gripper_rad": [round(action[15], 5) for action in result["actions"]],
        "recorded_gripper_rad_t3": round(EPISODE["grippers"][frame+3][1], 5),
        "latency_ms": round(result["latency_ms"], 2),
    }


if __name__ == "__main__":
    print(json.dumps({"episode": EPISODE["episode_index"],
                      "checkpoint": "pi05-cup-clean-30000",
                      "note": "diagnostic only; reference episode may overlap training"},
                     separators=(",", ":")), flush=True)
    for frame in FRAMES:
        for source in ("recorded", "replay_aligned_sim", "simulated",
                       "real_left_sim_right", "sim_left_real_right"):
            response, request = infer(frame, source)
            print(json.dumps(summarize(frame, response, request, source), separators=(",", ":")), flush=True)
