"""Causal π0.5 -> dual-Franka YUBI Isaac Sim trajectory adapter.

Called by the simulator at 10 Hz. The inference request contains only images
rendered at the CURRENT step and the CURRENT robot state. A shared-world-frame
rigid transform cancels in the inter-hand state and body-relative actions.
The enabled mirrored profile transforms current tool observations to source
hand poses and predicted hand poses back to tools. It is a simulation prior,
not measured real-robot calibration.
"""

from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time
import sys
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_calibration import MirroredReplayPrior, Reference259632, TunedOnlineV1

_PROFILE = os.environ.get("UMI_ONLINE_CALIBRATION", "")
if _PROFILE not in ("", "mirrored_replay_prior_20260929", "reference_259632_v1", "tuned_online_v1"):
    raise ValueError("unknown online calibration profile; refusing silent fallback")
CALIBRATION = (TunedOnlineV1() if _PROFILE == 'tuned_online_v1'
               else Reference259632() if _PROFILE == 'reference_259632_v1'
               else MirroredReplayPrior() if _PROFILE else None)
POSE_FRAME = "source_hand_reference_259632_v1" if _PROFILE in ('reference_259632_v1', 'tuned_online_v1') else "source_hand_mirrored_replay_prior_v1"


SOURCE_GRIPPER_CLOSED_RAD = -0.45  # approx clean-cup training q01
SOURCE_GRIPPER_OPEN_RAD = 0.78     # approx clean-cup training q99
SIM_GRIPPER_CLOSED_RAD = -0.1 if _PROFILE == 'tuned_online_v1' else 0.0
SIM_GRIPPER_OPEN_RAD = 0.7 if _PROFILE == 'tuned_online_v1' else 0.6
ENDPOINT_ASSUMPTION = POSE_FRAME if CALIBRATION else "hand_root_equals_sim_yubi_tool_provisional"
URL = os.environ.get("PI05_ONLINE_URL", "http://127.0.0.1:18782/infer")
AUDIT_DIR = Path(os.environ.get("SIM_ADAPTER_AUDIT_DIR", "/tmp/pi05_online_audit"))


class PregraspApproach:
    """Add a bounded pre-close offset to the model's own grasp waypoint.

    The model controls the original arm path until its right gripper requests
    closure near the cup. That model waypoint is then frozen as the anchor;
    the extra target is toward the cup FROM THE MODEL WAYPOINT, not a fixed
    amount of total robot travel. The default diagnostic is +30 mm; a separate
    opt-in diagnostic may add another +30 mm before jaw closure. Measured tool
    feedback is required before release. This is simulation-only assistance.
    """

    def __init__(self):
        self.episode = None
        self.phase = "waiting"
        self.progress_m = 0.0
        self.commanded_progress_m = 0.0
        self.direction_xy = None
        self.model_anchor_xy = None
        self.extra_target_xy = None
        self.hold_target_xy = None
        self.hold_target_z = None
        self.extra_reached = False
        self.approach_quat = None
        self.level_before_approach = False
        self.early_lateral_origin_xy = None
        self.early_lateral_axis_xy = None
        self.level_progress_deg = 0.0
        self.level_origin_quat = None
        self.level_target_quat = None
        self.level_target_pos = None
        self.level_finger_local = None
        self.initial_cup_xy = None
        self.lateral_progress_m = 0.0
        self.lateral_direction_xy = None
        self.lateral_target_m = 0.0
        self.dual_contact_steps = 0
        self.preload_progress = 0.0

    def adjust(self, observation, step, episode, waypoints):
        base_amount = float(os.environ.get("UMI_PREGRASP_APPROACH_M", "0"))
        additional_amount = float(os.environ.get("UMI_PREGRASP_ADDITIONAL_M", "0"))
        amount = base_amount + additional_amount
        if amount == 0:
            return waypoints, None
        if (not 0 < base_amount <= 0.0300001 or not 0 <= additional_amount <= 0.0300001
                or amount > 0.0600001 or len(waypoints) != 3):
            raise ValueError("pregrasp requires 0–30 mm base plus 0–30 mm explicit addition and 30 Hz waypoints")
        lateral_amount = float(os.environ.get("UMI_PREGRASP_LATERAL_CENTER_M", "0"))
        if not 0 <= lateral_amount <= 0.0400001:
            raise ValueError("pregrasp lateral centering is limited to 40 mm")
        level_limit_deg = float(os.environ.get("UMI_PREGRASP_WRIST_LEVEL_MAX_DEG", "0"))
        if not 0 <= level_limit_deg <= 15.0001:
            raise ValueError("pregrasp wrist leveling is limited to 15 degrees")
        preload_fraction = float(os.environ.get("UMI_PREGRASP_CONTACT_PRELOAD_FRACTION", "0"))
        if not 0 <= preload_fraction <= 0.150001:
            raise ValueError("pregrasp contact preload is limited to 0.15 gripper fraction")
        hold_trigger_orientation = os.environ.get("UMI_PREGRASP_HOLD_TRIGGER_ORIENTATION", "0")
        if hold_trigger_orientation not in ("0", "1"):
            raise ValueError("pregrasp trigger orientation hold must be 0 or 1")
        early_lateral = os.environ.get("UMI_PREGRASP_EARLY_LATERAL_CENTER", "0")
        if early_lateral not in ("0", "1"):
            raise ValueError("pregrasp early lateral centering must be 0 or 1")
        if early_lateral == "1" and not lateral_amount:
            raise ValueError("early lateral centering requires a bounded lateral allowance")
        balance_pad_gaps = os.environ.get("UMI_PREGRASP_BALANCE_PAD_GAPS", "0")
        if balance_pad_gaps not in ("0", "1") or (balance_pad_gaps == "1" and early_lateral != "1"):
            raise ValueError("pad-gap balancing requires early lateral centering")
        early_level = os.environ.get("UMI_PREGRASP_EARLY_LEVEL", "0")
        if early_level not in ("0", "1") or (early_level == "1" and
                (not level_limit_deg or hold_trigger_orientation != "1")):
            raise ValueError("early leveling requires bounded leveling and held wrist orientation")
        if step == 0 or self.episode != episode:
            self.__init__()
            self.episode = episode
        cup = np.asarray(observation["objects"]["cup"]["position_m"], dtype=float)
        tool = np.asarray(observation["robots"]["right"]["tool_pose"]["position_m"], dtype=float)
        forces = observation["robots"]["right"].get("cup_contact_force_N")
        if cup.shape != (3,) or tool.shape != (3,) or not np.isfinite(cup).all() or not np.isfinite(tool).all():
            raise ValueError("invalid observed cup or tool position for pregrasp approach")
        if not isinstance(forces, dict) or any(name not in forces for name in ("left_finger", "right_finger")):
            raise ValueError("pregrasp approach requires measured cup contact forces")
        contact_n = max(float(np.linalg.norm(forces[name])) for name in ("left_finger", "right_finger"))
        if not math.isfinite(contact_n):
            raise ValueError("invalid cup contact force")
        if self.initial_cup_xy is None:
            self.initial_cup_xy = cup[:2].copy()
        cup_shift_m = float(np.linalg.norm(cup[:2] - self.initial_cup_xy))
        # Wrist leveling produced a measured 2.9 mm settling/contact shift
        # at <=0.16 N before the centering phase. Allow 4 mm, but still abort
        # on stronger contact or any further push before jaw closure.
        preclose_shift_limit_m = 0.004
        horizontal = cup[:2] - tool[:2]
        horizontal_m = float(np.linalg.norm(horizontal))
        tool_cup_m = float(np.linalg.norm(cup - tool))
        link_poses = observation["robots"]["right"].get("link_poses", {})
        if not all(name in link_poses for name in _JAW_PIVOTS):
            raise ValueError("pregrasp approach requires both observed finger-link poses")
        pads = {name: _pad_clearance(name, link_poses[name], cup) for name in _JAW_PIVOTS}
        pad_midpoint_xy = None
        pad_center_error_m = None
        if lateral_amount:
            vertices_xy = [np.asarray(pads[name]["nearest_vertex_world_m"][:2], dtype=float)
                           for name in _JAW_PIVOTS]
            pad_midpoint_xy = (vertices_xy[0] + vertices_xy[1]) / 2
            pad_center_error_m = float(np.linalg.norm(cup[:2] - pad_midpoint_xy))
        pad_separation_deg = abs((pads["left_finger"]["azimuth_deg"]
                                  - pads["right_finger"]["azimuth_deg"] + 180) % 360 - 180)
        if self.phase in ("waiting", "approaching", "leveling", "waiting_for_grasp_corridor") and (contact_n > 0.25 or cup_shift_m > preclose_shift_limit_m):
            raise ValueError("pregrasp stopped before closure: unexpected cup contact or displacement")
        adjusted = [{side: dict(pose) for side, pose in item.items()} for item in waypoints]
        if (self.phase == "waiting" and
                waypoints[-1]["right"]["gripper_open_fraction"] <= 0.85 and
                0.035 <= tool_cup_m <= 0.090 and
                0.020 <= tool[2] - cup[2] <= 0.110):
            self.model_anchor_xy = np.asarray(
                waypoints[-1]["right"]["position_m"][:2], dtype=float).copy()
            toward = cup[:2] - self.model_anchor_xy
            length = float(np.linalg.norm(toward))
            if length < 0.015 or length > 0.100:
                raise ValueError("model grasp waypoint is outside pre-close approach bounds")
            self.direction_xy = toward / length
            self.extra_target_xy = self.model_anchor_xy + self.direction_xy * amount
            if hold_trigger_orientation == "1":
                self.approach_quat = normalize(
                    observation["robots"]["right"]["tool_pose"]["quaternion_wxyz"])
            if early_lateral == "1":
                vertices = [np.asarray(pads[name]["nearest_vertex_world_m"][:2], dtype=float)
                            for name in ("left_finger", "right_finger")]
                separation = vertices[1]-vertices[0]
                if not 0.04 <= np.linalg.norm(separation) <= 0.14:
                    raise ValueError("pregrasp finger separation is outside centering bounds")
                self.early_lateral_origin_xy = tool[:2].copy()
                self.early_lateral_axis_xy = separation/np.linalg.norm(separation)
            self.level_before_approach = early_level == "1"
            self.phase = "leveling" if self.level_before_approach else "approaching"
            if self.level_before_approach:
                self._set_level_target(observation, pads, link_poses, tool, level_limit_deg)
        increment_m = 0.0
        remaining_m = None
        early_lateral_error_m = None
        early_lateral_step_m = 0.0
        gap_balance_paused_forward = False
        if self.phase == "approaching":
            measured_m = float(np.dot(tool[:2] - self.model_anchor_xy,
                                      self.direction_xy))
            self.progress_m = max(0.0, min(amount, measured_m))
            if self.early_lateral_axis_xy is not None:
                vertices = [np.asarray(pads[name]["nearest_vertex_world_m"][:2], dtype=float)
                            for name in ("left_finger", "right_finger")]
                separation = vertices[1]-vertices[0]
                norm = float(np.linalg.norm(separation))
                if not 0.04 <= norm <= 0.14:
                    raise ValueError("pregrasp finger separation changed outside centering bounds")
                jaw_axis = separation/norm
                early_lateral_error_m = float(np.dot(cup[:2] - (vertices[0]+vertices[1])/2,
                                                      jaw_axis))
                measured_lateral = float(np.dot(tool[:2]-self.early_lateral_origin_xy,
                                                 self.early_lateral_axis_xy))
                if abs(measured_lateral) > 0.025:
                    raise ValueError("pregrasp measured lateral correction exceeds 25 mm")
                remaining_m = max(0.0, amount-self.progress_m)
                arrived = remaining_m <= 0.002 and abs(early_lateral_error_m) <= 0.004
            else:
                delta_xy = self.extra_target_xy - tool[:2]
                remaining_m = float(np.linalg.norm(delta_xy))
                arrived = remaining_m <= 0.002
            if arrived:
                self.extra_reached = True
                if self.early_lateral_axis_xy is not None:
                    self.hold_target_xy = tool[:2].copy()
                if self.level_before_approach:
                    self.hold_target_z = float(tool[2])
                self.phase = ("waiting_for_grasp_corridor" if self.level_before_approach
                              else "leveling" if level_limit_deg else "waiting_for_grasp_corridor")
                if level_limit_deg and not self.level_before_approach:
                    self._set_level_target(observation, pads, link_poses, tool, level_limit_deg)
            else:
                if self.early_lateral_axis_xy is not None:
                    left_gap = float(pads["left_finger"]["sidewall_gap_m"])
                    right_gap = float(pads["right_finger"]["sidewall_gap_m"])
                    min_gap = min(left_gap, right_gap)
                    gap_imbalance = left_gap-right_gap
                    gap_balance_paused_forward = (balance_pad_gaps == "1" and
                        min_gap < 0.012 and abs(gap_imbalance) > 0.004)
                    early_lateral_step_m = (
                        math.copysign(0.0015, gap_imbalance)
                        if gap_balance_paused_forward else
                        float(np.clip(early_lateral_error_m, -0.0015, 0.0015))
                        if abs(early_lateral_error_m) > 0.003 else 0.0)
                    # Near one-sided contact, finish lateral centering before
                    # any further forward motion. The cup displacement and
                    # force gates above remain active throughout.
                    increment_m = (0.0 if gap_balance_paused_forward or
                                   min_gap < 0.008 and abs(early_lateral_error_m) > 0.004 else
                                   min(0.002, remaining_m))
                    step_xy = (self.direction_xy*increment_m
                               + jaw_axis*early_lateral_step_m)
                    if np.linalg.norm(step_xy) > 0.003:
                        step_xy *= 0.003/np.linalg.norm(step_xy)
                else:
                    increment_m = min(0.003, remaining_m)
                    step_xy = delta_xy / remaining_m * increment_m
                self.commanded_progress_m = max(self.commanded_progress_m,
                                                min(amount, self.progress_m + increment_m))
                # Freeze only right-arm XY while the extra displacement is
                # executed. Continuing to integrate the model's relative
                # actions here would compound the offset on every call.
                for index, waypoint in enumerate(adjusted):
                    pose = waypoint["right"]
                    position = np.asarray(pose["position_m"], dtype=float).copy()
                    position[:2] = tool[:2] + step_xy * (index + 1) / 3
                    pose["position_m"] = position.tolist()
                    if self.approach_quat is not None:
                        pose["quaternion_wxyz"] = self.approach_quat.tolist()
                    pose["gripper_open_fraction"] = 1.0
        level_step_deg = 0.0
        if self.phase == "leveling":
            tool_quat = normalize(observation["robots"]["right"]["tool_pose"]["quaternion_wxyz"])
            self.level_progress_deg = math.degrees(2 * math.acos(
                min(1.0, abs(float(np.dot(self.level_origin_quat, tool_quat))))))
            delta_q = normalize(mul(self.level_target_quat, conj(tool_quat)))
            if delta_q[0] < 0:
                delta_q = -delta_q
            axis_norm = float(np.linalg.norm(delta_q[1:]))
            remaining_deg = math.degrees(2 * math.atan2(axis_norm, delta_q[0]))
            remaining_pos_m = float(np.linalg.norm(self.level_target_pos-tool))
            fixed_world = {}
            for name in ("left_finger", "right_finger"):
                pose = link_poses[name]
                fixed_world[name] = (np.asarray(pose["position_m"], dtype=float)
                                     + rotate(normalize(pose["quaternion_wxyz"]),
                                              self.level_finger_local[name]))
            fixed_height_gap_m = abs(fixed_world["left_finger"][2]-fixed_world["right_finger"][2])
            if remaining_deg <= 1.5 and remaining_pos_m <= 0.002:
                if fixed_height_gap_m > 0.003:
                    raise ValueError("fixed finger points remain at unequal heights after wrist leveling")
                self.phase = ("approaching" if self.level_before_approach and
                              not self.extra_reached else "waiting_for_grasp_corridor")
                if self.phase == "approaching":
                    self.approach_quat = self.level_target_quat.copy()
            else:
                room = level_limit_deg - self.level_progress_deg
                if room < 0.5 and remaining_deg > 1.5:
                    raise ValueError("wrist leveling cap reached without aligning finger heights")
                level_step_deg = min(1.0, remaining_deg, max(0.0, room))
                axis = delta_q[1:] / axis_norm if axis_norm > 1e-9 else np.zeros(3)
                step_to_pos = self.level_target_pos-tool
                distance = float(np.linalg.norm(step_to_pos))
                step_to_pos *= min(1.0, 0.002/distance) if distance > 1e-9 else 0.0
                for index, waypoint in enumerate(adjusted):
                    angle = math.radians(level_step_deg * (index+1)/3)
                    partial_q = np.array((math.cos(angle/2), *(axis * math.sin(angle/2))))
                    pose = waypoint["right"]
                    pose["position_m"] = (tool + step_to_pos*(index+1)/3).tolist()
                    pose["quaternion_wxyz"] = normalize(mul(partial_q, tool_quat)).tolist()
                    pose["gripper_open_fraction"] = 1.0
        lateral_increment_m = 0.0
        if (lateral_amount and self.phase == "waiting_for_grasp_corridor" and
                self.extra_reached and tool_cup_m <= 0.065 and
                pad_center_error_m > 0.004 and
                cup_shift_m <= preclose_shift_limit_m and contact_n <= 0.1 and
                self.lateral_progress_m < lateral_amount - 1e-9):
            if self.lateral_direction_xy is None:
                vector = cup[:2] - pad_midpoint_xy
                self.lateral_direction_xy = vector / pad_center_error_m
            self.lateral_target_m = min(lateral_amount,
                                        max(self.lateral_target_m,
                                            self.lateral_progress_m + pad_center_error_m))
            lateral_increment_m = min(0.0015, self.lateral_target_m-self.lateral_progress_m)
            self.lateral_progress_m += lateral_increment_m
        pad_corridor = (pads["left_finger"]["sidewall_gap_m"] <= 0.015
                        and pads["right_finger"]["sidewall_gap_m"] <= 0.015
                        and all(0.015 <= pad["height_above_cup_base_m"] <= 0.080
                                for pad in pads.values())
                        and (not level_limit_deg or abs(
                            pads["left_finger"]["nearest_vertex_world_m"][2]
                            - pads["right_finger"]["nearest_vertex_world_m"][2]) <= 0.005)
                        and pad_separation_deg >= (155 if lateral_amount else 110)
                        and (not lateral_amount or pad_center_error_m <= 0.004))
        if (self.phase == "waiting_for_grasp_corridor" and pad_corridor and
                tool_cup_m <= 0.055 and 0.020 <= tool[2] - cup[2] <= 0.080 and
                cup_shift_m <= preclose_shift_limit_m and contact_n <= 0.25):
            self.phase = "closing_allowed"
        if self.phase == "waiting_for_grasp_corridor" and self.extra_target_xy is not None:
            desired_xy = (self.hold_target_xy if self.hold_target_xy is not None
                          else self.extra_target_xy).copy()
            if self.lateral_direction_xy is not None:
                desired_xy += self.lateral_direction_xy * self.lateral_progress_m
            to_target = desired_xy - tool[:2]
            distance = float(np.linalg.norm(to_target))
            step_xy = to_target * min(1.0, 0.003 / distance) if distance > 1e-9 else to_target
            for index, waypoint in enumerate(adjusted):
                position = np.asarray(waypoint["right"]["position_m"], dtype=float).copy()
                position[:2] = tool[:2] + step_xy * (index+1)/3
                # Preserve the corrected pad height/orientation throughout
                # centering. Reusing the model's changing Z/quaternion here
                # reintroduced an 11 mm finger-height mismatch before close.
                if self.level_target_pos is not None:
                    position[2] = (self.hold_target_z if self.level_before_approach
                                   else self.level_target_pos[2])
                    waypoint["right"]["quaternion_wxyz"] = self.level_target_quat.tolist()
                waypoint["right"]["position_m"] = position.tolist()
        if self.phase in ("waiting", "approaching", "leveling", "waiting_for_grasp_corridor"):
            # The model may request closure far away; keep the pads open until
            # the bounded extra approach AND an observed near-cup corridor.
            for waypoint in adjusted:
                pose = waypoint["right"]
                pose["gripper_open_fraction"] = 1.0
        if self.phase == "closing_allowed" and preload_fraction:
            model_grip = float(waypoints[-1]["right"]["gripper_open_fraction"])
            if model_grip >= 0.70:
                # Do not block the model's later release command.
                self.preload_progress = 0.0
                self.dual_contact_steps = 0
            else:
                both_forces = [float(np.linalg.norm(forces[name]))
                               for name in ("left_finger", "right_finger")]
                self.dual_contact_steps = (self.dual_contact_steps + 1
                    if min(both_forces) >= 0.05 else 0)
                if self.dual_contact_steps >= 3:
                    self.preload_progress = min(preload_fraction,
                                                self.preload_progress + 0.01)
                for waypoint in adjusted:
                    pose = waypoint["right"]
                    pose["gripper_open_fraction"] = max(
                        0.20, pose["gripper_open_fraction"] - self.preload_progress)
        audit = {"mode": "assisted_preclosure_diagnostic", "phase": self.phase,
                 "requested_m": amount,
                 "base_requested_m": base_amount,
                 "additional_requested_m": additional_amount,
                 "hold_trigger_orientation": self.approach_quat is not None,
                 "level_before_approach": self.level_before_approach,
                 "trigger_orientation_wxyz": (self.approach_quat.tolist()
                                              if self.approach_quat is not None else None),
                 "early_lateral_centering": self.early_lateral_axis_xy is not None,
                 "early_lateral_error_m": early_lateral_error_m,
                 "early_lateral_commanded_this_step_m": early_lateral_step_m,
                 "gap_balance_paused_forward": gap_balance_paused_forward,
                 "extra_commanded_total_m": self.commanded_progress_m,
                 "extra_commanded_this_step_m": increment_m,
                 "measured_approach_m": self.progress_m,
                 "model_anchor_xy_m": (self.model_anchor_xy.tolist()
                                       if self.model_anchor_xy is not None else None),
                 "extra_target_xy_m": (self.extra_target_xy.tolist()
                                       if self.extra_target_xy is not None else None),
                 "hold_target_xy_m": (self.hold_target_xy.tolist()
                                      if self.hold_target_xy is not None else None),
                 "hold_target_z_m": self.hold_target_z,
                 "extra_target_reached": self.extra_reached,
                 "wrist_level_step_deg": level_step_deg,
                 "wrist_level_total_deg": self.level_progress_deg,
                 "wrist_level_target_quaternion_wxyz": (self.level_target_quat.tolist()
                                                        if self.level_target_quat is not None else None),
                 "remaining_to_extra_target_m": remaining_m,
                 "lateral_commanded_total_m": self.lateral_progress_m,
                 "lateral_commanded_this_step_m": lateral_increment_m,
                 "lateral_target_m": self.lateral_target_m,
                 "dual_contact_steps": self.dual_contact_steps,
                 "contact_preload_fraction": self.preload_progress,
                 "tool_cup_distance_m": tool_cup_m, "horizontal_distance_m": horizontal_m,
                 "cup_xy_displacement_m": cup_shift_m, "finger_contact_force_max_N": contact_n,
                 "preclosure_cup_shift_limit_m": preclose_shift_limit_m,
                 "pad_sidewall": pads, "pad_azimuth_separation_deg": pad_separation_deg,
                 "pad_midpoint_xy_m": pad_midpoint_xy.tolist() if pad_midpoint_xy is not None else None,
                 "pad_center_error_m": pad_center_error_m,
                 "pad_corridor": pad_corridor,
                 "right_jaw_held_open": self.phase in ("waiting", "approaching", "leveling", "waiting_for_grasp_corridor")}
        return adjusted, audit

    def _set_level_target(self, observation, pads, link_poses, tool, level_limit_deg):
        self.level_origin_quat = normalize(
            observation["robots"]["right"]["tool_pose"]["quaternion_wxyz"])
        left_vertex = np.asarray(pads["left_finger"]["nearest_vertex_world_m"], dtype=float)
        right_vertex = np.asarray(pads["right_finger"]["nearest_vertex_world_m"], dtype=float)
        midpoint = (left_vertex + right_vertex) / 2
        separation = right_vertex - left_vertex
        xy_length = float(np.linalg.norm(separation[:2]))
        if not 0.04 <= xy_length <= 0.14:
            raise ValueError("finger separation is outside leveling bounds")
        angle = math.atan2(separation[2], xy_length)
        if abs(math.degrees(angle)) > level_limit_deg + 0.5:
            raise ValueError("required wrist leveling exceeds 15-degree diagnostic cap")
        axis = np.array((-separation[1], separation[0], 0.0)) / xy_length
        delta_q = np.array((math.cos(angle/2), *(axis * math.sin(angle/2))))
        self.level_target_quat = normalize(mul(delta_q, self.level_origin_quat))
        self.level_target_pos = midpoint + rotate(delta_q, tool-midpoint)
        self.level_finger_local = {}
        for name, vertex in (("left_finger", left_vertex),
                             ("right_finger", right_vertex)):
            pose = link_poses[name]
            link_pos = np.asarray(pose["position_m"], dtype=float)
            link_quat = normalize(pose["quaternion_wxyz"])
            self.level_finger_local[name] = rotate(conj(link_quat), vertex-link_pos)


PREGRASP_APPROACH = PregraspApproach()


def mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array((aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw), dtype=np.float64)


def conj(q):
    return np.array((q[0], -q[1], -q[2], -q[3]), dtype=np.float64)


def rotate(q, v):
    return mul(mul(q, np.array((0.0, *v))), conj(q))[1:]


def normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(n) or n < 1e-8:
        raise ValueError("invalid quaternion")
    return q / n


_JAW_MESH_DTYPE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)),
                            ("attributes", "<u2")])
_JAW_PIVOTS = {"left_finger": (0.015, 0.025, 0.045),
               "right_finger": (-0.015, 0.025, 0.045)}


@lru_cache(maxsize=2)
def _jaw_vertices(part):
    directory = os.environ.get("UMI_FINGER_MESH_DIR")
    if part not in _JAW_PIVOTS or not directory:
        raise ValueError("pregrasp assist requires explicit YUBI jaw mesh directory")
    path = Path(directory) / ("jaw_left.stl" if part == "left_finger" else "jaw_right.stl")
    blob = path.read_bytes()
    count = int.from_bytes(blob[80:84], "little")
    if count < 1 or count > 500000 or len(blob) != 84 + 50 * count:
        raise ValueError(f"invalid YUBI jaw mesh: {path}")
    triangles = np.frombuffer(blob, dtype=_JAW_MESH_DTYPE, count=count, offset=84)
    return (np.unique(triangles["vertices"].reshape(-1, 3), axis=0).astype(float)
            - np.asarray(_JAW_PIVOTS[part], dtype=float))


def _pad_clearance(part, pose, cup):
    q = normalize(pose["quaternion_wxyz"])
    w, x, y, z = q
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    vertices = _jaw_vertices(part) @ rotation.T + np.asarray(pose["position_m"], dtype=float)
    relative = vertices - cup
    z_relative = relative[:, 2]
    clamped_z = np.clip(z_relative, 0., .075)
    radius = .027 + (.040-.027) * clamped_z / .075
    radial = np.linalg.norm(relative[:, :2], axis=1)
    distance = np.hypot(radial-radius, z_relative-clamped_z)
    index = int(np.argmin(distance))
    return {"sidewall_gap_m": float(distance[index]),
            "height_above_cup_base_m": float(z_relative[index]),
            "azimuth_deg": math.degrees(math.atan2(relative[index, 1], relative[index, 0])),
            "nearest_vertex_world_m": vertices[index].tolist()}


def current_robot(observation, side):
    robot = observation["robots"][side]
    pose = robot["tool_pose"]
    p = np.asarray(pose["position_m"], dtype=np.float64)
    q = normalize(pose["quaternion_wxyz"])
    fraction = float(robot["gripper_open_fraction"])
    if p.shape != (3,) or not np.isfinite(p).all() or not math.isfinite(fraction) or not -0.25 <= fraction <= 1.25:
        raise ValueError(f"invalid {side} robot observation: position={p.tolist()}, open_fraction={fraction}")
    # At reset the motorized joint can slightly overshoot its nominal endpoints.
    if CALIBRATION:
        p, q = CALIBRATION.tool_to_hand(side, p, q)
    return p, q, min(1.0, max(0.0, fraction))


def encode_wrist(image, side):
    pixels = np.asarray(image)
    if pixels.shape != (480, 640, 3) or pixels.dtype != np.uint8:
        raise ValueError(f"{side} wrist image must be 480x640 RGB uint8: {pixels.shape} {pixels.dtype}")
    if int(pixels.max()) - int(pixels.min()) < 2:
        raise ValueError(f"{side} wrist image is blank")
    buffer = BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def _fraction_from_source(gripper_rad):
    if CALIBRATION:
        return CALIBRATION.sim_gripper(gripper_rad)
    value = (float(gripper_rad) - SOURCE_GRIPPER_CLOSED_RAD) / (SOURCE_GRIPPER_OPEN_RAD - SOURCE_GRIPPER_CLOSED_RAD)
    return min(1.0, max(0.0, value))


def _source_from_fraction(fraction):
    if CALIBRATION:
        return CALIBRATION.source_gripper(fraction)
    return SOURCE_GRIPPER_CLOSED_RAD + fraction * (SOURCE_GRIPPER_OPEN_RAD - SOURCE_GRIPPER_CLOSED_RAD)


def _waypoint(current, actions, *, gripper_rate_limit=0.25):
    result = {}
    diagnostics = {}
    right_translation_gain = float(os.environ.get("UMI_PI05_RIGHT_TRANSLATION_GAIN", "1"))
    if not math.isfinite(right_translation_gain) or not 1 <= right_translation_gain <= 1.350001:
        raise ValueError("right-hand translation gain must be within [1, 1.35]")
    for side, offset, grip_index in (("left", 0, 14), ("right", 7, 15)):
        position, quaternion, grip = current[side]
        initial_position, initial_quaternion = position.copy(), quaternion.copy()
        source_target = None
        for action in actions:
            delta_p = action[offset:offset+3]
            delta_q_xyzw = action[offset+3:offset+7]
            if np.linalg.norm(delta_p) > 0.04:
                raise ValueError(f"{side} predicted 30Hz position jump exceeds 4 cm")
            if side == "right":
                delta_p = delta_p * right_translation_gain
                if np.linalg.norm(delta_p) > 0.04:
                    raise ValueError("scaled right 30Hz position jump exceeds 4 cm")
            delta_q = normalize((delta_q_xyzw[3], *delta_q_xyzw[:3]))
            if 2 * math.acos(min(1.0, abs(float(delta_q[0])))) > 0.35:
                raise ValueError(f"{side} predicted 30Hz rotation jump exceeds 0.35 rad")
            position = position + rotate(quaternion, delta_p)
            quaternion = normalize(mul(quaternion, delta_q))
            source_target = float(action[grip_index])
        distance = float(np.linalg.norm(position - initial_position))
        source_delta = (position - initial_position).tolist()
        if distance > 0.08:
            raise ValueError(f"{side} 10Hz waypoint exceeds 8 cm")
        target_grip = _fraction_from_source(source_target)
        # Rate bound prevents a model spike from snapping the motorized gripper.
        target_grip = min(grip + gripper_rate_limit, max(grip - gripper_rate_limit, target_grip))
        if CALIBRATION:
            position, quaternion = CALIBRATION.hand_to_world_tool(side, position, quaternion)
            initial_position, initial_quaternion = CALIBRATION.hand_to_world_tool(side, initial_position, initial_quaternion)
            distance = float(np.linalg.norm(position - initial_position))
            if distance > 0.08:
                raise ValueError(f"{side} calibrated tool waypoint exceeds 8 cm")
        result[side] = {
            "position_m": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "gripper_open_fraction": target_grip,
        }
        diagnostics[side] = {
            "translation_gain": right_translation_gain if side == "right" else 1.0,
            "source_target_gripper_rad": source_target,
            "sim_target_gripper_rad": SIM_GRIPPER_CLOSED_RAD + target_grip * (SIM_GRIPPER_OPEN_RAD-SIM_GRIPPER_CLOSED_RAD),
            "world_position_delta_m": distance,
            "source_hand_position_delta_m": source_delta,
            "world_position_delta_vector_m": (position-initial_position).tolist(),
            "world_orientation_delta_deg": math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(initial_quaternion, quaternion)))))),
        }
    return result, diagnostics


def _waypoints_30hz(current, actions):
    """Retain each predicted 30 Hz pose and jaw target in temporal order."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (3, 16) or not np.isfinite(actions).all():
        raise ValueError("30 Hz chunk must contain exactly three finite 16D actions")
    waypoints = []
    diagnostic = None
    for index in range(1, 4):
        waypoint, diagnostic = _waypoint(
            current, actions[:index], gripper_rate_limit=0.25 * index / 3,
        )
        waypoints.append(waypoint)
    return waypoints, diagnostic


def predict(observation, step, episode):
    start = time.monotonic()
    if "images" not in observation:
        raise ValueError("simulator did not provide live wrist images")
    current = {side: current_robot(observation, side) for side in ("left", "right")}
    left_p, left_q, left_grip = current["left"]
    right_p, right_q, right_grip = current["right"]
    relative_p = rotate(conj(left_q), right_p-left_p)
    relative_q = normalize(mul(conj(left_q), right_q))
    relative_pose = [*relative_p.tolist(), *relative_q[1:].tolist(), float(relative_q[0])]
    jpg = {side: encode_wrist(observation["images"][f"{side}_wrist"], side) for side in ("left", "right")}
    payload = {
        "left_jpeg": base64.b64encode(jpg["left"]).decode("ascii"),
        "right_jpeg": base64.b64encode(jpg["right"]).decode("ascii"),
        "relative_pose_xyzw": relative_pose,
        "gripper_rad": [_source_from_fraction(left_grip), _source_from_fraction(right_grip)],
        "step": int(step),
    }
    request = Request(URL, data=json.dumps(payload, separators=(",", ":")).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=120) as response:
        prediction = json.load(response)
    if prediction.get("step") != step:
        raise ValueError("inference response step mismatch")
    expected_timing = {"pose_rows": [1, 2, 3], "gripper_rows": [0, 1, 2]}
    if os.environ.get("UMI_REQUIRE_CAUSAL_PI05") == "1":
        timing = prediction.get("action_timing") or {}
        if any(timing.get(key) != value for key, value in expected_timing.items()):
            raise ValueError("π0.5 server did not confirm causal pose/gripper row alignment")
    actions = np.asarray(prediction["actions"], dtype=np.float64)
    if actions.shape != (3, 16) or not np.isfinite(actions).all():
        raise ValueError("inference output must be 3x16 finite")
    use_30hz = os.environ.get("UMI_EXECUTE_30HZ") == "1"
    if use_30hz:
        waypoints, diagnostic = _waypoints_30hz(current, actions)
        unassisted_model_waypoint = {side: dict(waypoints[-1][side])
                                     for side in ("left", "right")}
        waypoints, pregrasp_audit = PREGRASP_APPROACH.adjust(observation, step, episode, waypoints)
        waypoint = waypoints[-1]
    else:
        waypoint, diagnostic = _waypoint(current, actions)
        unassisted_model_waypoint = {side: dict(waypoint[side])
                                     for side in ("left", "right")}
        pregrasp_audit = None
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if step < 3:
        for side in ("left", "right"):
            (AUDIT_DIR / f"input_{side}_{step:04d}.jpg").write_bytes(jpg[side])
    audit = {
        "episode": int(episode), "step": int(step),
        "observation_origin": "current_simulator_render_and_robot_state",
        "image_metadata": observation.get("image_metadata", {}),
        "image_shapes": {side: [480, 640, 3] for side in ("left", "right")},
        "image_sha256": {side: hashlib.sha256(jpg[side]).hexdigest() for side in ("left", "right")},
        "relative_pose_xyzw": relative_pose,
        "model_input_pose": {side: {"position_m": current[side][0].tolist(), "quaternion_wxyz": current[side][1].tolist()}
                            for side in ("left", "right")},
        "source_gripper_rad": payload["gripper_rad"],
        "source_action_gripper_rad": [float(actions[-1, 14]), float(actions[-1, 15])],
        "action_timing": prediction.get("action_timing"),
        'future_observation_used': False,
        "reference_light_intensity": float(os.environ.get("UMI_REFERENCE_LIGHT_INTENSITY", "220")),
        "reference_dark_fingers": os.environ.get("UMI_REFERENCE_DARK_FINGERS") == "1",
        "pregrasp_approach": pregrasp_audit,
        "unassisted_model_waypoint": unassisted_model_waypoint,
        "waypoint": waypoint, "mapping": diagnostic,
        "execution_hz": 30 if use_30hz else 10,
        "substep_waypoints": waypoints if use_30hz else None,
        "model_latency_ms": prediction["latency_ms"],
        "total_adapter_latency_ms": (time.monotonic()-start)*1000,
        "endpoint_assumption": ENDPOINT_ASSUMPTION,
        "calibration": CALIBRATION.audit() if CALIBRATION else None,
        "world_tool_pose": {s: observation["robots"][s]["tool_pose"] for s in ("left", "right")},
        "cup_observation": observation.get("objects", {}).get("cup"),
        "right_finger_link_poses": {
            name: observation["robots"]["right"].get("link_poses", {}).get(name)
            for name in ("left_finger", "right_finger")
        },
        "right_cup_contact_force_N": observation["robots"]["right"].get("cup_contact_force_N"),
        "gripper_calibration": {"source_closed_rad": SOURCE_GRIPPER_CLOSED_RAD,
                                "source_open_rad": SOURCE_GRIPPER_OPEN_RAD,
                                "sim_closed_rad": SIM_GRIPPER_CLOSED_RAD,
                                "sim_open_rad": SIM_GRIPPER_OPEN_RAD},
    }
    if CALIBRATION:
        audit["gripper_calibration"] = {**CALIBRATION.gripper, "sim_closed_rad": SIM_GRIPPER_CLOSED_RAD,
                                      "sim_open_rad": SIM_GRIPPER_OPEN_RAD}
    with (AUDIT_DIR / "online_adapter.jsonl").open("a") as output:
        output.write(json.dumps(audit, separators=(",", ":")) + "\n")
    if use_30hz:
        return {"action_dt_s": 1 / 30, "waypoints": waypoints, "execute_steps": 3}
    return {"action_dt_s": 0.1, "waypoints": [waypoint], "execute_steps": 1}
