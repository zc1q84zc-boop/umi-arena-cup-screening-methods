"""Controlled, observation-driven cup grasp probe for the dual-YUBI simulator.

This is a diagnostic controller, not a trained policy and not a real-robot
calibration. It targets the simulator's CAD-estimated ``yubi_tool`` pinch
frame, logs the actual tool/jaw/cup state at every 10 Hz step, and aborts
before descending or closing if the preceding approach did not converge.

Use with ``--trajectory-policy-script``. The runner executes each returned
world-frame waypoint through its bounded dual-arm differential IK.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np


_PHASES = ("align_above", "orient_above", "descend", "close", "settle", "lift", "hold")
_LIMITS = {"align_above": 70, "descend": 55, "close": 45, "lift": 80}
_TOLERANCE_M = {"align_above": 0.005, "descend": 0.006, "lift": 0.018}
_STATE: dict | None = None


def _xyz(values, label: str) -> list[float]:
    if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} must be three finite metres")
    return [float(value) for value in values]


def _distance(a, b) -> float:
    return math.dist(a, b)


def _offset(episode: int) -> list[float]:
    grid = os.environ.get("SIM_GRASP_OFFSET_GRID_JSON")
    if grid is not None:
        offsets = json.loads(grid)
        if not isinstance(offsets, list) or not offsets or episode >= len(offsets):
            raise ValueError("SIM_GRASP_OFFSET_GRID_JSON must contain an offset for each episode")
        values = offsets[episode]
    else:
        values = os.environ.get("SIM_GRASP_OFFSET_XYZ_M", "0,0,0").split(",")
    result = _xyz(values, "SIM_GRASP_OFFSET_XYZ_M")
    # The CAD probe found the jaw contact patch about 40 mm above the authored
    # tool frame in world Z at the reference approach orientation.  Keep this
    # larger *diagnostic-only* vertical search range separate from XY, where
    # an offset large enough to shove the cup would invalidate the probe.
    if any(abs(value) > 0.025 for value in result[:2]) or abs(result[2]) > 0.055:
        raise ValueError("grasp offsets are bounded to +/-25 mm XY and +/-55 mm Z")
    return result


def _lift_increment() -> float | None:
    raw = os.environ.get("SIM_GRASP_LIFT_STEP_M")
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value) or not 0.001 <= value <= 0.01:
        raise ValueError("SIM_GRASP_LIFT_STEP_M must be 0.001 to 0.01 m per policy step")
    return value


def _descent_increment() -> float | None:
    raw = os.environ.get("SIM_GRASP_DESCEND_STEP_M")
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value) or not 0.001 <= value <= 0.01:
        raise ValueError("SIM_GRASP_DESCEND_STEP_M must be 0.001 to 0.01 m per policy step")
    return value


def _bounded_xy_target(current: list[float], target: list[float],
                       max_step_m: float | None) -> list[float]:
    """Limit diagnostic lateral approach before asking the IK executor to move."""
    if max_step_m is None:
        return target
    delta = np.asarray(target[:2], dtype=float) - np.asarray(current[:2], dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= max_step_m:
        return target
    bounded = np.asarray(current[:2], dtype=float) + delta * max_step_m / distance
    return [float(bounded[0]), float(bounded[1]), target[2]]


def _hold_jaw_delta() -> float | None:
    raw = os.environ.get("SIM_GRASP_HOLD_JAW_DELTA_RAD")
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value) or not 0.005 <= value <= 0.08:
        raise ValueError("SIM_GRASP_HOLD_JAW_DELTA_RAD must be 0.005 to 0.08 rad")
    return value


def _close_increment() -> float | None:
    raw = os.environ.get("SIM_GRASP_CLOSE_STEP_RAD")
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value) or not 0.005 <= value <= 0.04:
        raise ValueError("SIM_GRASP_CLOSE_STEP_RAD must be 0.005 to 0.04 rad per policy step")
    return value


def _hold_open_fraction(state: dict) -> float:
    delta = state["hold_jaw_delta"]
    if delta is None:
        return 0.0
    target_q = max(0.0, state["grasp_jaw_q_rad"] - delta)
    return min(1.0, target_q / 0.6)


def _target_orientation(initial: list[float]) -> list[float]:
    raw = os.environ.get("SIM_GRASP_TARGET_QUAT_WXYZ")
    if raw is None:
        return initial
    values = [float(value) for value in raw.split(",")]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("SIM_GRASP_TARGET_QUAT_WXYZ must be four finite numbers")
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-9:
        raise ValueError("SIM_GRASP_TARGET_QUAT_WXYZ must be nonzero")
    return [value / norm for value in values]


def _rotate_tool_vector(quaternion_wxyz: list[float], vector: list[float]) -> list[float]:
    """Rotate a CAD-local pinch offset into the world frame."""
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-9:
        raise ValueError("invalid live tool orientation")
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    vx, vy, vz = vector
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return [vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz,
            vz+w*tz+x*ty-y*tx]


def _pinch_offset_tool() -> list[float] | None:
    raw = os.environ.get("SIM_GRASP_PINCH_OFFSET_TOOL_M")
    if raw is None:
        return None
    result = _xyz(raw.split(","), "SIM_GRASP_PINCH_OFFSET_TOOL_M")
    if math.dist(result, (0, 0, 0)) > 0.035:
        raise ValueError("CAD pinch offset must be within 35 mm of tool origin")
    return result


def _approach_offset_world() -> list[float] | None:
    raw = os.environ.get("SIM_GRASP_APPROACH_OFFSET_XYZ_M")
    if raw is None:
        return None
    result = _xyz(raw.split(","), "SIM_GRASP_APPROACH_OFFSET_XYZ_M")
    if not 0.04 <= math.dist(result, (0, 0, 0)) <= 0.15:
        raise ValueError("approach standoff must be 40 to 150 mm")
    return result


def _track_cup_xy(state: dict, cup: list[float]) -> None:
    """Diagnostic feedback for known-target grasping; never used by models."""
    if not state["track_cup_xy"] or state["phase"] not in ("lift", "hold"):
        return
    if math.dist(cup[:2], state["cup_initial"][:2]) > 0.03:
        state["phase"] = "abort"
        state["abort_reason"] = "cup moved over 30 mm from known target"
        return
    for index in (0, 1):
        delta = cup[index] - state["grasp"][index]
        increment = max(-0.002, min(0.002, delta))
        state["grasp"][index] += increment
        state["lift"][index] += increment


def _rotation_error(goal: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Shortest world-frame rotation vector from current to goal."""
    gw, gx, gy, gz = goal / np.linalg.norm(goal)
    cw, cx, cy, cz = current / np.linalg.norm(current)
    # goal * conjugate(current)
    scalar = gw * cw + gx * cx + gy * cy + gz * cz
    vector = np.array((
        -gw * cx + gx * cw - gy * cz + gz * cy,
        -gw * cy + gx * cz + gy * cw - gz * cx,
        -gw * cz - gx * cy + gy * cx + gz * cw,
    ))
    if scalar < 0:
        scalar, vector = -scalar, -vector
    size = float(np.linalg.norm(vector))
    return vector * (2 * math.atan2(size, scalar) / size) if size > 1e-12 else np.zeros(3)


def _initialize(observation: dict, episode: int) -> dict:
    cup = _xyz(observation["objects"]["cup"]["position_m"], "cup pose")
    requested_side = os.environ.get("SIM_GRASP_SIDE", "auto")
    if requested_side not in ("auto", "left", "right"):
        raise ValueError("SIM_GRASP_SIDE must be auto, left, or right")
    side = (requested_side if requested_side != "auto" else
            min(("left", "right"), key=lambda name: _distance(
                observation["robots"][name]["tool_pose"]["position_m"], cup)))
    pose = observation["robots"][side]["tool_pose"]
    orientation = [float(value) for value in pose["quaternion_wxyz"]]
    if len(orientation) != 4 or not all(math.isfinite(value) for value in orientation):
        raise ValueError("initial tool orientation is invalid")
    target_orientation = _target_orientation(orientation)
    staged_orientation = os.environ.get("SIM_GRASP_STAGED_ORIENTATION") == "1"
    if staged_orientation and os.environ.get("SIM_GRASP_TARGET_QUAT_WXYZ") is None:
        raise ValueError("SIM_GRASP_STAGED_ORIENTATION requires SIM_GRASP_TARGET_QUAT_WXYZ")
    pinch_offset_tool = _pinch_offset_tool()
    approach_offset_world = _approach_offset_world()
    track_cup_xy = os.environ.get("SIM_GRASP_TRACK_CUP_XY") == "1"
    align_tolerance_m = float(os.environ.get("SIM_GRASP_ALIGN_TOLERANCE_M", "0.005"))
    if not 0.005 <= align_tolerance_m <= 0.015:
        raise ValueError("diagnostic alignment tolerance must be 5 to 15 mm")
    offset = _offset(episode)
    lift_increment = _lift_increment()
    descent_increment = _descent_increment()
    descend_max_steps = int(os.environ.get("SIM_GRASP_DESCEND_MAX_STEPS", "55"))
    if not 55 <= descend_max_steps <= 180:
        raise ValueError("SIM_GRASP_DESCEND_MAX_STEPS must be 55 to 180")
    descend_max_xy_raw = os.environ.get("SIM_GRASP_DESCEND_MAX_XY_STEP_M")
    descend_max_xy_step = (float(descend_max_xy_raw)
                           if descend_max_xy_raw is not None else None)
    if (descend_max_xy_step is not None
            and (not math.isfinite(descend_max_xy_step)
                 or not 0.001 <= descend_max_xy_step <= 0.01)):
        raise ValueError("SIM_GRASP_DESCEND_MAX_XY_STEP_M must be 0.001 to 0.01 m")
    hold_jaw_delta = _hold_jaw_delta()
    close_increment = _close_increment()
    stall_gate = os.environ.get("SIM_GRASP_CONTACT_STALL_GATE") == "1"
    force_gate = os.environ.get("SIM_GRASP_CONTACT_FORCE_GATE") == "1"
    if stall_gate and close_increment is None:
        raise ValueError("SIM_GRASP_CONTACT_STALL_GATE requires slow closure")
    if force_gate and (close_increment is None or os.environ.get("YUBI_DIAGNOSTIC_CUP_CONTACTS") != "1"):
        raise ValueError("SIM_GRASP_CONTACT_FORCE_GATE requires slow closure and cup contact tracking")
    settle_steps = int(os.environ.get("SIM_GRASP_SETTLE_STEPS", "0"))
    brake_margin_rad = float(os.environ.get("SIM_GRASP_BRAKE_MARGIN_RAD", "0"))
    force_center = os.environ.get("SIM_GRASP_FORCE_CENTER") == "1"
    if not 0 <= settle_steps <= 30 or not math.isfinite(brake_margin_rad) or not 0 <= brake_margin_rad <= 0.06:
        raise ValueError("diagnostic settle steps must be 0-30 and brake margin 0-0.06 rad")
    if (settle_steps or force_center) and not force_gate:
        raise ValueError("grip settling and force centering require cup contact force gate")
    close_max_steps = int(os.environ.get("SIM_GRASP_CLOSE_MAX_STEPS", "45"))
    if not 45 <= close_max_steps <= 90:
        raise ValueError("SIM_GRASP_CLOSE_MAX_STEPS must be 45 to 90")
    lift_keep_orientation = os.environ.get("SIM_GRASP_LIFT_KEEP_ORIENTATION") == "1"
    descend_keep_orientation = os.environ.get("SIM_GRASP_DESCEND_KEEP_ORIENTATION") == "1"
    strict_open = os.environ.get("SIM_GRASP_REQUIRE_OPEN") == "1"
    min_open_rad = float(os.environ.get("SIM_GRASP_MIN_OPEN_RAD", "0.56"))
    if not math.isfinite(min_open_rad) or not 0.45 <= min_open_rad <= 0.60:
        raise ValueError("SIM_GRASP_MIN_OPEN_RAD must be 0.45 to 0.60 rad")
    confirmed_jaw_raw = os.environ.get("SIM_GRASP_MAX_CONFIRMED_JAW_RAD")
    max_confirmed_jaw_rad = (float(confirmed_jaw_raw)
                             if confirmed_jaw_raw is not None else None)
    if (max_confirmed_jaw_rad is not None
            and (not math.isfinite(max_confirmed_jaw_rad)
                 or not 0.30 <= max_confirmed_jaw_rad <= 0.60)):
        raise ValueError("SIM_GRASP_MAX_CONFIRMED_JAW_RAD must be 0.30 to 0.60 rad")
    base_contact_abort_raw = os.environ.get("SIM_GRASP_ABORT_BASE_CONTACT_N")
    base_contact_abort_N = (float(base_contact_abort_raw)
                            if base_contact_abort_raw is not None else None)
    if (base_contact_abort_N is not None
            and (not math.isfinite(base_contact_abort_N)
                 or not 0.005 <= base_contact_abort_N <= 1.0
                 or os.environ.get("YUBI_DIAGNOSTIC_CUP_CONTACTS") != "1")):
        raise ValueError("SIM_GRASP_ABORT_BASE_CONTACT_N requires contact tracking and 0.005-1.0 N")
    grasp = [cup[index] + offset[index] for index in range(3)]
    grasp[2] += 0.058  # Near the upper half of the nominal 75 mm cup.
    above = ([grasp[index] + approach_offset_world[index] for index in range(3)]
             if approach_offset_world is not None
             else [grasp[0], grasp[1], cup[2] + 0.19])
    lift = [grasp[0], grasp[1], cup[2] + 0.20]
    trace_dir = os.environ.get("SIM_ADAPTER_AUDIT_DIR")
    trace_name = (f"grasp_calibration_episode{episode:02d}.jsonl"
                  if os.environ.get("SIM_GRASP_OFFSET_GRID_JSON") is not None
                  else "grasp_calibration.jsonl")
    trace = Path(trace_dir) / trace_name if trace_dir else None
    if trace is not None:
        trace.parent.mkdir(parents=True, exist_ok=True)
        with trace.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "event": "reset", "episode": episode, "side": side,
                "side_selection": requested_side,
                "cup_initial_position_m": cup,
                "other_arm": "right" if side == "left" else "left",
                "pinch_frame": "CAD-estimated yubi_tool; hardware extrinsic unmeasured",
                "pinch_offset_tool_m": pinch_offset_tool,
                "approach_offset_world_m": approach_offset_world,
                "track_cup_xy": track_cup_xy,
                "align_tolerance_m": align_tolerance_m,
                "controller": os.environ.get("SIM_GRASP_CONTROLLER", "translation_ik_with_optional_orientation_hold"),
                "grasp_offset_m": offset,
                "lift_increment_m_per_step": lift_increment,
                "descent_increment_m_per_step": descent_increment,
                "descend_max_steps": descend_max_steps,
                "descend_max_xy_step_m": descend_max_xy_step,
                "hold_jaw_delta_rad": hold_jaw_delta,
                "close_increment_rad_per_step": close_increment,
                "contact_stall_gate": stall_gate,
                "contact_force_gate": force_gate,
                "settle_steps": settle_steps,
                "brake_margin_rad": brake_margin_rad,
                "force_center": force_center,
                "close_max_steps": close_max_steps,
                "lift_keep_orientation": lift_keep_orientation,
                "descend_keep_orientation": descend_keep_orientation,
                "require_open_before_descend": strict_open,
                "minimum_measured_open_rad": min_open_rad,
                "maximum_confirmed_jaw_rad": max_confirmed_jaw_rad,
                "base_contact_abort_N": base_contact_abort_N,
                "target_grasp_m": grasp, "target_above_m": above,
                "target_lift_m": lift,
                "tool_initial_pose": pose,
                "target_tool_orientation_wxyz": target_orientation,
                "staged_orientation": staged_orientation,
                "yubi_base_initial_pose": observation["robots"][side]["link_poses"]["base"],
            }, separators=(",", ":")) + "\n")
    return {
        "episode": episode, "side": side, "cup_initial": cup,
        "orientation": target_orientation, "initial_orientation": orientation,
        "staged_orientation": staged_orientation, "grasp": grasp,
        "pinch_offset_tool": pinch_offset_tool,
        "track_cup_xy": track_cup_xy,
        "align_tolerance_m": align_tolerance_m,
        "above": above, "lift": lift, "lift_increment": lift_increment,
        "descent_increment": descent_increment,
        "descend_max_steps": descend_max_steps,
        "descend_max_xy_step": descend_max_xy_step,
        "hold_jaw_delta": hold_jaw_delta,
        "close_increment": close_increment,
        "contact_stall_gate": stall_gate,
        "contact_force_gate": force_gate,
        "settle_steps": settle_steps,
        "brake_margin_rad": brake_margin_rad,
        "force_center": force_center,
        "force_center_origin_xy": grasp[:2],
        "force_center_offset_xy": [0.0, 0.0],
        "close_max_steps": close_max_steps,
        "base_contact_abort_N": base_contact_abort_N,
        "lift_keep_orientation": lift_keep_orientation,
        "descend_keep_orientation": descend_keep_orientation,
        "strict_open": strict_open,
        "min_open_rad": min_open_rad,
        "max_confirmed_jaw_rad": max_confirmed_jaw_rad,
        "phase": "align_above",
        "phase_steps": 0, "converged_steps": 0, "trace": trace,
    }


def _target(state: dict, orientation_wxyz: list[float]) -> list[float]:
    if state["phase"] in ("align_above", "orient_above"):
        desired = state["above"]
    elif state["phase"] in ("descend", "close", "settle"):
        desired = state["grasp"]
    else:
        desired = state["lift"]
    offset = state["pinch_offset_tool"]
    if offset is None:
        return desired
    world_offset = _rotate_tool_vector(orientation_wxyz, offset)
    return [desired[index] - world_offset[index] for index in range(3)]


def _ramped_lift_z(state: dict, actual_z: float, goal_z: float) -> float:
    """Advance the diagnostic lift setpoint without chasing contact-induced sag.

    An ``actual_z + step`` target retreats when the arm is pushed downward by
    the cup.  Keep the commanded setpoint monotonic instead; joint velocity,
    acceleration and collision gates still bound its physical execution.
    """
    increment = state["lift_increment"]
    if increment is None:
        return goal_z
    previous = state.setdefault("lift_command_z", actual_z)
    commanded = min(goal_z, previous + increment)
    state["lift_command_z"] = commanded
    return commanded


def _advance(state: dict, error_m: float, cup_delta_m: float, jaw_rad: float,
             contact_forces_N: dict[str, float] | None = None,
             orientation_error_rad: float | None = None,
             base_contact_force_N: float = 0.0) -> None:
    phase = state["phase"]
    if phase == "abort":
        return
    base_limit = state.get("base_contact_abort_N")
    if (phase in ("align_above", "orient_above", "descend")
            and base_limit is not None and base_contact_force_N >= base_limit):
        state["phase"] = "abort"
        state["abort_reason"] = "gripper base contacted cup before grip"
        return
    if phase == "hold":
        if state.get("contact_force_gate"):
            both_maintained = (contact_forces_N is not None
                               and contact_forces_N["left_finger"] >= 0.02
                               and contact_forces_N["right_finger"] >= 0.02)
            state["lost_contact_steps"] = (0 if both_maintained
                                           else state.get("lost_contact_steps", 0) + 1)
            if state["lost_contact_steps"] >= 3:
                state["phase"] = "abort"
                state["abort_reason"] = "two-sided cup contact lost during hold"
        return
    if phase == "orient_above":
        if cup_delta_m > 0.015:
            state["phase"], state["abort_reason"] = "abort", "cup displaced during wrist preorientation"
            return
        if orientation_error_rad is None or not math.isfinite(orientation_error_rad):
            raise ValueError("staged wrist preorientation needs measured orientation error")
        settled = error_m <= 0.008 and orientation_error_rad <= math.radians(3)
        state["converged_steps"] = state["converged_steps"] + 1 if settled else 0
        if state["converged_steps"] >= 3:
            state["phase"], state["phase_steps"], state["converged_steps"] = "descend", 0, 0
        elif state["phase_steps"] >= 60:
            state["phase"], state["abort_reason"] = "abort", "wrist preorientation did not converge"
        return
    if phase in ("align_above", "descend") and cup_delta_m > 0.015:
        state["phase"] = "abort"
        state["abort_reason"] = "cup displaced before commanded grip"
        return
    if phase == "close":
        if cup_delta_m > 0.03:
            state["phase"] = "abort"
            state["abort_reason"] = "cup displaced over 30 mm during closure"
            return
        # Wait for *measured* closure; the gripper drive can lag the command
        # by >15 policy steps, which previously caused an open-jaw lift.
        if state.get("contact_force_gate"):
            two_sided = (contact_forces_N is not None
                         and contact_forces_N["left_finger"] >= 0.05
                         and contact_forces_N["right_finger"] >= 0.05
                         and (state.get("max_confirmed_jaw_rad") is None
                              or jaw_rad <= state["max_confirmed_jaw_rad"]))
            state["close_confirmed_steps"] = (state.get("close_confirmed_steps", 0) + 1
                                               if two_sided else 0)
            required_steps = 3
        elif state.get("contact_stall_gate"):
            history = state.setdefault("jaw_history", [])
            history.append(jaw_rad)
            del history[:-8]
            stalled_against_target = (
                len(history) == 8 and max(history) - min(history) <= 0.003
                and 0.12 < jaw_rad < 0.55
                and state["close_target_q_rad"] <= jaw_rad - 0.08
            )
            state["close_confirmed_steps"] = 5 if stalled_against_target else 0
            required_steps = 5
        else:
            slow_target_ready = (state.get("close_increment") is None
                                 or state.get("close_target_q_rad", 0.6) <= 0.35)
            state["close_confirmed_steps"] = (state.get("close_confirmed_steps", 0) + 1
                                               if jaw_rad <= 0.40 and slow_target_ready else 0)
            required_steps = 5
        if state["close_confirmed_steps"] >= required_steps:
            state["grasp_jaw_q_rad"] = jaw_rad
            state["phase"] = "settle" if state.get("settle_steps", 0) else "lift"
            state["phase_steps"] = 0
            state["converged_steps"] = 0
            return
        if state["phase_steps"] >= state.get("close_max_steps", _LIMITS["close"]):
            state["phase"] = "abort"
            state["abort_reason"] = (f"two-sided cup contact not confirmed within {state.get('close_max_steps', 45)} steps"
                                     if state.get("contact_force_gate") else
                                     "gripper did not close to 0.40 rad within close-step limit")
        return
    if phase == "settle":
        if cup_delta_m > 0.03:
            state["phase"] = "abort"
            state["abort_reason"] = "cup displaced over 30 mm during grip settling"
            return
        both_maintained = (contact_forces_N is not None
                           and contact_forces_N["left_finger"] >= 0.02
                           and contact_forces_N["right_finger"] >= 0.02)
        state["lost_contact_steps"] = (0 if both_maintained
                                       else state.get("lost_contact_steps", 0) + 1)
        state["settle_confirmed_steps"] = (state.get("settle_confirmed_steps", 0) + 1
                                            if both_maintained else 0)
        if state["lost_contact_steps"] >= 3:
            state["phase"] = "abort"
            state["abort_reason"] = "two-sided cup contact lost before lift"
        elif (state["phase_steps"] >= state["settle_steps"]
              and state["settle_confirmed_steps"] >= 5):
            state["phase"] = "lift"
            state["phase_steps"] = 0
        elif state["phase_steps"] >= state["settle_steps"] + 10:
            state["phase"] = "abort"
            state["abort_reason"] = "two-sided cup contact never stabilized before lift"
        return
    if phase == "lift" and state.get("contact_force_gate"):
        both_maintained = (contact_forces_N is not None
                           and contact_forces_N["left_finger"] >= 0.02
                           and contact_forces_N["right_finger"] >= 0.02)
        state["lost_contact_steps"] = (0 if both_maintained
                                       else state.get("lost_contact_steps", 0) + 1)
        if state["lost_contact_steps"] >= 3:
            state["phase"] = "abort"
            state["abort_reason"] = "two-sided cup contact lost during lift"
            return
    if phase in _TOLERANCE_M:
        tolerance = (state.get("align_tolerance_m", _TOLERANCE_M[phase])
                     if phase == "align_above" else _TOLERANCE_M[phase])
        state["converged_steps"] = (state["converged_steps"] + 1
                                     if error_m <= tolerance else 0)
        if state["converged_steps"] >= 2:
            next_phase = {"align_above": ("orient_above" if state.get("staged_orientation") else "descend"), "descend": "close",
                          "lift": "hold"}[phase]
            state["phase"] = next_phase
            if next_phase == "close" and state.get("close_increment") is not None:
                state["close_target_q_rad"] = jaw_rad
            state["phase_steps"] = 0
            state["converged_steps"] = 0
            return
    phase_limit = (state.get("descend_max_steps", _LIMITS["descend"])
                   if phase == "descend" else _LIMITS[phase])
    if state["phase_steps"] >= phase_limit:
        state["phase"] = "abort"
        state["abort_reason"] = f"{phase} did not converge; error={error_m:.3f} m"


def _force_center(state: dict, robot: dict, contact_forces: dict[str, float] | None,
                  cup_delta_m: float) -> bool:
    """Bounded, diagnostic-only lateral centering from unilateral pad force."""
    if (not state.get("force_center") or state["phase"] != "close"
            or contact_forces is None or cup_delta_m >= 0.01):
        return False
    left = contact_forces["left_finger"]
    right = contact_forces["right_finger"]
    if left >= 0.15 and right < 0.03:
        sign = 1.0
    elif right >= 0.15 and left < 0.03:
        sign = -1.0
    else:
        return False
    left_link = np.asarray(robot["link_poses"]["left_finger"]["position_m"][:2], dtype=float)
    right_link = np.asarray(robot["link_poses"]["right_finger"]["position_m"][:2], dtype=float)
    axis = left_link - right_link
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length < 0.01:
        raise ValueError("cannot center grip from degenerate finger-link axis")
    candidate = np.asarray(state["force_center_offset_xy"], dtype=float) + sign * 0.001 * axis / length
    if float(np.linalg.norm(candidate)) > 0.01:
        return False
    state["force_center_offset_xy"] = candidate.tolist()
    for target in (state["grasp"], state["lift"]):
        target[:2] = (np.asarray(state["force_center_origin_xy"]) + candidate).tolist()
    return True


def predict(observation: dict, step: int, episode: int) -> dict:
    global _STATE
    if step == 0 or _STATE is None or _STATE["episode"] != episode:
        _STATE = _initialize(observation, episode)
    state = _STATE
    side = state["side"]
    robot = observation["robots"][side]
    tool = _xyz(robot["tool_pose"]["position_m"], "tool pose")
    tool_orientation = list(robot["tool_pose"]["quaternion_wxyz"])
    cup = _xyz(observation["objects"]["cup"]["position_m"], "cup pose")
    _track_cup_xy(state, cup)
    error = _distance(tool, _target(state, tool_orientation))
    cup_delta = _distance(cup, state["cup_initial"])
    contact_vectors = robot.get("cup_contact_force_N")
    if state["contact_force_gate"] and contact_vectors is None:
        raise ValueError("diagnostic cup contact forces missing from observation")
    contact_forces = ({name: float(np.linalg.norm(_xyz(contact_vectors[name], name)))
                       for name in ("left_finger", "right_finger")}
                      if contact_vectors is not None else None)
    base_contact_force_N = (float(np.linalg.norm(_xyz(contact_vectors["base"], "base")))
                            if contact_vectors is not None else 0.0)
    orientation_error_rad = float(np.linalg.norm(_rotation_error(
        np.asarray(state["orientation"], dtype=np.float64),
        np.asarray(tool_orientation, dtype=np.float64))))
    _advance(state, error, cup_delta, float(robot["gripper_joint_position_rad"]),
             contact_forces, orientation_error_rad, base_contact_force_N)
    force_center_applied = _force_center(state, robot, contact_forces, cup_delta)
    if state["phase"] == "descend" and "descent_orientation" not in state:
        state["descent_orientation"] = list(robot["tool_pose"]["quaternion_wxyz"])
    if state["phase"] == "lift" and "lift_orientation" not in state:
        state["lift_orientation"] = list(robot["tool_pose"]["quaternion_wxyz"])
    target = _target(state, tool_orientation)
    if state["phase"] == "descend":
        full_xy_error = math.dist(tool[:2], target[:2])
        state["unbounded_descent_xy_error_m"] = full_xy_error
    if state["phase"] == "descend" and state["descent_increment"] is not None:
        # Keep the jaw centre over the cup while descending.  A large Z error
        # in the original DLS command consumed the entire step budget and let
        # the gripper drift laterally into the cup before the close phase.
        next_z = (tool[2] if full_xy_error > 0.008
                  else max(target[2], tool[2] - state["descent_increment"]))
        target = [target[0], target[1], next_z]
    if state["phase"] == "descend":
        target = _bounded_xy_target(tool, target, state.get("descend_max_xy_step"))
    if state["phase"] == "lift" and state["lift_increment"] is not None:
        target = [target[0], target[1], _ramped_lift_z(state, tool[2], target[2])]
    pause_for_opening = (state["phase"] == "descend" and state["strict_open"]
                         and float(robot["gripper_joint_position_rad"]) < state.get("min_open_rad", 0.56))
    if pause_for_opening:
        state["opening_wait_steps"] = state.get("opening_wait_steps", 0) + 1
        # Opening is a separate physical phase.  Do not consume the descent
        # timeout while the fingers have not reached the measured open gate.
        state["phase_steps"] = max(0, state["phase_steps"] - 1)
        if state["opening_wait_steps"] > 30:
            state["phase"] = "abort"
            state["abort_reason"] = ("gripper did not reach measured open gate "
                                     f"{state.get('min_open_rad', 0.56):.2f} rad within 30 steps")
            pause_for_opening = False
    open_fraction = 1.0 if state["phase"] in ("align_above", "orient_above", "descend", "abort") else 0.0
    if state["phase"] == "close" and state["close_increment"] is not None:
        state["close_target_q_rad"] = (
            min(0.6, max(state["close_target_q_rad"],
                         float(robot["gripper_joint_position_rad"]) + 0.015))
            if force_center_applied else
            max(0.0, state["close_target_q_rad"] - state["close_increment"])
        )
        open_fraction = min(1.0, state["close_target_q_rad"] / 0.6)
    if state["phase"] == "settle":
        jaw_target = (state["grasp_jaw_q_rad"] + state["brake_margin_rad"]
                      if state["phase_steps"] <= 3 else
                      state["grasp_jaw_q_rad"] - (state["hold_jaw_delta"] or 0.0))
        open_fraction = max(0.0, min(1.0, jaw_target / 0.6))
    if state["phase"] in ("lift", "hold"):
        # Keep a small, explicit preload after measured closure instead of
        # driving all the way to the hard stop and potentially ejecting a cup.
        open_fraction = _hold_open_fraction(state)
    if state["trace"] is not None:
        arm_indices = [robot["joint_names"].index(f"panda_joint{i}") for i in range(1, 8)]
        arm_joints = [float(robot["joint_positions"][index]) for index in arm_indices]
        arm_limits = [[float(value) for value in pair]
                      for pair in robot["arm_joint_limits_rad"]]
        translation_singular_values = np.linalg.svd(
            np.asarray(robot["tool_translation_jacobian"], dtype=np.float64),
            compute_uv=False,
        ).tolist()
        row = {
            "event": "step", "step": step, "phase": state["phase"],
            "phase_steps": state["phase_steps"], "side": side,
            "target_tool_position_m": target, "tool_position_m": tool,
            "tool_orientation_wxyz": robot["tool_pose"]["quaternion_wxyz"],
            "pinch_point_world_m": ([tool[index] + value for index, value in enumerate(
                _rotate_tool_vector(tool_orientation, state["pinch_offset_tool"]))]
                if state["pinch_offset_tool"] is not None else None),
            "lift_orientation_wxyz": state.get("lift_orientation"),
            "descent_orientation_wxyz": state.get("descent_orientation"),
            "tool_error_m": _distance(tool, target),
            "cup_position_m": cup, "cup_displacement_m": cup_delta,
            "cup_quaternion_wxyz": observation["objects"]["cup"]["quaternion_wxyz"],
            "cup_linear_velocity_m_s": observation["objects"]["cup"]["linear_velocity_m_s"],
            "cup_angular_velocity_rad_s": observation["objects"]["cup"]["angular_velocity_rad_s"],
            "finger_link_poses": {
                name: robot["link_poses"][name]
                for name in ("left_finger", "right_finger")
            },
            "gripper_base_pose": robot["link_poses"]["base"],
            "driven_jaw_rad": robot["gripper_joint_position_rad"],
            "mimic_jaw_rad": robot["gripper_mimic_joint_position_rad"],
            "measured_open_fraction": robot["gripper_open_fraction"],
            "command_open_fraction": open_fraction,
            "grasp_jaw_q_rad": state.get("grasp_jaw_q_rad"),
            "settle_confirmed_steps": state.get("settle_confirmed_steps"),
            "force_center_offset_xy_m": state.get("force_center_offset_xy"),
            "force_center_applied": force_center_applied,
            "close_target_q_rad": state.get("close_target_q_rad"),
            "contact_stall_gate": state["contact_stall_gate"],
            "cup_contact_force_N": contact_forces,
            "cup_contact_force_vectors_N": contact_vectors,
            "cup_contact_points": robot.get("cup_contact_points"),
            "paused_for_opening": pause_for_opening,
            "opening_wait_steps": state.get("opening_wait_steps", 0),
            "descent_xy_error_m": math.dist(tool[:2], target[:2]) if state["phase"] == "descend" else None,
            "unbounded_descent_xy_error_m": state.get("unbounded_descent_xy_error_m")
            if state["phase"] == "descend" else None,
            "arm_joint_positions_rad": arm_joints,
            "arm_joint_limits_rad": arm_limits,
            "translation_jacobian_singular_values": translation_singular_values,
            "previous_ik_target_rad": state.get("previous_ik_target_rad"),
            "previous_ik_error_m": state.get("previous_ik_error_m"),
            "abort_reason": state.get("abort_reason"),
        }
        with state["trace"].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    state["phase_steps"] += 1
    command = {"gripper_open_fraction": open_fraction}
    if state["phase"] == "abort":
        # A gripper-only waypoint leaves the trajectory executor's previous
        # arm target active. Replace it with the observed pose so the online
        # governor decelerates instead of continuing into an obstacle.
        command.update({"position_m": tool, "quaternion_wxyz": tool_orientation})
    elif state["phase"] != "hold":
        # The trajectory executor expects a complete arm waypoint in order
        # to advance the gripper target.  Hold the *observed* pose during the
        # open gate; a gripper-only chunk left the jaw fixed at ~0.522 rad.
        command.update({"position_m": tool if pause_for_opening else target,
                        "quaternion_wxyz": tool_orientation if pause_for_opening
                        else (state["initial_orientation"] if state["phase"] == "align_above"
                              and state.get("staged_orientation") else state["orientation"])})
    return {"action_dt_s": 0.1, "waypoints": [{side: command}],
            "execute_steps": 1}


def act(observation: dict, step: int, episode: int) -> dict:
    """Translation-only bounded IK, to isolate workspace from 6D orientation.

    This intentionally leaves wrist orientation unconstrained and is only a
    diagnostic control. The simulator still clips to its USD joint limits.
    """
    chunk = predict(observation, step, episode)
    side, command = next(iter(chunk["waypoints"][0].items()))
    if "position_m" not in command:
        return {side: {"gripper_open_fraction": command["gripper_open_fraction"]}}
    robot = observation["robots"][side]
    names = robot["joint_names"]
    indices = [names.index(f"panda_joint{i}") for i in range(1, 8)]
    joints = np.asarray(robot["joint_positions"], dtype=np.float64)[indices]
    limits = np.asarray(robot["arm_joint_limits_rad"], dtype=np.float64)
    jacobian = np.asarray(robot["tool_translation_jacobian"], dtype=np.float64)
    current = np.asarray(robot["tool_pose"]["position_m"], dtype=np.float64)
    target = np.asarray(command["position_m"], dtype=np.float64)
    if (joints.shape != (7,) or limits.shape != (7, 2) or jacobian.shape != (3, 7)
            or not all(np.isfinite(array).all() for array in (joints, limits, jacobian, current, target))):
        raise ValueError("invalid live robot state for bounded translation IK")
    error = target - current
    length = float(np.linalg.norm(error))
    maximum_position_step = 0.012 if _STATE is not None and _STATE["phase"] == "descend" else 0.025
    if length > maximum_position_step:
        error *= maximum_position_step / length
    damping = 0.05
    increment = 0.7 * jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping**2 * np.eye(3), error
    )
    maintain_orientation = (_STATE is not None and
                            ((_STATE["phase"] == "lift" and _STATE["lift_keep_orientation"])
                             or (_STATE["phase"] == "descend" and _STATE["descend_keep_orientation"])))
    if maintain_orientation:
        full_jacobian = np.asarray(robot["tool_jacobian"], dtype=np.float64)
        if full_jacobian.shape != (6, 7) or not np.isfinite(full_jacobian).all():
            raise ValueError("invalid live 6D tool Jacobian")
        orientation_key = "lift_orientation" if _STATE["phase"] == "lift" else "descent_orientation"
        angular_error = _rotation_error(
            np.asarray(_STATE[orientation_key], dtype=np.float64),
            np.asarray(robot["tool_pose"]["quaternion_wxyz"], dtype=np.float64),
        )
        angular_length = float(np.linalg.norm(angular_error))
        if angular_length > 0.2:
            angular_error *= 0.2 / angular_length
        twist = np.concatenate((0.7 * error, 0.7 * angular_error))
        increment = full_jacobian.T @ np.linalg.solve(
            full_jacobian @ full_jacobian.T + damping**2 * np.eye(6), twist
        )
    joint_target = np.clip(
        joints + np.clip(increment, -0.08, 0.08), limits[:, 0], limits[:, 1]
    )
    if _STATE is not None:
        _STATE["previous_ik_target_rad"] = joint_target.tolist()
        _STATE["previous_ik_error_m"] = float(np.linalg.norm(target - current))
    return {side: {"arm_joint_targets_rad": joint_target.tolist(),
                   "gripper_open_fraction": command["gripper_open_fraction"]}}
