"""Independent cup-on-plate-then-return metric for the Track 1 cup task.

The upstream environment's success flag means *only* cup-on-plate.  Keeping
this evaluator separate preserves that flag for existing runs while making the
actual two-stage instruction measurable.  It uses observed rigid-body and
gripper state; it never teleports an object or changes physics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


class CupPlateReturnEvaluator:
    def __init__(
        self,
        initial_observation: Mapping,
        *,
        origin_radius_m: float = 0.035,
        origin_height_tolerance_m: float = 0.008,
        max_tilt_deg: float = 15.0,
        max_linear_speed_m_s: float = 0.05,
        max_angular_speed_rad_s: float = 0.3,
        min_gripper_open_fraction: float = 0.65,
        return_dwell_steps: int = 5,
    ) -> None:
        self.origin = tuple(float(x) for x in initial_observation["objects"]["cup"]["position_m"])
        self.origin_radius_m = origin_radius_m
        self.origin_height_tolerance_m = origin_height_tolerance_m
        self.max_tilt_deg = max_tilt_deg
        self.max_linear_speed_m_s = max_linear_speed_m_s
        self.max_angular_speed_rad_s = max_angular_speed_rad_s
        self.min_gripper_open_fraction = min_gripper_open_fraction
        self.return_dwell_steps = return_dwell_steps
        self.plate_placed = False
        self.return_streak = 0
        self.full_success = False

    def update(self, observation: Mapping, *, plate_success: bool) -> dict:
        if plate_success:
            self.plate_placed = True
        cup = observation["objects"]["cup"]
        xyz = cup["position_m"]
        w, x, y, z = cup["quaternion_wxyz"]
        vertical = 1.0 - 2.0 * (x * x + y * y)
        released = all(
            float(robot["gripper_open_fraction"]) >= self.min_gripper_open_fraction
            for robot in observation["robots"].values()
        )
        at_origin = (
            math.dist(xyz[:2], self.origin[:2]) <= self.origin_radius_m
            and abs(xyz[2] - self.origin[2]) <= self.origin_height_tolerance_m
            and vertical >= math.cos(math.radians(self.max_tilt_deg))
            and math.sqrt(sum(v * v for v in cup["linear_velocity_m_s"])) <= self.max_linear_speed_m_s
            and math.sqrt(sum(v * v for v in cup["angular_velocity_rad_s"])) <= self.max_angular_speed_rad_s
            and released
        )
        if self.plate_placed and at_origin:
            self.return_streak += 1
        else:
            self.return_streak = 0
        self.full_success = self.return_streak >= self.return_dwell_steps
        return {
            "stage": "complete" if self.full_success else "return_to_origin" if self.plate_placed else "place_on_plate",
            "plate_placed": self.plate_placed,
            "return_candidate": at_origin,
            "return_streak": self.return_streak,
            "released": released,
            "full_task_success": self.full_success,
        }
