"""CPU-only checks that a plate placement is not mislabeled as task completion."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yubi_isaac_sim_env.two_stage_task import CupPlateReturnEvaluator


def observation(x=0.0, z=0.75, open_fraction=1.0, tilt=False):
    return {
        "objects": {"cup": {
            "position_m": [x, 0.0, z],
            "quaternion_wxyz": [0.9659258, 0.258819, 0, 0] if tilt else [1, 0, 0, 0],
            "linear_velocity_m_s": [0, 0, 0],
            "angular_velocity_rad_s": [0, 0, 0],
        }},
        "robots": {side: {"gripper_open_fraction": open_fraction} for side in ("left", "right")},
    }


class CupPlateReturnTests(unittest.TestCase):
    def test_plate_then_stable_released_return(self):
        metric = CupPlateReturnEvaluator(observation())
        self.assertEqual(metric.update(observation(x=0.3), plate_success=True)["stage"], "return_to_origin")
        for _ in range(4):
            self.assertFalse(metric.update(observation(), plate_success=False)["full_task_success"])
        self.assertTrue(metric.update(observation(), plate_success=False)["full_task_success"])

    def test_return_without_placement_does_not_count(self):
        metric = CupPlateReturnEvaluator(observation())
        for _ in range(10):
            self.assertEqual(metric.update(observation(), plate_success=False)["return_streak"], 0)

    def test_held_tilted_or_unstable_cup_does_not_count(self):
        metric = CupPlateReturnEvaluator(observation())
        metric.update(observation(x=0.3), plate_success=True)
        self.assertFalse(metric.update(observation(open_fraction=0.1), plate_success=False)["return_candidate"])
        self.assertFalse(metric.update(observation(tilt=True), plate_success=False)["return_candidate"])
        moving = copy.deepcopy(observation())
        moving["objects"]["cup"]["linear_velocity_m_s"] = [0.2, 0, 0]
        self.assertFalse(metric.update(moving, plate_success=False)["return_candidate"])
        self.assertEqual(metric.return_streak, 0)


if __name__ == "__main__":
    unittest.main()
