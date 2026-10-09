import math
import os
import unittest
from unittest.mock import patch

import numpy as np
import sim_grasp_calibration_probe as probe

from analyze_grasp_geometry import cup_sidewall_distance, rotate
from sim_grasp_calibration_probe import (_advance, _bounded_xy_target, _hold_open_fraction, _initialize, _offset,
                                          _force_center, _rotate_tool_vector, _rotation_error,
                                          _ramped_lift_z, _target, _target_orientation,
                                          _track_cup_xy)


class GraspCalibrationTest(unittest.TestCase):
    def test_lateral_diagnostic_step_is_capped_without_changing_height(self):
        self.assertEqual(_bounded_xy_target([0.0, 0.0, 0.8],
                                            [0.0, 0.15, 0.75], 0.005),
                         [0.0, 0.005, 0.75])
        self.assertEqual(_bounded_xy_target([0.0, 0.0, 0.8],
                                            [0.001, 0.0, 0.75], 0.005),
                         [0.001, 0.0, 0.75])

    def test_lift_setpoint_does_not_retreat_when_contact_pushes_arm_down(self):
        state = {"lift_increment": 0.001}
        self.assertAlmostEqual(_ramped_lift_z(state, 0.788, 0.95), 0.789)
        self.assertAlmostEqual(_ramped_lift_z(state, 0.787, 0.95), 0.790)
        self.assertAlmostEqual(_ramped_lift_z(state, 0.786, 0.95), 0.791)

    def test_staged_wrist_orientation_waits_above_cup(self):
        state = {"phase": "align_above", "phase_steps": 0,
                 "converged_steps": 0, "staged_orientation": True,
                 "align_tolerance_m": 0.005}
        for _ in range(2):
            _advance(state, error_m=0.002, cup_delta_m=0.0, jaw_rad=0.55)
        self.assertEqual(state["phase"], "orient_above")
        _advance(state, error_m=0.002, cup_delta_m=0.0, jaw_rad=0.55,
                 orientation_error_rad=math.radians(8))
        self.assertEqual(state["phase"], "orient_above")
        for _ in range(3):
            _advance(state, error_m=0.002, cup_delta_m=0.0, jaw_rad=0.55,
                     orientation_error_rad=math.radians(2))
        self.assertEqual(state["phase"], "descend")

    def test_staged_wrist_orientation_aborts_if_cup_moves(self):
        state = {"phase": "orient_above", "phase_steps": 2,
                 "converged_steps": 0}
        _advance(state, error_m=0.0, cup_delta_m=0.016, jaw_rad=0.55,
                 orientation_error_rad=0.0)
        self.assertEqual(state["phase"], "abort")

    def test_unilateral_force_centers_toward_loaded_finger_with_small_bound(self):
        state = {"force_center": True, "phase": "close",
                 "force_center_origin_xy": [0.0, 0.0],
                 "force_center_offset_xy": [0.0, 0.0],
                 "grasp": [0.0, 0.0, 0.8], "lift": [0.0, 0.0, 0.95]}
        robot = {"link_poses": {
            "left_finger": {"position_m": [-0.05, -0.03, 0.8]},
            "right_finger": {"position_m": [-0.03, -0.05, 0.8]},
        }}
        self.assertTrue(_force_center(state, robot,
                                      {"left_finger": 0.3, "right_finger": 0.0}, 0.0))
        self.assertLess(state["grasp"][0], 0)
        self.assertGreater(state["grasp"][1], 0)
        self.assertAlmostEqual(state["grasp"][0], state["lift"][0])
        self.assertFalse(_force_center(state, robot,
                                       {"left_finger": 0.3, "right_finger": 0.0}, 0.02))

    def test_diagnostic_descent_budget_can_match_bounded_ik_speed(self):
        state = {"phase": "descend", "phase_steps": 56,
                 "converged_steps": 0, "descend_max_steps": 140}
        _advance(state, error_m=0.06, cup_delta_m=0.001, jaw_rad=0.53)
        self.assertEqual(state["phase"], "descend")
        state["phase_steps"] = 140
        _advance(state, error_m=0.06, cup_delta_m=0.001, jaw_rad=0.53)
        self.assertEqual(state["phase"], "abort")

    def test_base_contact_aborts_before_grip_without_disabling_other_safety(self):
        state = {"phase": "descend", "phase_steps": 5,
                 "base_contact_abort_N": 0.01}
        _advance(state, error_m=0.01, cup_delta_m=0.004, jaw_rad=0.55,
                 base_contact_force_N=0.009)
        self.assertEqual(state["phase"], "descend")
        _advance(state, error_m=0.01, cup_delta_m=0.004, jaw_rad=0.55,
                 base_contact_force_N=0.011)
        self.assertEqual(state["phase"], "abort")
        self.assertIn("base contacted", state["abort_reason"])

    def test_contact_settle_rejects_transient_grip_before_lift(self):
        state = {"phase": "close", "phase_steps": 10,
                 "converged_steps": 0, "contact_force_gate": True,
                 "settle_steps": 10}
        force = {"left_finger": 0.2, "right_finger": 0.3}
        for _ in range(3):
            _advance(state, error_m=0.0, cup_delta_m=0.001,
                     jaw_rad=0.43, contact_forces_N=force)
        self.assertEqual(state["phase"], "settle")
        for index in range(3):
            state["phase_steps"] = index + 1
            _advance(state, error_m=0.0, cup_delta_m=0.001,
                     jaw_rad=0.42,
                     contact_forces_N={"left_finger": 0.0, "right_finger": 0.1})
        self.assertEqual(state["phase"], "abort")
        self.assertIn("before lift", state["abort_reason"])

    def test_force_contact_does_not_confirm_while_jaw_is_still_wide_open(self):
        state = {"phase": "close", "phase_steps": 10,
                 "contact_force_gate": True, "settle_steps": 10,
                 "max_confirmed_jaw_rad": 0.50}
        force = {"left_finger": 0.2, "right_finger": 0.3}
        for _ in range(4):
            _advance(state, error_m=0.0, cup_delta_m=0.001,
                     jaw_rad=0.59, contact_forces_N=force)
        self.assertEqual(state["phase"], "close")
        for _ in range(3):
            _advance(state, error_m=0.0, cup_delta_m=0.001,
                     jaw_rad=0.49, contact_forces_N=force)
        self.assertEqual(state["phase"], "settle")

    def test_contact_settle_waits_for_sustained_contact(self):
        state = {"phase": "settle", "phase_steps": 0, "settle_steps": 5,
                 "settle_confirmed_steps": 0}
        force = {"left_finger": 0.2, "right_finger": 0.3}
        for index in range(5):
            state["phase_steps"] = index + 1
            _advance(state, error_m=0.0, cup_delta_m=0.001,
                     jaw_rad=0.43, contact_forces_N=force)
        self.assertEqual(state["phase"], "lift")

    def test_reference_probe_can_force_robot_right_arm(self):
        observation = {
            "objects": {"cup": {"position_m": [0.0, 0.0, 0.75]}},
            "robots": {side: {
                "tool_pose": {"position_m": [0.01 if side == "left" else 0.1, 0.0, 0.82],
                              "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                "link_poses": {"base": {"position_m": [0.0, 0.0, 1.0],
                                         "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            } for side in ("left", "right")},
        }
        with patch.dict(os.environ, {"SIM_GRASP_SIDE": "right"}):
            self.assertEqual(_initialize(observation, 0)["side"], "right")
        with patch.dict(os.environ, {"SIM_GRASP_SIDE": "swapped"}):
            with self.assertRaisesRegex(ValueError, "SIM_GRASP_SIDE"):
                _initialize(observation, 0)

    def test_wait_for_open_sends_hold_pose_and_open_gripper(self):
        state = {"episode": 0, "side": "right", "phase": "descend",
                 "phase_steps": 0, "converged_steps": 0,
                 "cup_initial": [0.0, 0.0, 0.75], "grasp": [0.0, 0.0, 0.80],
                 "above": [0.0, 0.0, 0.95], "lift": [0.0, 0.0, 0.95],
                 "orientation": [1.0, 0.0, 0.0, 0.0],
                 "pinch_offset_tool": None, "track_cup_xy": False,
                 "descent_increment": 0.005, "lift_increment": None,
                 "hold_jaw_delta": None, "close_increment": None,
                 "contact_stall_gate": False, "contact_force_gate": False,
                 "lift_keep_orientation": False, "descend_keep_orientation": False,
                 "strict_open": True, "trace": None}
        observation = {
            "objects": {"cup": {"position_m": [0.0, 0.0, 0.75]}},
            "robots": {"right": {
                "tool_pose": {"position_m": [0.0, 0.0, 0.90],
                              "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                "gripper_joint_position_rad": 0.522,
            }},
        }
        with patch.object(probe, "_STATE", state):
            chunk = probe.predict(observation, 1, 0)
        target = chunk["waypoints"][0]["right"]
        self.assertEqual(target["position_m"], [0.0, 0.0, 0.90])
        self.assertEqual(target["gripper_open_fraction"], 1.0)
        self.assertEqual(state["opening_wait_steps"], 1)
        self.assertEqual(state["phase"], "descend")
        state["min_open_rad"] = 0.50
        with patch.object(probe, "_STATE", state):
            moving = probe.predict(observation, 2, 0)["waypoints"][0]["right"]
        self.assertLess(moving["position_m"][2], 0.90)
        self.assertEqual(state["opening_wait_steps"], 1)

    def test_abort_replaces_stale_arm_target_with_observed_pose(self):
        state = {"episode": 0, "side": "right", "phase": "abort",
                 "phase_steps": 0, "cup_initial": [0.0, 0.0, 0.75],
                 "lift": [0.2, 0.2, 0.95], "pinch_offset_tool": None,
                 "orientation": [1.0, 0.0, 0.0, 0.0],
                 "track_cup_xy": False,
                 "contact_force_gate": False, "trace": None}
        pose = {"position_m": [0.01, -0.02, 0.8],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
        observation = {"objects": {"cup": {"position_m": [0.0, 0.0, 0.75]}},
                       "robots": {"right": {"tool_pose": pose,
                                             "gripper_joint_position_rad": 0.55}}}
        with patch.object(probe, "_STATE", state):
            waypoint = probe.predict(observation, 1, 0)["waypoints"][0]["right"]
        self.assertEqual(waypoint["position_m"], pose["position_m"])
        self.assertEqual(waypoint["quaternion_wxyz"], pose["quaternion_wxyz"])

    def test_quaternion_world_rotation(self):
        point = rotate(np.array([[1.0, 0.0, 0.0]]),
                       [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])[0]
        np.testing.assert_allclose(point, [0.0, 1.0, 0.0], atol=1e-7)

    def test_pinching_target_tracks_tool_orientation(self):
        offset = [0.0, -0.0084, -0.0165]
        q = [0.051, 0.688, -0.724, 0.024]
        desired = [-0.254, -0.208, 0.808]
        target = _target({"phase": "descend", "grasp": desired,
                          "pinch_offset_tool": offset}, q)
        actual = np.asarray(target) + np.asarray(_rotate_tool_vector(q, offset))
        np.testing.assert_allclose(actual, desired, atol=1e-8)

    def test_known_cup_feedback_is_bounded(self):
        state = {"track_cup_xy": True, "phase": "lift",
                 "cup_initial": [0.0, 0.0, 0.75],
                 "grasp": [0.0, 0.0, 0.8], "lift": [0.0, 0.0, 0.95]}
        _track_cup_xy(state, [0.01, -0.01, 0.75])
        self.assertEqual(state["grasp"][:2], [0.002, -0.002])
        self.assertEqual(state["lift"][:2], [0.002, -0.002])
        _track_cup_xy(state, [0.04, 0.0, 0.75])
        self.assertEqual(state["phase"], "abort")

    def test_exact_cup_rim_vertex(self):
        metric = cup_sidewall_distance(np.array([[0.04, 0.0, 0.075]]),
                                       np.zeros(3))
        self.assertAlmostEqual(metric["nearest_sidewall_vertex_m"], 0.0)
        self.assertEqual(metric["near_contact_height_range_m"], [0.075, 0.075])

    def test_rotation_error_is_shortest(self):
        goal = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
        np.testing.assert_allclose(_rotation_error(goal, np.array([1.0, 0.0, 0.0, 0.0])),
                                   [math.pi / 2, 0.0, 0.0], atol=1e-7)

    def test_abort_before_closing_when_cup_is_pushed(self):
        state = {"phase": "descend", "phase_steps": 4, "converged_steps": 0}
        _advance(state, error_m=0.02, cup_delta_m=0.02, jaw_rad=0.6)
        self.assertEqual(state["phase"], "abort")
        self.assertIn("displaced", state["abort_reason"])

    def test_lift_requires_measured_jaw_closure(self):
        state = {"phase": "close", "phase_steps": 15, "converged_steps": 0}
        _advance(state, error_m=0.0, cup_delta_m=0.004, jaw_rad=0.6)
        self.assertEqual(state["phase"], "close")
        for _ in range(5):
            _advance(state, error_m=0.0, cup_delta_m=0.004, jaw_rad=0.35)
        self.assertEqual(state["phase"], "lift")
        self.assertAlmostEqual(state["grasp_jaw_q_rad"], 0.35)

    def test_preload_stops_short_of_hard_closed(self):
        state = {"hold_jaw_delta": 0.03, "grasp_jaw_q_rad": 0.38}
        self.assertAlmostEqual(_hold_open_fraction(state), 0.35 / 0.6)
        state["hold_jaw_delta"] = None
        self.assertEqual(_hold_open_fraction(state), 0.0)

    def test_explicit_tool_orientation_normalized(self):
        with patch.dict(os.environ, {"SIM_GRASP_TARGET_QUAT_WXYZ": "2,0,0,0"}):
            self.assertEqual(_target_orientation([0, 1, 0, 0]), [1, 0, 0, 0])

    def test_episode_offset_grid_is_bounded(self):
        with patch.dict(os.environ, {"SIM_GRASP_OFFSET_GRID_JSON": "[[0,0,-0.01],[0.005,0,-0.01]]"}):
            self.assertEqual(_offset(1), [0.005, 0.0, -0.01])
            with self.assertRaises(ValueError):
                _offset(2)

    def test_abort_if_closure_shoves_cup(self):
        state = {"phase": "close", "phase_steps": 4, "converged_steps": 0}
        _advance(state, error_m=0.0, cup_delta_m=0.031, jaw_rad=0.35)
        self.assertEqual(state["phase"], "abort")

    def test_slow_closure_requires_target_below_contact_angle(self):
        state = {"phase": "close", "phase_steps": 10, "converged_steps": 0,
                 "close_increment": 0.015, "close_target_q_rad": 0.39}
        for _ in range(5):
            _advance(state, error_m=0.0, cup_delta_m=0.002, jaw_rad=0.38)
        self.assertEqual(state["phase"], "close")
        state["close_target_q_rad"] = 0.34
        for _ in range(5):
            _advance(state, error_m=0.0, cup_delta_m=0.002, jaw_rad=0.38)
        self.assertEqual(state["phase"], "lift")

    def test_stall_gate_rejects_free_motion_and_accepts_contact(self):
        state = {"phase": "close", "phase_steps": 10, "converged_steps": 0,
                 "close_increment": 0.015, "contact_stall_gate": True,
                 "close_target_q_rad": 0.35}
        for jaw in (0.50, 0.48, 0.46, 0.44, 0.42):
            _advance(state, error_m=0.0, cup_delta_m=0.002, jaw_rad=jaw)
        self.assertEqual(state["phase"], "close")
        state["close_target_q_rad"] = 0.25
        for _ in range(8):
            _advance(state, error_m=0.0, cup_delta_m=0.002, jaw_rad=0.38)
        self.assertEqual(state["phase"], "lift")
        self.assertAlmostEqual(state["grasp_jaw_q_rad"], 0.38)

    def test_force_gate_requires_both_fingers(self):
        state = {"phase": "close", "phase_steps": 10, "converged_steps": 0,
                 "contact_force_gate": True, "close_increment": 0.015}
        for _ in range(4):
            _advance(state, 0.0, 0.002, 0.38,
                     {"left_finger": 0.10, "right_finger": 0.0})
        self.assertEqual(state["phase"], "close")
        for _ in range(3):
            _advance(state, 0.0, 0.002, 0.38,
                     {"left_finger": 0.10, "right_finger": 0.10})
        self.assertEqual(state["phase"], "lift")

    def test_force_gate_aborts_if_lift_loses_contact(self):
        state = {"phase": "lift", "phase_steps": 4, "converged_steps": 0,
                 "contact_force_gate": True}
        for _ in range(3):
            _advance(state, 0.03, 0.04, 0.37,
                     {"left_finger": 0.08, "right_finger": 0.0})
        self.assertEqual(state["phase"], "abort")
        self.assertIn("contact lost", state["abort_reason"])

    def test_force_gate_aborts_if_hold_loses_contact(self):
        state = {"phase": "hold", "contact_force_gate": True}
        for _ in range(3):
            _advance(state, 0.0, 0.12, 0.37,
                     {"left_finger": 0.0, "right_finger": 0.08})
        self.assertEqual(state["phase"], "abort")


if __name__ == "__main__":
    unittest.main()
