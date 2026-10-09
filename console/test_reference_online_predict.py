import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pi05_isaac_online_adapter as adapter
from online_calibration import Reference259632

class ReferencePredictTest(unittest.TestCase):
    def test_right_translation_gain_is_explicit_and_bounded(self):
        current = {s: (np.zeros(3), np.array([1., 0., 0., 0.]), 1.)
                   for s in ("left", "right")}
        action = np.zeros((1, 16), dtype=float)
        action[0, 0] = action[0, 7] = .01
        action[0, 6] = action[0, 13] = 1.
        action[0, 14:] = .78
        with patch.object(adapter, 'CALIBRATION', None), \
             patch.dict('os.environ', {'UMI_PI05_RIGHT_TRANSLATION_GAIN': '1.35'}):
            target, details = adapter._waypoint(current, action)
            self.assertAlmostEqual(target['left']['position_m'][0], .01)
            self.assertAlmostEqual(target['right']['position_m'][0], .0135)
            self.assertEqual(details['right']['translation_gain'], 1.35)
        with patch.dict('os.environ', {'UMI_PI05_RIGHT_TRANSLATION_GAIN': '1.36'}):
            with self.assertRaisesRegex(ValueError, 'gain must be within'):
                adapter._waypoint(current, action)

    def test_pregrasp_approach_precedes_closure_and_is_capped(self):
        assist = adapter.PregraspApproach()
        def observation(y=-.06, cup_y=0., contact=0.):
            return {"objects": {"cup": {"position_m": [0., cup_y, .75]}},
                    "robots": {"right": {"tool_pose": {"position_m": [0., y, .80]},
                                         "link_poses": {s: {} for s in ("left_finger", "right_finger")},
                                         "cup_contact_force_N": {
                                             "left_finger": [contact, 0, 0],
                                             "right_finger": [0, 0, 0]}}}}
        def waypoints(y=-.05, grip=.8):
            return [{"right": {"position_m": [0., y, .80],
                               "gripper_open_fraction": grip}} for _ in range(3)]
        def pad(part, pose, cup):
            return {"sidewall_gap_m": .02, "height_above_cup_base_m": .05,
                    "azimuth_deg": -175 if part == "left_finger" else -45}
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030'}), \
             patch.object(adapter, '_pad_clearance', side_effect=pad):
            waiting, info = assist.adjust(observation(), 0, 1, waypoints(grip=.95))
            self.assertEqual(info['phase'], 'waiting')
            self.assertTrue(all(w['right']['gripper_open_fraction'] == 1.0 for w in waiting))
            measured_y = -.06
            for step in range(1, 30):
                # The model keeps changing its own target, but the extra
                # target must stay anchored to its FIRST pre-close waypoint.
                shifted, info = assist.adjust(observation(y=measured_y), step, 1,
                                            waypoints(y=-.05 + step*.001))
                self.assertEqual(shifted[0]['right']['gripper_open_fraction'], 1.0)
                measured_y = shifted[-1]['right']['position_m'][1]
                if info['phase'] == 'waiting_for_grasp_corridor':
                    break
            self.assertEqual(info['phase'], 'waiting_for_grasp_corridor')
            self.assertAlmostEqual(info['model_anchor_xy_m'][1], -.049)
            self.assertAlmostEqual(info['extra_target_xy_m'][1], -.019)
            self.assertAlmostEqual(measured_y, -.019, delta=.002)
            self.assertAlmostEqual(info['measured_approach_m'], .03, delta=.002)
            self.assertAlmostEqual(shifted[-1]['right']['gripper_open_fraction'], 1.0)
            # Releasing the jaw still requires the independent two-sided pad gate.
            with patch.object(adapter, '_pad_clearance', side_effect=lambda part, pose, cup: {
                    "sidewall_gap_m": .01, "height_above_cup_base_m": .05,
                    "azimuth_deg": -175 if part == "left_finger" else -45}):
                released, info = assist.adjust(observation(y=measured_y), step+1, 1,
                                               waypoints(y=-.005, grip=.4))
            self.assertEqual(info['phase'], 'closing_allowed')
            self.assertAlmostEqual(released[-1]['right']['gripper_open_fraction'], .4)

    def test_explicit_second_30mm_is_from_original_model_anchor_and_keeps_jaw_open(self):
        assist = adapter.PregraspApproach()
        def observation(y=-.07, contact=0.):
            return {"objects": {"cup": {"position_m": [0., 0., .75]}},
                    "robots": {"right": {"tool_pose": {"position_m": [0., y, .80],
                                                       "quaternion_wxyz": [1., 0., 0., 0.]},
                                         "link_poses": {s: {} for s in ("left_finger", "right_finger")},
                                         "cup_contact_force_N": {
                                             "left_finger": [contact, 0., 0.],
                                             "right_finger": [0., 0., 0.]}}}}
        def waypoints(y=-.06, grip=.8):
            return [{"right": {"position_m": [0., y, .80],
                               "quaternion_wxyz": [0., 0., 0., 1.],
                               "gripper_open_fraction": grip}} for _ in range(3)]
        pad = {"sidewall_gap_m": .03, "height_above_cup_base_m": .05,
               "azimuth_deg": -150.}
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                      'UMI_PREGRASP_ADDITIONAL_M': '.030',
                                      'UMI_PREGRASP_HOLD_TRIGGER_ORIENTATION': '1'}), \
             patch.object(adapter, '_pad_clearance', return_value=pad):
            assist.adjust(observation(), 0, 5, waypoints(grip=.95))
            adjusted, info = assist.adjust(observation(), 1, 5, waypoints())
            self.assertAlmostEqual(info['model_anchor_xy_m'][1], -.06)
            self.assertAlmostEqual(info['extra_target_xy_m'][1], 0.)
            self.assertAlmostEqual(info['requested_m'], .06)
            self.assertAlmostEqual(info['additional_requested_m'], .03)
            self.assertEqual(info['phase'], 'approaching')
            self.assertTrue(info['hold_trigger_orientation'])
            self.assertTrue(all(item['right']['quaternion_wxyz'] == [1., 0., 0., 0.]
                                for item in adjusted))
            self.assertTrue(all(item['right']['gripper_open_fraction'] == 1.
                                for item in adjusted))
            with self.assertRaisesRegex(ValueError, 'unexpected cup contact'):
                assist.adjust(observation(contact=.3), 2, 5, waypoints(y=-.07))
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                      'UMI_PREGRASP_ADDITIONAL_M': '.031'}):
            with self.assertRaisesRegex(ValueError, '0–30 mm explicit addition'):
                assist.adjust(observation(), 3, 5, waypoints())

    def test_early_lateral_centering_pauses_forward_motion_near_one_sided_contact(self):
        assist = adapter.PregraspApproach()
        observation = {"objects": {"cup": {"position_m": [0., 0., .75]}},
                       "robots": {"right": {
                           "tool_pose": {"position_m": [0., -.07, .80],
                                         "quaternion_wxyz": [1., 0., 0., 0.]},
                           "link_poses": {s: {} for s in ("left_finger", "right_finger")},
                           "cup_contact_force_N": {s: [0., 0., 0.]
                                                   for s in ("left_finger", "right_finger")}}}}
        waypoints = [{"right": {"position_m": [0., -.06, .80],
                                 "quaternion_wxyz": [1., 0., 0., 0.],
                                 "gripper_open_fraction": .8}} for _ in range(3)]
        def pad(part, pose, cup):
            x = -.06 if part == "left_finger" else .04
            return {"sidewall_gap_m": .02 if part == "left_finger" else .005,
                    "height_above_cup_base_m": .05,
                    "azimuth_deg": -170 if part == "left_finger" else -60,
                    "nearest_vertex_world_m": [x, -.05, .80]}
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                      'UMI_PREGRASP_ADDITIONAL_M': '.030',
                                      'UMI_PREGRASP_LATERAL_CENTER_M': '.040',
                                      'UMI_PREGRASP_EARLY_LATERAL_CENTER': '1'}), \
             patch.object(adapter, '_pad_clearance', side_effect=pad):
            adjusted, info = assist.adjust(observation, 0, 12, waypoints)
            self.assertEqual(info['phase'], 'approaching')
            self.assertAlmostEqual(info['early_lateral_error_m'], .01)
            self.assertAlmostEqual(info['extra_commanded_this_step_m'], 0.)
            self.assertAlmostEqual(info['early_lateral_commanded_this_step_m'], .0015)
            self.assertAlmostEqual(adjusted[-1]['right']['position_m'][0], .0015)
            self.assertAlmostEqual(adjusted[-1]['right']['position_m'][1], -.07)
            self.assertEqual(adjusted[-1]['right']['gripper_open_fraction'], 1.)
        def almost_centered_pad(part, pose, cup):
            x = -.052 if part == "left_finger" else .048
            return {"sidewall_gap_m": .020 if part == "left_finger" else .005,
                    "height_above_cup_base_m": .05,
                    "azimuth_deg": -170 if part == "left_finger" else -60,
                    "nearest_vertex_world_m": [x, -.05, .80]}
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                      'UMI_PREGRASP_ADDITIONAL_M': '.030',
                                      'UMI_PREGRASP_LATERAL_CENTER_M': '.040',
                                      'UMI_PREGRASP_EARLY_LATERAL_CENTER': '1',
                                      'UMI_PREGRASP_BALANCE_PAD_GAPS': '1'}), \
             patch.object(adapter, '_pad_clearance', side_effect=almost_centered_pad):
            adjusted, info = assist.adjust(observation, 1, 12, waypoints)
            self.assertAlmostEqual(info['early_lateral_error_m'], .002)
            self.assertTrue(info['gap_balance_paused_forward'])
            self.assertAlmostEqual(info['extra_commanded_this_step_m'], 0.)
            self.assertAlmostEqual(info['early_lateral_commanded_this_step_m'], .0015)
            self.assertAlmostEqual(adjusted[-1]['right']['position_m'][1], -.07)

    def test_early_wrist_level_precedes_extra_approach_and_keeps_jaw_open(self):
        assist = adapter.PregraspApproach()
        link_poses = {
            'left_finger': {'position_m': [-.05, -.05, .80],
                            'quaternion_wxyz': [1., 0., 0., 0.]},
            'right_finger': {'position_m': [.05, -.05, .82],
                             'quaternion_wxyz': [1., 0., 0., 0.]},
        }
        observation = {'objects': {'cup': {'position_m': [0., 0., .75]}},
                       'robots': {'right': {
                           'tool_pose': {'position_m': [0., -.07, .80],
                                         'quaternion_wxyz': [1., 0., 0., 0.]},
                           'link_poses': link_poses,
                           'cup_contact_force_N': {s: [0., 0., 0.]
                                                   for s in link_poses}}}}
        waypoints = [{'right': {'position_m': [0., -.06, .80],
                                 'quaternion_wxyz': [1., 0., 0., 0.],
                                 'gripper_open_fraction': .5}} for _ in range(3)]
        def pad(part, pose, cup):
            vertex = pose['position_m']
            return {'sidewall_gap_m': .02,
                    'height_above_cup_base_m': vertex[2] - cup[2],
                    'azimuth_deg': -170 if part == 'left_finger' else -10,
                    'nearest_vertex_world_m': vertex}
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                       'UMI_PREGRASP_ADDITIONAL_M': '.030',
                                       'UMI_PREGRASP_LATERAL_CENTER_M': '.040',
                                       'UMI_PREGRASP_EARLY_LATERAL_CENTER': '1',
                                       'UMI_PREGRASP_BALANCE_PAD_GAPS': '1',
                                       'UMI_PREGRASP_WRIST_LEVEL_MAX_DEG': '15',
                                       'UMI_PREGRASP_HOLD_TRIGGER_ORIENTATION': '1',
                                       'UMI_PREGRASP_EARLY_LEVEL': '1'}), \
             patch.object(adapter, '_pad_clearance', side_effect=pad):
            adjusted, info = assist.adjust(observation, 0, 12, waypoints)
        self.assertEqual(info['phase'], 'leveling')
        self.assertTrue(info['level_before_approach'])
        self.assertAlmostEqual(info['extra_commanded_this_step_m'], 0.)
        self.assertGreater(info['wrist_level_step_deg'], 0.)
        self.assertTrue(all(item['right']['gripper_open_fraction'] == 1.
                            for item in adjusted))

    def test_early_level_approach_follows_model_height(self):
        assist = adapter.PregraspApproach()
        assist.episode = 12
        assist.phase = 'approaching'
        assist.level_before_approach = True
        assist.model_anchor_xy = np.array([0., -.06])
        assist.direction_xy = np.array([0., 1.])
        assist.extra_target_xy = np.array([0., 0.])
        assist.level_target_pos = np.array([0., -.07, .83])
        assist.approach_quat = np.array([1., 0., 0., 0.])
        observation = {'objects': {'cup': {'position_m': [0., 0., .75]}},
                       'robots': {'right': {
                           'tool_pose': {'position_m': [0., -.07, .80]},
                           'link_poses': {s: {} for s in ('left_finger', 'right_finger')},
                           'cup_contact_force_N': {s: [0., 0., 0.]
                                                   for s in ('left_finger', 'right_finger')}}}}
        waypoints = [{'right': {'position_m': [0., -.06, .78],
                                 'quaternion_wxyz': [0., 0., 0., 1.],
                                 'gripper_open_fraction': .5}} for _ in range(3)]
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                       'UMI_PREGRASP_ADDITIONAL_M': '.030',
                                       'UMI_PREGRASP_WRIST_LEVEL_MAX_DEG': '15',
                                       'UMI_PREGRASP_HOLD_TRIGGER_ORIENTATION': '1',
                                       'UMI_PREGRASP_EARLY_LEVEL': '1'}), \
             patch.object(adapter, '_pad_clearance', return_value={
                 'sidewall_gap_m': .02, 'height_above_cup_base_m': .06,
                 'azimuth_deg': -150}):
            adjusted, info = assist.adjust(observation, 1, 12, waypoints)
        self.assertEqual(info['phase'], 'approaching')
        self.assertAlmostEqual(adjusted[-1]['right']['position_m'][2], .78)
        self.assertEqual(adjusted[-1]['right']['quaternion_wxyz'], [1., 0., 0., 0.])
        self.assertEqual(adjusted[-1]['right']['gripper_open_fraction'], 1.)

    def test_pregrasp_does_not_count_unexecuted_motion(self):
        assist = adapter.PregraspApproach()
        observation = {"objects": {"cup": {"position_m": [0., 0., .75]}},
                       "robots": {"right": {"tool_pose": {"position_m": [0., -.06, .80]},
                                            "link_poses": {s: {} for s in ("left_finger", "right_finger")},
                                            "cup_contact_force_N": {s: [0., 0., 0.]
                                                                    for s in ("left_finger", "right_finger")}}}}
        waypoints = [{"right": {"position_m": [0., -.05, .80],
                                  "gripper_open_fraction": .8}} for _ in range(3)]
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030'}), \
             patch.object(adapter, '_pad_clearance', return_value={
                 "sidewall_gap_m": .01, "height_above_cup_base_m": .05,
                 "azimuth_deg": -170}):
            for step in range(15):
                current_waypoints = waypoints if step else [
                    {"right": {**w["right"], "gripper_open_fraction": .95}}
                    for w in waypoints]
                adjusted, info = assist.adjust(observation, step, 1, current_waypoints)
                self.assertEqual(info['phase'], 'waiting' if step == 0 else 'approaching')
                self.assertAlmostEqual(info['measured_approach_m'], 0.)
                self.assertEqual(adjusted[-1]['right']['gripper_open_fraction'], 1.)
            self.assertAlmostEqual(info['extra_commanded_total_m'], .003)

    def test_pregrasp_approach_stops_before_pushing_cup(self):
        assist = adapter.PregraspApproach()
        def observation(cup_y=0., contact=0.):
            return {"objects": {"cup": {"position_m": [0., cup_y, .75]}},
                    "robots": {"right": {"tool_pose": {"position_m": [0., -.06, .80]},
                                         "link_poses": {s: {} for s in ("left_finger", "right_finger")},
                                         "cup_contact_force_N": {
                                             "left_finger": [contact, 0, 0],
                                             "right_finger": [0, 0, 0]}}}}
        waypoints = [{"right": {"position_m": [0., -.05, .80],
                                "gripper_open_fraction": .8}} for _ in range(3)]
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030'}), \
             patch.object(adapter, '_pad_clearance', return_value={
                 "sidewall_gap_m": .02, "height_above_cup_base_m": .05,
                 "azimuth_deg": -170}):
            assist.adjust(observation(), 0, 4, [{"right": {**w["right"],
                "gripper_open_fraction": .95}} for w in waypoints])
            with self.assertRaisesRegex(ValueError, 'unexpected cup contact'):
                assist.adjust(observation(contact=.3), 1, 4, waypoints)
            self.assertEqual(assist.progress_m, 0.)
            self.assertEqual(assist.commanded_progress_m, 0.)

    def test_pregrasp_lateral_centers_far_pad_before_closure(self):
        assist = adapter.PregraspApproach()
        obs = {"objects": {"cup": {"position_m": [0., 0., .75]}},
               "robots": {"right": {"tool_pose": {"position_m": [0., -.03, .79]},
                                    "link_poses": {s: {} for s in ("left_finger", "right_finger")},
                                    "cup_contact_force_N": {s: [0., 0., 0.]
                                                            for s in ("left_finger", "right_finger")}}}}
        waypoints = [{"right": {"position_m": [0., -.03, .79],
                                "gripper_open_fraction": .4}} for _ in range(3)]
        def pad(part, pose, cup):
            if part == "left_finger":
                return {"sidewall_gap_m": .025, "height_above_cup_base_m": .05,
                        "azimuth_deg": -175, "nearest_vertex_world_m": [-.05, 0., .80]}
            return {"sidewall_gap_m": .005, "height_above_cup_base_m": .05,
                    "azimuth_deg": -45, "nearest_vertex_world_m": [.01, -.04, .80]}
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                       'UMI_PREGRASP_LATERAL_CENTER_M': '.040'}), \
             patch.object(adapter, '_pad_clearance', side_effect=pad):
            assist.adjust(obs, 0, 8, waypoints)
            assist.progress_m = .03
            assist.phase = "waiting_for_grasp_corridor"
            assist.extra_target_xy = np.array([0., -.03])
            assist.extra_reached = True
            adjusted, info = assist.adjust(obs, 1, 8, waypoints)
            self.assertEqual(info['phase'], 'waiting_for_grasp_corridor')
            self.assertAlmostEqual(info['lateral_target_m'], np.sqrt(.02**2+.02**2))
            self.assertAlmostEqual(info['lateral_commanded_total_m'], .0015)
            self.assertAlmostEqual(adjusted[-1]['right']['position_m'][0], .0015/np.sqrt(2))
            self.assertEqual(adjusted[-1]['right']['gripper_open_fraction'], 1.0)

    def test_pregrasp_requires_measured_two_sided_center_before_closure(self):
        assist = adapter.PregraspApproach()
        assist.episode = 8
        assist.phase = "waiting_for_grasp_corridor"
        assist.extra_reached = True
        assist.extra_target_xy = np.array([0., -.03])
        obs = {"objects": {"cup": {"position_m": [0., 0., .75]}},
               "robots": {"right": {"tool_pose": {"position_m": [0., -.03, .79]},
                                    "link_poses": {s: {} for s in ("left_finger", "right_finger")},
                                    "cup_contact_force_N": {s: [0., 0., 0.]
                                                            for s in ("left_finger", "right_finger")}}}}
        waypoints = [{"right": {"position_m": [0., -.03, .79],
                                  "gripper_open_fraction": .4}} for _ in range(3)]
        def pads(offset):
            def pad(part, pose, cup):
                x = -.04 if part == "left_finger" else .04
                return {"sidewall_gap_m": .005, "height_above_cup_base_m": .05,
                        "azimuth_deg": 180 if part == "left_finger" else 0,
                        "nearest_vertex_world_m": [x, offset, .80]}
            return pad
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                       'UMI_PREGRASP_LATERAL_CENTER_M': '.040'}):
            with patch.object(adapter, '_pad_clearance', side_effect=pads(-.03)):
                held, info = assist.adjust(obs, 1, 8, waypoints)
            self.assertEqual(info['phase'], 'waiting_for_grasp_corridor')
            self.assertAlmostEqual(info['pad_center_error_m'], .03)
            self.assertEqual(held[-1]['right']['gripper_open_fraction'], 1.)
            with patch.object(adapter, '_pad_clearance', side_effect=pads(0.)):
                released, info = assist.adjust(obs, 2, 8, waypoints)
            self.assertEqual(info['phase'], 'closing_allowed')
            self.assertAlmostEqual(released[-1]['right']['gripper_open_fraction'], .4)

    def test_lateral_centering_retains_leveled_height_and_wrist(self):
        assist = adapter.PregraspApproach()
        assist.episode = 8
        assist.phase = "waiting_for_grasp_corridor"
        assist.extra_reached = True
        assist.extra_target_xy = np.array([0., -.03])
        assist.level_target_pos = np.array([0., -.03, .78])
        assist.level_target_quat = np.array([1., 0., 0., 0.])
        obs = {"objects": {"cup": {"position_m": [0., 0., .75]}},
               "robots": {"right": {"tool_pose": {"position_m": [0., -.03, .79]},
                                    "link_poses": {s: {} for s in ("left_finger", "right_finger")},
                                    "cup_contact_force_N": {s: [0., 0., 0.]
                                                            for s in ("left_finger", "right_finger")}}}}
        waypoints = [{"right": {"position_m": [0., -.03, .84],
                                  "quaternion_wxyz": [0., 1., 0., 0.],
                                  "gripper_open_fraction": .4}} for _ in range(3)]
        def pad(part, pose, cup):
            return {"sidewall_gap_m": .005, "height_above_cup_base_m": .06,
                    "azimuth_deg": 180 if part == "left_finger" else 0,
                    "nearest_vertex_world_m": [-.04 if part == "left_finger" else .04,
                                                -.03, .81]}
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                       'UMI_PREGRASP_LATERAL_CENTER_M': '.040',
                                       'UMI_PREGRASP_WRIST_LEVEL_MAX_DEG': '15'}), \
             patch.object(adapter, '_pad_clearance', side_effect=pad):
            adjusted, info = assist.adjust(obs, 1, 8, waypoints)
        self.assertEqual(info['phase'], 'waiting_for_grasp_corridor')
        self.assertEqual(adjusted[-1]['right']['quaternion_wxyz'], [1., 0., 0., 0.])
        self.assertAlmostEqual(adjusted[-1]['right']['position_m'][2], .78)
        self.assertEqual(adjusted[-1]['right']['gripper_open_fraction'], 1.)

    def test_pregrasp_levels_finger_heights_before_lateral_contact(self):
        assist = adapter.PregraspApproach()
        assist.episode = 8
        assist.phase = "leveling"
        assist.extra_reached = True
        assist.extra_target_xy = np.array([0., -.03])
        assist.level_origin_quat = np.array([1., 0., 0., 0.])
        assist.level_target_quat = np.array([np.cos(np.deg2rad(5)), 0.,
                                             np.sin(np.deg2rad(5)), 0.])
        assist.level_target_pos = np.array([0., -.03, .79])
        assist.level_finger_local = {"left_finger": np.array([-.05, 0., .80]),
                                     "right_finger": np.array([.05, 0., .82])}
        obs = {"objects": {"cup": {"position_m": [0., 0., .75]}},
               "robots": {"right": {"tool_pose": {"position_m": [0., -.03, .79],
                                                    "quaternion_wxyz": [1., 0., 0., 0.]},
                                    "link_poses": {s: {"position_m": [0., 0., 0.],
                                                        "quaternion_wxyz": [1., 0., 0., 0.]}
                                                   for s in ("left_finger", "right_finger")},
                                    "cup_contact_force_N": {s: [0., 0., 0.]
                                                            for s in ("left_finger", "right_finger")}}}}
        waypoints = [{"right": {"position_m": [0., -.03, .79],
                                  "quaternion_wxyz": [1., 0., 0., 0.],
                                  "gripper_open_fraction": .4}} for _ in range(3)]
        def pad(part, pose, cup):
            if part == "left_finger":
                return {"sidewall_gap_m": .02, "height_above_cup_base_m": .05,
                        "azimuth_deg": 180, "nearest_vertex_world_m": [-.05, 0., .80]}
            return {"sidewall_gap_m": .005, "height_above_cup_base_m": .07,
                    "azimuth_deg": 0, "nearest_vertex_world_m": [.05, 0., .82]}
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                       'UMI_PREGRASP_WRIST_LEVEL_MAX_DEG': '15'}), \
             patch.object(adapter, '_pad_clearance', side_effect=pad):
            adjusted, info = assist.adjust(obs, 1, 8, waypoints)
            self.assertEqual(info['phase'], 'leveling')
            self.assertAlmostEqual(info['wrist_level_step_deg'], 1.)
            self.assertGreater(adjusted[-1]['right']['quaternion_wxyz'][2], 0.)
            self.assertEqual(adjusted[-1]['right']['gripper_open_fraction'], 1.)

    def test_contact_preload_requires_both_fingers_and_preserves_release(self):
        assist = adapter.PregraspApproach()
        assist.episode = 8
        assist.phase = "closing_allowed"
        obs = {"objects": {"cup": {"position_m": [0., 0., .75]}},
               "robots": {"right": {"tool_pose": {"position_m": [0., -.03, .79]},
                                    "link_poses": {s: {} for s in ("left_finger", "right_finger")},
                                    "cup_contact_force_N": {s: [.1, 0., 0.]
                                                            for s in ("left_finger", "right_finger")}}}}
        def waypoints(grip):
            return [{"right": {"position_m": [0., -.03, .79],
                                "gripper_open_fraction": grip}} for _ in range(3)]
        with patch.dict('os.environ', {'UMI_PREGRASP_APPROACH_M': '.030',
                                       'UMI_PREGRASP_CONTACT_PRELOAD_FRACTION': '.10'}), \
             patch.object(adapter, '_pad_clearance', return_value={
                 "sidewall_gap_m": .005, "height_above_cup_base_m": .05,
                 "azimuth_deg": -170}):
            for step in range(1, 13):
                adjusted, info = assist.adjust(obs, step, 8, waypoints(.5))
            self.assertAlmostEqual(info['contact_preload_fraction'], .10)
            self.assertAlmostEqual(adjusted[-1]['right']['gripper_open_fraction'], .4)
            released, info = assist.adjust(obs, 13, 8, waypoints(.8))
            self.assertEqual(info['contact_preload_fraction'], 0.)
            self.assertAlmostEqual(released[-1]['right']['gripper_open_fraction'], .8)

    def test_full_predict_including_audit(self):
        obs={'robots':{s:{'tool_pose':{'position_m':[.1,.2 if s=='left' else -.2,.95],
            'quaternion_wxyz':[1,0,0,0]},'gripper_open_fraction':.8} for s in ['left','right']},
            'images':{s+'_wrist':np.random.default_rng(4).integers(0,256,(480,640,3),dtype=np.uint8) for s in ['left','right']}}
        actions=np.zeros((3,16)); actions[:,6]=actions[:,13]=1; actions[:,14:]=.624
        response={'step':0,'actions':actions.tolist(),'latency_ms':1}
        with tempfile.TemporaryDirectory() as tmp, patch.object(adapter,'CALIBRATION',Reference259632()), \
             patch.object(adapter,'AUDIT_DIR',Path(tmp)), \
             patch.object(adapter,'urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            result=adapter.predict(obs,0,0)
            audit=json.loads((Path(tmp)/'online_adapter.jsonl').read_text())
            self.assertEqual(audit['calibration']['id'],'reference_259632_v1')
            self.assertEqual(audit['gripper_calibration']['source_open_rad'],.78)
            for side in ['left','right']:
                np.testing.assert_allclose(result['waypoints'][0][side]['position_m'],obs['robots'][side]['tool_pose']['position_m'])
                self.assertAlmostEqual(result['waypoints'][0][side]['gripper_open_fraction'],.8)

    def test_three_distinct_30hz_targets_and_gripper_timing(self):
        obs={'robots':{s:{'tool_pose':{'position_m':[.1,.2 if s=='left' else -.2,.95],
            'quaternion_wxyz':[1,0,0,0]},'gripper_open_fraction':.8} for s in ['left','right']},
            'images':{s+'_wrist':np.random.default_rng(4).integers(0,256,(480,640,3),dtype=np.uint8)
                      for s in ['left','right']}}
        actions=np.zeros((3,16));actions[:,6]=actions[:,13]=1
        actions[:,7]=.01;actions[:,15]=[.58,.43,.30];actions[:,14]=.624
        response={'step':0,'actions':actions.tolist(),'latency_ms':1}
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'UMI_EXECUTE_30HZ':'1',
                                                                           'UMI_REFERENCE_LIGHT_INTENSITY':'500',
                                                                           'UMI_REFERENCE_DARK_FINGERS':'1'}), \
             patch.object(adapter,'CALIBRATION',Reference259632()), \
             patch.object(adapter,'AUDIT_DIR',Path(tmp)), \
             patch.object(adapter,'urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            result=adapter.predict(obs,0,0)
            self.assertEqual(result['execute_steps'],3)
            self.assertAlmostEqual(result['action_dt_s'],1/30)
            waypoints=[w['right'] for w in result['waypoints']]
            positions=np.array([w['position_m'] for w in waypoints])
            np.testing.assert_allclose(np.linalg.norm(np.diff(positions,axis=0),axis=1),[.01,.01],atol=1e-10)
            grips=[w['gripper_open_fraction'] for w in waypoints]
            self.assertTrue(grips[0] > grips[1] > grips[2])
            audit=json.loads((Path(tmp)/'online_adapter.jsonl').read_text())
            self.assertEqual(audit['execution_hz'],30)
            self.assertEqual(audit['reference_light_intensity'],500)
            self.assertTrue(audit['reference_dark_fingers'])
            self.assertEqual(len(audit['substep_waypoints']),3)

if __name__=='__main__': unittest.main()
