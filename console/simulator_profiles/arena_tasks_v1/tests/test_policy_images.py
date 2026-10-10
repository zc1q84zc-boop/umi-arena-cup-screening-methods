"""Offline checks for the model-facing RGB observation contract."""

import unittest
from unittest.mock import patch

import numpy as np

from yubi_isaac_sim_env import run


class _Camera:
    def get_clipping_range(self):
        return (.01,100.)

    def get_world_pose(self, camera_axes):
        assert camera_axes == "usd"
        return np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0])

    def get_current_frame(self):
        return {"rendering_time": 1.5}


class PolicyImagesTest(unittest.TestCase):
    def test_camera_gpu_pose_is_copied_to_host_before_audit(self):
        class GpuPose:
            def detach(self): return self
            def cpu(self): return np.array([1.,2.,3.])
            def __array__(self, *args): raise RuntimeError('GPU tensor cannot become NumPy directly')
        np.testing.assert_array_equal(run._cpu_array(GpuPose()), [1.,2.,3.])

    def test_wrist_recording_is_independent_of_policy_image_input(self):
        args = run._parse_args(['--record-run', '/tmp/tuned_test_unused_output', '--record-wrists'])
        self.assertTrue(args.record_wrists)
        self.assertEqual(args.policy_images, 'none')
        for side in ('left', 'right'):
            spec = run._camera_spec(side + '_wrist')
            self.assertEqual(spec['robot_side'], side)
            self.assertEqual(spec['clipping_range_m'], [0.01,100.0])
            self.assertFalse(spec['horizontal_flip_for_dataset'])

    def test_console_stop_marker_is_supported(self):
        args = run._parse_args(['--stop-file','/tmp/tuned_stop_test_marker'])
        self.assertEqual(str(args.stop_file), '/tmp/tuned_stop_test_marker')

    def test_float_rgb_is_copied_as_uint8(self):
        source = np.full((2, 4, 3), 0.5, dtype=np.float32)
        spec = {"name": "head", "resolution": [4, 2]}
        result = run._policy_rgb(source, spec)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (2, 4, 3))
        self.assertEqual(int(result[0, 0, 0]), 127)
        self.assertTrue(result.flags.writeable)
        source[0, 0, 0] = 0.0
        self.assertEqual(int(result[0, 0, 0]), 127)

    def test_three_view_observation_does_not_mutate_state(self):
        names = ("head", "left_wrist", "right_wrist")
        cameras = {name: _Camera() for name in names}
        specs = {name: {"name": name, "resolution": [4, 2], "lens_model": "test"} for name in names}
        state = {"physics_time_s": 1.0, "robots": {}}
        with patch.object(run, "_rendered_rgb", return_value=np.full((2, 4, 3), 42, dtype=np.uint8)):
            observation = run._policy_observation(state, cameras, specs, object())
        self.assertEqual(set(observation["images"]), set(names))
        self.assertEqual(observation["image_metadata"]["head"]["physics_time_s"], 1.0)
        self.assertEqual(observation["image_metadata"]["head"]["rendering_time_s"], 1.5)
        self.assertNotIn("images", state)

    def test_images_require_custom_policy(self):
        with self.assertRaises(SystemExit):
            run._parse_args(["--policy-images", "all"])

    def test_target_interpolation_flag(self):
        args = run._parse_args(["--interpolate-targets"])
        self.assertTrue(args.interpolate_targets)

    def test_portable_joint_profile_does_not_change_policy_contract(self):
        args = run._parse_args([
            "--joint-command-profile", "franka-panda-interface",
            "--trajectory-controller-profile", "franka-transfer",
        ])
        self.assertEqual(args.joint_command_profile, "franka-panda-interface")
        self.assertEqual(args.trajectory_controller_profile, "franka-transfer")
        self.assertFalse(args.interpolate_targets)

    def test_joint_profile_and_linear_interpolation_are_exclusive(self):
        with self.assertRaises(SystemExit):
            run._parse_args([
                "--interpolate-targets", "--joint-command-profile", "franka-panda-interface"
            ])


if __name__ == "__main__":
    unittest.main()
