import os
import unittest
from unittest.mock import patch

import numpy as np

from yubi_isaac_sim_env.policy_adapter import TrajectoryChunkExecutor, _natural_knot_velocities


class OnlineSmoothingTests(unittest.TestCase):
    def test_straight_chunk_preview_preserves_timed_velocity(self):
        positions = np.array([[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.], [.03, 0., 0.]])
        velocities = _natural_knot_velocities(positions, 1 / 30)
        np.testing.assert_allclose(velocities[:, 0], .3, atol=1e-12)
        np.testing.assert_allclose(velocities[:, 1:], 0, atol=1e-12)
        with patch.dict(os.environ, {"UMI_TRAJECTORY_PREVIEW": "1"}):
            executor = TrajectoryChunkExecutor(policy_hz=30)
        self.assertAlmostEqual(executor.damping, .08)
        self.assertAlmostEqual(executor.preview_feedforward_gain, .3)
        waypoint = lambda x: {"right": {"position_m": [x, 0, 0],
                                         "quaternion_wxyz": [1, 0, 0, 0]}}
        executor.submit_chunk({"action_dt_s": 1 / 30,
                               "waypoints": [waypoint(.01), waypoint(.02), waypoint(.03)]})
        executor._prepare_preview({"robots": {"right": {"tool_pose": {
            "position_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}}}})
        for preview in executor._preview:
            np.testing.assert_allclose(preview["right"][0], [.3, 0, 0], atol=1e-12)
            np.testing.assert_allclose(preview["right"][1], 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
