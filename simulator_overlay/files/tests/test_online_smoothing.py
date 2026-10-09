import unittest

import numpy as np

from yubi_isaac_sim_env.policy_adapter import (
    TrajectoryChunkExecutor, _natural_cubic_tangents, trajectory_executor_profile,
)


class OnlineSmoothingTests(unittest.TestCase):
    def test_straight_chunk_preview_preserves_timed_velocity(self):
        positions = np.array([[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.], [.03, 0., 0.]])
        velocities = _natural_cubic_tangents(positions, np.arange(len(positions)) / 30)
        np.testing.assert_allclose(velocities[:, 0], .3, atol=1e-12)
        np.testing.assert_allclose(velocities[:, 1:], 0, atol=1e-12)
        executor = TrajectoryChunkExecutor(policy_hz=30, **trajectory_executor_profile('franka-lookahead'))
        self.assertAlmostEqual(executor.damping, .08)
        self.assertAlmostEqual(executor.lookahead_feedforward_gain, .3)
        waypoint = lambda x: {"right": {"position_m": [x, 0, 0],
                                         "quaternion_wxyz": [1, 0, 0, 0]}}
        executor.submit_chunk({"action_dt_s": 1 / 30,
                               "waypoints": [waypoint(.01), waypoint(.02), waypoint(.03)]})
        for preview in executor._queue:
            np.testing.assert_allclose(preview["right"]['_linear_velocity_m_s'], [.3, 0, 0], atol=1e-12)
            np.testing.assert_allclose(preview["right"]['_angular_velocity_rad_s'], 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
