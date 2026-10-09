import unittest

import numpy as np

from openwam_sim_server import absolute_to_body_actions


class OpenWamActionMappingTest(unittest.TestCase):
    def test_absolute_world_target_becomes_body_delta(self):
        current = {
            "left": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "right": np.array([0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        }
        predicted = np.zeros((3, 2, 7))
        for step in range(3):
            predicted[step, 0] = [0.01 * (step + 1), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            predicted[step, 1] = [0.5, 0.01 * (step + 1), 0.0, 0.0, 0.0, 0.0, 1.0]
        grips = np.full((3, 2), 0.25)
        result = absolute_to_body_actions(predicted, grips, current)
        self.assertEqual(result.shape, (3, 16))
        np.testing.assert_allclose(result[:, :3], [[0.01, 0, 0]] * 3, atol=1e-9)
        np.testing.assert_allclose(result[:, 7:10], [[0, 0.01, 0]] * 3, atol=1e-9)
        np.testing.assert_allclose(result[:, 14:], 0.25, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
