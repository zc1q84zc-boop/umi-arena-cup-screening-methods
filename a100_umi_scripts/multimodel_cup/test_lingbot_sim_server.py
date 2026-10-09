import unittest

import numpy as np

from lingbot_sim_server import REFERENCE_LEFT, qconj, qmul, qnorm, rotate


class SimFrameMappingTest(unittest.TestCase):
    def test_anchor_preserves_relative_pose(self):
        sim_left_p = np.array([-0.07, -0.27, 1.10])
        sim_right_p = np.array([-0.03, 0.25, 1.07])
        sim_left_q = qnorm([0.0, 0.72, -0.69, 0.0])
        sim_right_q = qnorm([0.2, 0.68, -0.70, 0.1])
        mapping = qnorm(qmul(REFERENCE_LEFT[3:], qconj(sim_left_q)))
        translation = REFERENCE_LEFT[:3] - rotate(mapping, sim_left_p)
        mapped_left = rotate(mapping, sim_left_p) + translation
        mapped_right = rotate(mapping, sim_right_p) + translation
        np.testing.assert_allclose(mapped_left, REFERENCE_LEFT[:3], atol=1e-9)
        np.testing.assert_allclose(
            rotate(qconj(qmul(mapping, sim_left_q)), mapped_right-mapped_left),
            rotate(qconj(sim_left_q), sim_right_p-sim_left_p), atol=1e-9,
        )
        np.testing.assert_allclose(
            qmul(qconj(qmul(mapping, sim_left_q)), qmul(mapping, sim_right_q)),
            qmul(qconj(sim_left_q), sim_right_q), atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
