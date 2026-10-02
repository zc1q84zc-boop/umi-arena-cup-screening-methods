import unittest

import numpy as np

from pi05_online_server import ACTION_TIMING, next_three_causal_actions


class Pi05OnlineTimingTest(unittest.TestCase):
    def test_pose_and_jaw_use_their_distinct_source_rows(self):
        raw = np.zeros((32, 16), np.float32)
        for row in range(32):
            raw[row, :14] = row + 10
            raw[row, 14:] = row + 100
        aligned = next_three_causal_actions(raw)
        self.assertEqual(aligned.shape, (3, 16))
        np.testing.assert_array_equal(aligned[:, :14], raw[1:4, :14])
        np.testing.assert_array_equal(aligned[:, 14:], raw[:3, 14:])
        self.assertEqual(ACTION_TIMING["pose_rows"], [1, 2, 3])
        self.assertEqual(ACTION_TIMING["gripper_rows"], [0, 1, 2])

    def test_invalid_chunk_rejected_before_alignment(self):
        with self.assertRaisesRegex(ValueError, "invalid action shape"):
            next_three_causal_actions(np.zeros((3, 16)))
        broken = np.zeros((32, 16), np.float32)
        broken[12, 7] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid action shape"):
            next_three_causal_actions(broken)


if __name__ == "__main__":
    unittest.main()
