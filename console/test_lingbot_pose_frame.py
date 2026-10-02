import unittest

import numpy as np

from deploy_servers.lingbot_sim_server import TASK, align_causal_action_chunk, validated_pose_frame


class LingbotPoseFrameContractTests(unittest.TestCase):
    def test_both_deployed_source_hand_frames_are_accepted(self):
        required = {"left_jpeg", "right_jpeg", "step"}
        payload = {key: object() for key in required}
        self.assertIsNone(validated_pose_frame(payload, required))
        for frame in ("source_hand_mirrored_replay_prior_v1",
                      "source_hand_reference_259632_v1"):
            self.assertEqual(validated_pose_frame({**payload, "pose_frame": frame}, required), frame)
        with self.assertRaisesRegex(ValueError, "unsupported calibrated pose frame"):
            validated_pose_frame({**payload, "pose_frame": "unknown"}, required)

    def test_training_labels_and_online_chunk_use_the_same_frame_timing(self):
        self.assertEqual(TASK, "Pick up the cup with your right hand and set it on the plate")
        # Synthetic rows test the published code without distributing a
        # restricted episode. Pose row 0 is already observed; jaw row 0 is next.
        left = np.tile(np.arange(5, dtype=float)[:, None], (1, 7))
        right = left + 10
        grip = np.tile(np.arange(5, dtype=float)[:, None], (1, 2)) + 20
        actions = align_causal_action_chunk([left, right], grip)
        self.assertEqual(actions.shape, (3, 16))
        np.testing.assert_array_equal(actions[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(actions[:, 7], [11, 12, 13])
        np.testing.assert_array_equal(actions[:, 14], [20, 21, 22])


if __name__ == "__main__":
    unittest.main()
