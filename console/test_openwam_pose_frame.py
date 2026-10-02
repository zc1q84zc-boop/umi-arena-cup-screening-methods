import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "deploy_servers"))
from openwam_sim_server import validated_pose_frame


class OpenWAMPoseFrameTest(unittest.TestCase):
    def test_legacy_and_both_calibrated_frames(self):
        required = {"episode", "step"}
        payload = {"episode": 0, "step": 0}
        self.assertIsNone(validated_pose_frame(payload, required))
        for frame in ("source_hand_mirrored_replay_prior_v1",
                      "source_hand_reference_259632_v1"):
            self.assertEqual(validated_pose_frame({**payload, "pose_frame": frame}, required), frame)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validated_pose_frame({**payload, "pose_frame": "unknown"}, required)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validated_pose_frame({**payload, "other": 1}, required)


if __name__ == "__main__":
    unittest.main()
