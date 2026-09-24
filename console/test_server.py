import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from server import checkpoint_catalog, prediction_for


class ReplayMappingTest(unittest.TestCase):
    def test_all_three_checkpoints_are_registered_without_predictions(self):
        with TemporaryDirectory() as tmp:
            entries = checkpoint_catalog(Path(tmp), "revision")
            self.assertEqual([item["step"] for item in entries], [10000, 20000, 30000])
            self.assertTrue(all(item["replay_status"] == "pending" for item in entries))

    def test_baseline_is_never_model_prediction(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, "report.json").write_text(json.dumps({
                "baseline": "reference", "checkpoint_id": None, "dataset_revision": "revision",
                "episodes": []}))
            self.assertFalse(prediction_for(1, Path(tmp), "revision")["available"])

    def test_official_window_frames_follow_pose_timing(self):
        with TemporaryDirectory() as tmp:
            report = {"baseline": None, "checkpoint_id": "pi05-step-2000",
                      "dataset_revision": "revision", "settings": {"pose_timing": "previous"},
                      "episodes": [{"episode_index": 1, "repeat": 0, "status": "complete",
                                    "windows": [{"frame": 1,
                                                 "predicted_poses": [[[0] * 7] * 2] * 2,
                                                 "predicted_grippers": [[0, 0]] * 2}]}]}
            path = Path(tmp, "report.json")
            path.write_text(json.dumps(report))
            self.assertEqual(prediction_for(1, Path(tmp), "revision")["windows"][0]["indices"], [1, 2])
            report["settings"]["pose_timing"] = "next"
            path.write_text(json.dumps(report))
            self.assertEqual(prediction_for(1, Path(tmp), "revision")["windows"][0]["indices"], [2, 3])
            self.assertFalse(prediction_for(1, Path(tmp), "other-revision")["available"])
            self.assertFalse(prediction_for(1, Path(tmp), "revision", "another-checkpoint")["available"])

    def test_clean_replay_reports_grippers_and_excluded_episode(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, "report.json").write_text(json.dumps({
                "baseline": None, "checkpoint_id": "pi05-cup-clean-10000",
                "dataset_revision": "revision", "status": "complete",
                "settings": {"pose_timing": "previous"},
                "excluded_practice_episodes": [2],
                "episodes": [{"episode_index": 1, "repeat": 0, "status": "complete",
                              "training_overlap": True,
                              "windows": [{"frame": 1,
                                           "predicted_poses": [[[0] * 7] * 2],
                                           "predicted_grippers": [[0.2, 0.3]]}]}],
            }))
            result = prediction_for(1, Path(tmp), "revision", "pi05-cup-clean-10000")
            self.assertEqual(result["windows"][0]["grippers"], [[0.2, 0.3]])
            self.assertTrue(result["training_overlap"])
            self.assertIn("隔离", prediction_for(2, Path(tmp), "revision")["reason"])


if __name__ == "__main__":
    unittest.main()
