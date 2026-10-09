import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from dataset_replay.publish_tuned_branch_run import verify


class PublishTunedBranchRunTest(unittest.TestCase):
    def test_requires_corresponding_success_event_and_aligned_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            report = {
                "status": "completed", "episodes_requested": 1, "steps_requested": 215,
                "setup_requested": "replay_259632_259633_tuned",
                "scene": "/private/dual_franka_yubi_cup40k_cupfriction_trial.usda",
                "policy": "/private/umi_left_second_height_replay.py",
                "episodes": [{"ever_success": True, "first_success_policy_step": 108,
                              "policy_steps": 215, "success": False,
                              "video_frames": 646, "joint_samples": 646,
                              "transitions": [{"policy_step": step, "is_success": step == 108}
                                              for step in range(215)]}],
            }
            manifest = {"status": "completed", "setup_requested": "replay_259632_259633_tuned",
                        "video_fps": 30, "policy_fps": 10, "episodes": [{}]}
            (source / "report.json").write_text(json.dumps(report))
            (source / "manifest.json").write_text(json.dumps(manifest))
            (source / "video.mp4").write_bytes(b"0" * 2048)
            with patch("dataset_replay.publish_tuned_branch_run.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 0, "646\n", "")):
                self.assertEqual(verify(source)["verified_first_success_policy_step"], 108)
                with self.assertRaisesRegex(ValueError, 'verified simulator profile'):
                    verify(source, require_wrists=True)
                from simulator_profiles.tuned_v1.yubi_isaac_sim_env.wrist_rig import camera_pose
                mount = {'translation_m': [0,.05903,.05837], 'rotation_x_deg': 190,
                         'roll_about_optical_axis_deg': 180}
                base = {'position_m': [0,0,.9], 'quaternion_wxyz': [1,0,0,0]}
                p, q = camera_pose(base, mount)
                wrists = ('left_wrist', 'right_wrist')
                manifest.update(recorded_views=['overview', *wrists], simulator_profile='tuned_v1',
                                cameras={name: {'robot_side': name.split('_')[0], 'rigid_mount': mount,
                                                'clipping_range_m': [0.01,100.0]} for name in wrists})
                report['episodes'][0]['wrist_video_frames'] = {name: 646 for name in wrists}
                for name in wrists:
                    (source / f'video_{name}.mp4').write_bytes(b'0' * 2048)
                audit = [{'sample_index': n, 'physics_time_s': n/30,
                          'views': {name: {'robot_side': name.split('_')[0], 'base_pose': base,
                                          'camera_position_m': p.tolist(), 'camera_quaternion_wxyz': q.tolist(),
                                          'clipping_range_m': [0.01,100.0]}
                                    for name in wrists}} for n in range(646)]
                audit_path = source / 'wrist_camera_poses.jsonl'
                def write_audit():
                    audit_path.write_text('\n'.join(json.dumps(row) for row in audit))
                write_audit()
                (source / 'report.json').write_text(json.dumps(report))
                (source / 'manifest.json').write_text(json.dumps(manifest))
                self.assertEqual(verify(source, require_wrists=True)['recorded_views'], ['overview', *wrists])
                audit[100]['views']['right_wrist']['clipping_range_m'] = [1.,100.]
                write_audit()
                with self.assertRaisesRegex(ValueError, 'clipping plane'):
                    verify(source, require_wrists=True)
                audit[100]['views']['right_wrist']['clipping_range_m'] = [0.01,100.]
                audit[100]['views']['right_wrist']['robot_side'] = 'left'
                write_audit()
                with self.assertRaisesRegex(ValueError, 'wrong arm'):
                    verify(source, require_wrists=True)
                audit[100]['views']['right_wrist']['robot_side'] = 'right'
                audit[100]['views']['right_wrist']['camera_position_m'] = [3,3,3]
                write_audit()
                with self.assertRaisesRegex(ValueError, 'rigidly following'):
                    verify(source, require_wrists=True)
                report["episodes"][0]["transitions"][108]["is_success"] = False
                (source / "report.json").write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "first success"):
                    verify(source)
                report["episodes"][0]["transitions"] = report["episodes"][0]["transitions"][:10]
                (source / "report.json").write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "full 215-step"):
                    verify(source)


if __name__ == "__main__":
    unittest.main()
