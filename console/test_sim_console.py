import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from sim_console import INFERENCE_BACKENDS, SimulationRunner, _int_field, _stable_lift_streak


class SimulationConsoleTests(unittest.TestCase):
    def test_only_lan_5090_is_listed_and_remote_start_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "backend_status", return_value={}) as status:
                backends = runner.inference_backends()
                self.assertEqual(len(backends), 7)
                self.assertTrue(all(INFERENCE_BACKENDS[key]["host"] == "squirrel_5090" for key in backends))
                self.assertEqual(status.call_count, 7)
            with patch.object(runner, "_service_state") as state:
                with self.assertRaisesRegex(ValueError, "Only squirrel_5090"):
                    runner._start_backend_locked("a100")
                state.assert_not_called()

    def test_catalog_reuses_snapshot_instead_of_repolling_ssh(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "gpu_status", return_value={"available": True}) as gpu, \
                    patch.object(runner, "inference_backends", return_value={}) as backends:
                first = runner.catalog()
                self.assertIs(first, runner.catalog())
                self.assertEqual(gpu.call_count, 1)
                self.assertEqual(backends.call_count, 1)

    def test_stable_lift_needs_ten_observed_upright_hold_steps(self):
        row = {
            "phase": "hold", "cup_position_m": [0, 0, 0.805],
            "cup_quaternion_wxyz": [1, 0, 0, 0],
            "cup_linear_velocity_m_s": [0, 0, 0],
            "cup_angular_velocity_rad_s": [0, 0, 0],
            "driven_jaw_rad": 0.3,
        }
        self.assertEqual(_stable_lift_streak([row] * 9, 0.75), 9)
        self.assertEqual(_stable_lift_streak([row] * 10, 0.75), 10)
        self.assertEqual(_stable_lift_streak([dict(row, cup_position_m=[0, 0, 0.76])], 0.75), 0)
        self.assertEqual(_stable_lift_streak([dict(row, cup_quaternion_wxyz=None)], 0.75), 0)

    def test_registered_policies_and_pending_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            policies = {item["id"]: item for item in runner.policies()}
            self.assertTrue(policies["hold"]["ready"])
            self.assertTrue(policies["hold-trajectory"]["ready"])
            self.assertTrue(policies["pi05-cup-clean-30000"]["ready"])
            self.assertFalse(policies["pi05-cup-clean-30000-gain135"]["ready"])
            self.assertEqual(policies["pi05-cup-clean-30000-gain135"]["supported_backends"],
                             ["rtx5090"])
            self.assertTrue(policies["pi05-cup-clean-30000-assisted"]["ready"])
            self.assertEqual(policies["pi05-cup-clean-30000-assisted"]["supported_backends"],
                             ["rtx5090"])
            self.assertFalse(policies["pi05-cup-clean-30000-assisted-physics"]["ready"])
            self.assertEqual(policies["pi05-cup-clean-30000-assisted-physics"]["supported_backends"],
                             ["rtx5090"])
            self.assertFalse(policies["pi05-cup-clean-30000-assisted-friction"]["ready"])
            self.assertFalse(policies["pi05-cup-clean-30000-assisted-friction-only"]["ready"])
            self.assertTrue(policies["pi05-cup-clean-10000"]["ready"])
            self.assertEqual(policies["pi05-cup-clean-10000"]["supported_backends"],
                             ["pi05_10000_a100", "pi05_10000_rtx5090"])
            self.assertTrue(policies["pi05-cup-clean-20000"]["ready"])
            self.assertEqual(policies["pi05-cup-clean-20000"]["supported_backends"],
                             ["pi05_20000_a100", "pi05_20000_rtx5090"])
            self.assertTrue(policies["openwam-cup-clean-10000"]["ready"])
            self.assertEqual(policies["openwam-cup-clean-10000"]["supported_backends"], ["openwam_10000_rtx5090"])
            self.assertTrue(policies["openwam-cup-clean-5000"]["ready"])
            self.assertEqual(policies["openwam-cup-clean-5000"]["supported_backends"], ["openwam_5000_rtx5090"])
            self.assertFalse(policies["lingbot-cup-clean-5000"]["ready"])
            self.assertFalse(policies["lingbot-cup-clean-10000"]["ready"])
            self.assertEqual(policies["lingbot-cup-clean-5000"]["supported_backends"], ["lingbot_5000_rtx5090"])

    def test_input_limits(self):
        for value in (True, -1, 10001, "0", 0.5):
            with self.assertRaises(ValueError):
                _int_field(value, "setup_index", 0, 10000)
        self.assertEqual(_int_field(10000, "setup_index", 0, 10000), 10000)

    def test_rejects_unknown_task_objective_before_gpu_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "gpu_status") as gpu:
                with self.assertRaisesRegex(ValueError, "unsupported task objective"):
                    runner.start({
                        "policy": "hold", "setup_index": 0, "seed": 42, "steps": 2,
                        "camera": "head", "inference_backend": "a100",
                        "task_objective": "force_success",
                    })
                gpu.assert_not_called()

    def test_unvalidated_checkpoint_cannot_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "gpu_status") as gpu:
                with self.assertRaisesRegex(ValueError, "validated simulator adapter"):
                    runner.start({
                        "policy": "some-unregistered-checkpoint", "setup_index": 0,
                        "seed": 42, "steps": 2, "camera": "head", "inference_backend": "a100",
                    })
                gpu.assert_not_called()

    def test_lingbot_is_blocked_until_revalidated(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "gpu_status") as gpu:
                with self.assertRaisesRegex(ValueError, "validated simulator adapter"):
                    runner.start({
                        "policy": "lingbot-cup-clean-5000", "setup_index": 0,
                        "seed": 42, "steps": 2, "camera": "head", "inference_backend": "a100",
                    })
                gpu.assert_not_called()

    def test_openwam_refuses_unvalidated_a100_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "gpu_status") as gpu:
                with self.assertRaisesRegex(ValueError, "not validated for this policy"):
                    runner.start({
                        "policy": "openwam-cup-clean-10000", "setup_index": 0,
                        "seed": 42, "steps": 2, "camera": "head", "inference_backend": "a100",
                    })
                gpu.assert_not_called()

    def test_gpu_contention_rejects_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "gpu_status", return_value={
                "available": False, "reason": "GPU 6 has another compute process",
            }):
                with self.assertRaisesRegex(ValueError, "another compute process"):
                    runner.start({
                        "policy": "hold", "setup_index": 0,
                        "seed": 42, "steps": 2, "camera": "head", "inference_backend": "a100",
                    })
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_orin_is_not_mislabeled_as_a_ready_model_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            self.assertFalse(runner.backend_status("orin")["ready"])
            with patch.object(runner, "gpu_status") as gpu:
                with self.assertRaisesRegex(ValueError, "not validated for this policy"):
                    runner.start({
                        "policy": "pi05-cup-clean-30000", "setup_index": 0,
                        "seed": 42, "steps": 2, "camera": "head", "inference_backend": "orin",
                    })
                gpu.assert_not_called()

    def test_managed_inference_cannot_exit_during_simulation(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            runner.active_id = "123456789abc"
            with patch.object(runner, "_remote") as remote:
                with self.assertRaisesRegex(ValueError, "simulation 123456789abc is active"):
                    runner.manage_inference("rtx5090", "stop")
                remote.assert_not_called()

    def test_managed_inference_refuses_unidentified_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "_service_state", return_value={"active": True, "pid": 1234}):
                with patch.object(runner, "_remote", return_value=subprocess.CompletedProcess([], 0, "someone_else.py", "")) as remote:
                    with self.assertRaisesRegex(ValueError, "identity mismatch"):
                        runner.manage_inference("a100", "stop")
                    self.assertEqual(remote.call_count, 1)

    def test_managed_inference_stops_only_expected_unit(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            checkpoint = "/mnt/data/benyun/workspace/pi05_cup_clean_20260923/checkpoints/pi05_cup_clean_success/pi05_cup_clean_v1/30000"
            with patch.object(runner, "_service_state", return_value={"active": True, "pid": 1234}):
                with patch.object(runner, "_remote", side_effect=[
                    subprocess.CompletedProcess([], 0, f"python pi05_online_server.py --checkpoint {checkpoint}", ""),
                    subprocess.CompletedProcess([], 0, "", ""),
                ]) as remote:
                    with patch.object(runner, "backend_status", return_value={"active": False}):
                        with patch.object(runner, "_manage_bridges") as bridges:
                            runner.manage_inference("a100", "stop")
                            bridges.assert_called_once()
                    self.assertIn("systemctl --user stop pi05-online-a100.service", remote.call_args_list[1].args[1])

    def test_switch_stops_verified_sibling_before_starting_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            states = {
                "lingbot_5000_rtx5090": {"active": True, "pid": 1234},
                "lingbot_10000_rtx5090": {"active": False, "pid": 0},
            }
            with patch.object(runner, "_service_state", side_effect=lambda key: states.get(key, {"active": False, "pid": 0})), \
                 patch.object(runner, "_verify_managed_pid") as verify, \
                 patch.object(runner, "_gpu_processes", side_effect=[{1234}, set()]), \
                 patch.object(runner, "_remote", return_value=subprocess.CompletedProcess([], 0, "", "")) as remote, \
                 patch.object(runner, "_manage_bridges") as bridges, \
                 patch.object(runner, "backend_status", return_value={"ready": True}):
                runner._start_backend_locked("lingbot_10000_rtx5090")
            verify.assert_called_with(INFERENCE_BACKENDS["lingbot_5000_rtx5090"], 1234)
            commands = [call.args[1] for call in remote.call_args_list]
            self.assertIn("systemctl --user stop lingbot-5000-squirrel.service", commands[0])
            self.assertIn("systemctl --user start lingbot-10000-squirrel.service", commands[1])
            self.assertEqual([call.args[1] for call in bridges.call_args_list], ["stop", "start"])

    def test_switch_refuses_unmanaged_gpu_process_before_stopping_sibling(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            states = {
                "lingbot_5000_rtx5090": {"active": True, "pid": 1234},
                "lingbot_10000_rtx5090": {"active": False, "pid": 0},
            }
            with patch.object(runner, "_service_state", side_effect=lambda key: states.get(key, {"active": False, "pid": 0})), \
                 patch.object(runner, "_verify_managed_pid"), \
                 patch.object(runner, "_gpu_processes", return_value={1234, 9876}), \
                 patch.object(runner, "_remote") as remote, \
                 patch.object(runner, "_manage_bridges") as bridges:
                with self.assertRaisesRegex(ValueError, "unmanaged compute process"):
                    runner._start_backend_locked("lingbot_10000_rtx5090")
            remote.assert_not_called()
            bridges.assert_not_called()

    def test_online_run_can_start_without_step_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "gpu_status", return_value={"available": True}), \
                 patch.object(runner, "_start_backend_locked") as start_backend, \
                 patch("threading.Thread.start"):
                run = runner.start({
                    "policy": "pi05-cup-clean-10000", "setup_index": 0,
                    "seed": 42, "camera": "overview", "inference_backend": "pi05_10000_rtx5090",
                    "task_objective": "plate_return", "run_until_success": True,
                })
            start_backend.assert_called_once_with("pi05_10000_rtx5090")
            self.assertTrue(run["run_until_success"])
            self.assertIsNone(run["steps"])

    def test_local_inference_launch_checks_its_own_gpu_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "gpu_status") as gpu, \
                 patch.object(runner, "_start_backend_locked") as start_backend, \
                 patch("threading.Thread.start"):
                run = runner.start({
                    "policy": "pi05-cup-clean-30000", "setup_index": 0,
                    "seed": 42, "camera": "overview", "inference_backend": "rtx5090",
                    "task_objective": "plate_return", "steps": 600,
                })
            gpu.assert_not_called()
            start_backend.assert_called_once_with("rtx5090")
            self.assertEqual(run["steps"], 600)

    def test_stop_request_uses_the_selected_run_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            run_id = "123456789abc"
            directory = Path(temporary) / run_id
            directory.mkdir()
            (directory / "metadata.json").write_text(json.dumps({
                "id": run_id, "status": "running", "policy": "lingbot-cup-clean-10000",
            }))
            runner.active_id = run_id
            with patch.object(runner, "_remote", return_value=subprocess.CompletedProcess([], 0, "", "")) as remote:
                result = runner.stop_run(run_id)
            self.assertTrue(result["stop_requested"])
            self.assertIn(f".stop_{run_id}", remote.call_args.args[1])
            self.assertNotIn("systemctl", remote.call_args.args[1])

    def test_run_api_exposes_only_small_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "123456789abc"
            directory.mkdir()
            (directory / "metadata.json").write_text(json.dumps({
                "id": "123456789abc", "status": "completed",
                "remote_dir": "/private/weights/are/not/served",
            }))
            (directory / "report.json").write_text(json.dumps({
                "status": "completed",
                "episodes": [{"success": False, "video_frames": 7,
                              "joint_samples": 7, "policy_steps": 2,
                              "initial_observation": {"private": "omitted"}}],
            }))
            runner = SimulationRunner(Path(temporary))
            (directory / "preview.jpg").write_bytes(b"jpeg preview test")
            result = runner.run("123456789abc")
            self.assertNotIn("remote_dir", result)
            self.assertNotIn("initial_observation", result["result"]["episodes"][0])
            self.assertEqual(result["result"]["episodes"][0]["video_frames"], 7)
            self.assertTrue(result["preview_available"])
            self.assertIsInstance(result["preview_version"], int)


if __name__ == "__main__":
    unittest.main()
