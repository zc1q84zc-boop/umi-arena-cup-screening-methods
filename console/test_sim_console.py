import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from sim_console import INFERENCE_BACKENDS, SimulationRunner, REMOTE_ROOT, _int_field, _stable_lift_streak


class SimulationConsoleTests(unittest.TestCase):
    def test_ssh_and_result_transfer_share_only_a_short_lived_private_transport(self):
        import os
        from sim_console import SSH, SCP, SSH_CONTROL_PATH
        self.assertEqual(SSH_CONTROL_PATH, f'/run/user/{os.getuid()}/umi-simulation-console-%C')
        for command in (SSH, SCP):
            self.assertIn(f'ControlPath={SSH_CONTROL_PATH}', command)
            self.assertIn('ControlMaster=auto', command)
            self.assertIn('ControlPersist=30', command)
            self.assertIn('BatchMode=yes', command)
            self.assertNotIn('StrictHostKeyChecking=no', command)

    def test_only_lan_5090_is_listed_and_remote_start_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "backend_status", return_value={}) as status:
                backends = runner.inference_backends()
                self.assertEqual(len(backends), 10)
                self.assertTrue(all(INFERENCE_BACKENDS[key]["host"] == "squirrel_5090" for key in backends))
                self.assertEqual(status.call_count, 10)
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
                             ["pi05_10000_rtx5090"])
            self.assertTrue(policies["pi05-cup-clean-20000"]["ready"])
            self.assertEqual(policies["pi05-cup-clean-20000"]["supported_backends"],
                             ["pi05_20000_rtx5090"])
            self.assertTrue(policies["openwam-cup-clean-10000"]["ready"])
            self.assertEqual(policies["openwam-cup-clean-10000"]["supported_backends"], ["openwam_10000_rtx5090"])
            self.assertTrue(policies["openwam-cup-clean-5000"]["ready"])
            self.assertEqual(policies["openwam-cup-clean-5000"]["supported_backends"], ["openwam_5000_rtx5090"])
            self.assertTrue(policies["lingbot-cup-clean-5000"]["ready"])
            self.assertEqual(policies["lingbot-cup-clean-5000"]["supported_backends"], ["lingbot_5000_rtx5090"])
            baseline = ["pi05-cup-clean-10000", "pi05-cup-clean-20000", "pi05-cup-clean-30000",
                        "lingbot-cup-clean-5000", "lingbot-cup-clean-10000",
                        "openwam-cup-clean-5000", "openwam-cup-clean-10000"]
            for key in baseline:
                self.assertTrue(policies[key]["ready"])
                self.assertEqual(policies[key]["simulator_profile"], "tuned_online_v1")
                self.assertTrue(all(INFERENCE_BACKENDS[b]["host"] == "squirrel_5090"
                                    for b in policies[key]["supported_backends"]))
            self.assertNotEqual(policies["pi05-cup-clean-30000-assisted"].get("simulator_profile"),
                                "tuned_online_v1")

    def test_input_limits(self):
        for value in (True, -1, 10001, "0", 0.5):
            with self.assertRaises(ValueError):
                _int_field(value, "setup_index", 0, 10000)
        self.assertEqual(_int_field(10000, "setup_index", 0, 10000), 10000)

    def test_tuned_online_launcher_routes_models_and_task_objective(self):
        import shlex
        for key, backend in [("pi05-cup-clean-30000", "rtx5090"),
                             ("lingbot-cup-clean-10000", "lingbot_10000_rtx5090"),
                             ("lingbot-vla2-official-pretrained", "lingbot_official_rtx5090"),
                             ("openwam-cup-clean-5000", "openwam_5000_rtx5090")]:
            for objective in ("plate", "plate_return"):
                with self.subTest(policy=key, objective=objective), tempfile.TemporaryDirectory() as temporary:
                    runner = SimulationRunner(Path(temporary))
                    policy = next(p for p in runner.policies() if p['id'] == key)
                    run_id = '0123456789ab'
                    directory = Path(temporary)/run_id
                    directory.mkdir()
                    metadata = {'id':run_id, 'setup_index':0, 'seed':42, 'steps':10,
                                'camera':'head', 'task_objective':objective, 'inference_backend':backend,
                                'remote_dir':str(REMOTE_ROOT/'runs'/f'console_{run_id}'),
                                'status':'starting'}
                    (directory/'metadata.json').write_text(json.dumps(metadata))
                    with patch('sim_console.subprocess.run', return_value=subprocess.CompletedProcess([], 1)) as remote, \
                         patch.object(runner, '_service_state', return_value={'active':False}), \
                         patch.object(runner, '_stop_backend_locked'):
                        runner._execute(metadata, policy)
                    launch = shlex.split(remote.call_args_list[0].args[0][-1])
                    self.assertEqual(launch[-1], objective)
                    self.assertEqual(launch[-2], '10')
                    self.assertIn(f"UMI_MODEL_UNIT={INFERENCE_BACKENDS[backend]['unit']}", launch)
                    self.assertTrue(any(item.endswith('/run_tuned_online.sh') for item in launch))
                    self.assertNotIn('YUBI_CANONICAL_HAND_SIDES=1', launch)
                    self.assertEqual(metadata['simulator_profile'], 'tuned_online_v1')
                    self.assertFalse(metadata['online_control']['oracle_action_feedback'])

    def test_recovery_adopts_existing_runtime_without_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            run_id = '0123456789ab'
            directory = Path(temporary)/run_id
            directory.mkdir()
            metadata = {'id':run_id,'status':'running','policy':'lingbot-cup-clean-10000',
                        'simulator_profile':'tuned_online_v1',
                        'remote_dir':str(REMOTE_ROOT/'runs'/f'console_{run_id}')}
            (directory/'metadata.json').write_text(json.dumps(metadata))
            with patch.object(runner,'_runtime_snapshot',return_value={'MainPID':'123','ActiveState':'active'}), \
                 patch('sim_console.threading.Thread') as thread:
                self.assertEqual(runner.recover_active_run(), run_id)
                self.assertEqual(runner.active_id,run_id)
                self.assertTrue(thread.call_args.kwargs['args'][-1])
                self.assertTrue(runner._metadata(run_id)['console_tracking_recovered'])

    def test_recovery_snapshot_rejects_wrong_pid_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner,'_remote',return_value=subprocess.CompletedProcess([],0,
                  stdout='MainPID=123\nActiveState=active\nCommandLine=other-user-python\n')):
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    runner._runtime_snapshot('0123456789ab')

    def test_left_return_diagnostic_routes_explicitly(self):
        import shlex
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            policy = next(p for p in runner.policies() if p['id'] == 'pi05-cup-clean-30000')
            run_id = '0123456789ab'
            directory = Path(temporary)/run_id
            directory.mkdir()
            metadata = {'id':run_id, 'setup_index':0, 'seed':42, 'steps':600,
                        'camera':'head', 'task_objective':'plate_return', 'inference_backend':'rtx5090',
                        'remote_dir':str(REMOTE_ROOT/'runs'/f'console_{run_id}'),
                        'status':'starting', 'left_return_diagnostic':True}
            (directory/'metadata.json').write_text(json.dumps(metadata))
            with patch('sim_console.subprocess.run', return_value=subprocess.CompletedProcess([], 1)) as remote, \
                 patch.object(runner, '_service_state', return_value={'active':False}), \
                 patch.object(runner, '_stop_backend_locked'):
                runner._execute(metadata, policy)
            launch = shlex.split(remote.call_args_list[0].args[0][-1])
            self.assertEqual(launch[-3:], ['600','plate_return','left-return'])
            self.assertEqual(metadata['evaluation_class'], 'left_phase_reset_intervention_not_full_task_evaluation')
            self.assertTrue(metadata['diagnostic_interventions']['right_arm_commands_held'])

    def test_left_return_diagnostic_rejects_wrong_model_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, '_start_backend_locked') as start:
                with self.assertRaisesRegex(ValueError, 'left diagnostic requires'):
                    runner.start({'policy':'lingbot-cup-clean-10000','inference_backend':'lingbot_10000_rtx5090',
                                  'setup_index':0,'seed':42,'camera':'head','steps':600,
                                  'task_objective':'plate_return','left_return_diagnostic':True})
                start.assert_not_called()

    def test_extra_closure_routes_explicit_parameter_and_assistance_label(self):
        import shlex
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            policy = next(p for p in runner.policies() if p['id'] == 'pi05-cup-clean-30000')
            run_id = '0123456789ab'
            directory = Path(temporary)/run_id
            directory.mkdir()
            metadata = {'id':run_id, 'setup_index':0, 'seed':42, 'steps':600,
                        'camera':'head', 'task_objective':'plate_return', 'inference_backend':'rtx5090',
                        'remote_dir':str(REMOTE_ROOT/'runs'/f'console_{run_id}'),
                        'status':'starting', 'left_return_diagnostic':True, 'left_extra_closure_fraction':.05}
            (directory/'metadata.json').write_text(json.dumps(metadata))
            with patch('sim_console.subprocess.run', return_value=subprocess.CompletedProcess([], 1)) as remote, \
                 patch.object(runner, '_service_state', return_value={'active':False}), \
                 patch.object(runner, '_stop_backend_locked'):
                runner._execute(metadata, policy)
            launch = shlex.split(remote.call_args_list[0].args[0][-1])
            self.assertEqual(launch[-4:], ['600','plate_return','left-return','0.050000'])
            self.assertIn('额外闭合对照', metadata['policy_label'])
            self.assertFalse(metadata['diagnostic_interventions']['left_extra_closure']['pure_model_action'])

    def test_extra_closure_requires_explicit_diagnostic_and_safe_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            request = {'policy':'pi05-cup-clean-30000','inference_backend':'rtx5090',
                       'setup_index':0,'seed':42,'camera':'head','steps':600,'task_objective':'plate_return'}
            with patch.object(runner, '_start_backend_locked') as start:
                for extra, diagnostic in ((.05,False),(.051,True),(-.01,True),(float('nan'),True),(True,True)):
                    with self.assertRaisesRegex(ValueError, 'extra closure requires'):
                        runner.start({**request, 'left_extra_closure_fraction':extra, 'left_return_diagnostic':diagnostic})
                start.assert_not_called()

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

    def test_lingbot_refuses_pi05_inference_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = SimulationRunner(Path(temporary))
            with patch.object(runner, "gpu_status") as gpu:
                with self.assertRaisesRegex(ValueError, "not validated for this policy"):
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
                    "policy": "lingbot-cup-clean-10000", "setup_index": 0,
                    "seed": 42, "camera": "overview", "inference_backend": "lingbot_10000_rtx5090",
                    "task_objective": "plate_return", "run_until_success": True,
                })
            start_backend.assert_called_once_with("lingbot_10000_rtx5090")
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
