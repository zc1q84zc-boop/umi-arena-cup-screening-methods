import json
import hashlib
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sim_console import SimulationRunner, REMOTE_ROOT, _require_verified_pvc_probe
import sim_console


class ContactProfileTests(unittest.TestCase):
    request = dict(policy='pi05-cup-clean-30000', inference_backend='rtx5090',
                   setup_index=0, seed=42, camera='head', steps=600, task_objective='plate_return')

    def test_invalid_profile_rejected_before_model_start(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = SimulationRunner(Path(folder))
            with patch.object(runner, '_start_backend_locked') as start:
                for value in ('bad; rm', None, [], {'id': 'baseline'}):
                    with self.assertRaises(ValueError):
                        runner.start({**self.request, 'contact_profile': value})
                start.assert_not_called()

    def test_no_mixed_action_assistance_or_unbounded_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = SimulationRunner(Path(folder))
            with patch.object(runner, '_start_backend_locked') as start:
                for extra in ({'left_return_diagnostic': True}, {'steps': 601}, {'run_until_success': True}):
                    with self.assertRaisesRegex(ValueError, 'contact trial requires'):
                        runner.start({**self.request, 'contact_profile': 'official_fingertip_friction', **extra})
                start.assert_not_called()

    def test_pvc_cannot_start_before_physical_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = SimulationRunner(Path(folder)/'runs')
            policies=runner.policies()
            with patch('sim_console.ROOT',Path(folder)), patch.object(runner,'policies',return_value=policies), \
                 patch.object(runner,'_start_backend_locked') as start:
                with self.assertRaisesRegex(ValueError,'physical validation pending'):
                    runner.start({**self.request,'contact_profile':'pvc_elastic_shell_v1'})
                start.assert_not_called()

    def test_verified_precision_default_keeps_until_success(self):
        precision = sim_console._PVC_NUMERICS['PRECISION_ID']
        with tempfile.TemporaryDirectory() as folder:
            runner = SimulationRunner(Path(folder))
            with patch('sim_console.DEFAULT_CONTACT_PROFILE', precision), \
                 patch('sim_console._require_verified_pvc_probe') as probe, \
                 patch.object(runner, '_start_backend_locked'), patch('threading.Thread.start'):
                request = {**self.request, 'run_until_success': True}
                request.pop('steps')
                run = runner.start(request)
            self.assertEqual(run['contact_profile'], precision)
            self.assertTrue(run['run_until_success'])
            self.assertIsNone(run['steps'])
            probe.assert_called_once_with(precision)

    def test_explicit_rigid_comparison_overrides_precision_default(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = SimulationRunner(Path(folder))
            with patch('sim_console.DEFAULT_CONTACT_PROFILE', sim_console._PVC_NUMERICS['PRECISION_ID']), \
                 patch('sim_console._require_verified_pvc_probe') as probe, \
                 patch.object(runner, '_start_backend_locked'), patch('threading.Thread.start'):
                run = runner.start({**self.request, 'contact_profile': 'baseline'})
            self.assertEqual(run['contact_profile'], 'baseline')
            probe.assert_not_called()

    def test_continuous_precision_still_requires_physical_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = SimulationRunner(Path(folder))
            with patch('sim_console._require_verified_pvc_probe', side_effect=ValueError('physical validation pending')), \
                 patch.object(runner, '_start_backend_locked') as start:
                with self.assertRaisesRegex(ValueError, 'physical validation pending'):
                    runner.start({**self.request, 'run_until_success': True,
                                  'contact_profile': sim_console._PVC_NUMERICS['PRECISION_ID']})
            start.assert_not_called()

    def test_verified_pvc_report_is_bound_to_current_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/'simulator_profiles/tuned_v1/yubi_isaac_sim_env'
            package.mkdir(parents=True)
            for name in ('pvc_shell.py','pvc_shell_probe.py'):(package/name).write_text(name)
            report=dict(status='completed',purpose='physical_platen_compression_not_model_grasp',
                prescribed_vertex_animation=False,profile={'id':'pvc_elastic_shell_v1','measured':False},
                unloaded_shape_preserved=True,physical_deformation_observed=True,recovered_after_release=True,
                platen_motion_verified=True,
                sample_count=960,video_frames=240,
                shell_sha256=hashlib.sha256((package/'pvc_shell.py').read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256((package/'pvc_shell_probe.py').read_bytes()).hexdigest())
            validation=root/'sim_validation/pvc_shell_verified_probe.json';validation.parent.mkdir()
            validation.write_text(json.dumps(report))
            with patch('sim_console.ROOT',root):
                self.assertEqual(_require_verified_pvc_probe(),report)
                report['platen_motion_verified']=False
                validation.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'physical validation pending'):_require_verified_pvc_probe()
                report['platen_motion_verified']=True
                validation.write_text(json.dumps(report))
                (package/'pvc_shell.py').write_text('new untested version')
                with self.assertRaisesRegex(ValueError,'physical validation pending'):_require_verified_pvc_probe()

    def test_explicit_routing_and_physics_not_action_label(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = SimulationRunner(Path(folder))
            run_id = '0123456789ab'
            directory = Path(folder) / run_id
            directory.mkdir()
            metadata = dict(self.request, id=run_id, status='starting',
                            remote_dir=str(REMOTE_ROOT/'runs'/f'console_{run_id}'),
                            contact_profile='official_fingertip_friction')
            (directory/'metadata.json').write_text(json.dumps(metadata))
            policy = next(p for p in runner.policies() if p['id'] == self.request['policy'])
            with patch('sim_console.subprocess.run', return_value=subprocess.CompletedProcess([], 1)) as run, \
                 patch.object(runner, '_service_state', return_value={'active':False}), \
                 patch.object(runner, '_stop_backend_locked'):
                runner._execute(metadata, policy)
            self.assertEqual(shlex.split(run.call_args_list[0].args[0][-1])[-5:],
                             ['600', 'plate_return', 'baseline', '0', 'official_fingertip_friction'])
            self.assertFalse(metadata['physics_contact_trial']['model_actions_changed'])
            self.assertFalse(metadata['physics_contact_trial']['cup_deformable'])

    def test_layer_only_authors_three_fingertip_properties(self):
        path = Path(__file__).parent/'simulator_profiles/tuned_v1/yubi_isaac_sim_env/scenes/dual_franka_yubi_official_fingertip_friction_trial.usda'
        text = path.read_text()
        properties = [line.strip() for line in text.splitlines() if line.strip().startswith(('float ', 'uniform token '))]
        self.assertEqual(properties, ['float physics:staticFriction = 0.8',
            'float physics:dynamicFriction = 0.8', 'uniform token physxMaterial:frictionCombineMode = "max"'])
        self.assertNotIn('over "Cup"', text)

    def test_ssh_transport_loss_waits_for_durable_runtime_before_model_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = SimulationRunner(Path(folder))
            run_id = '0123456789ab'
            directory = Path(folder)/run_id
            directory.mkdir()
            metadata = dict(self.request,id=run_id,status='starting',
                remote_dir=str(REMOTE_ROOT/'runs'/f'console_{run_id}'),contact_profile='baseline')
            (directory/'metadata.json').write_text(json.dumps(metadata))
            policy = next(p for p in runner.policies() if p['id']==self.request['policy'])
            order = []
            with patch('sim_console.subprocess.run',return_value=subprocess.CompletedProcess([],255)), \
                 patch.object(runner,'_wait_existing_runtime',side_effect=lambda _: (order.append('wait'),subprocess.CompletedProcess([],0))[1]) as wait, \
                 patch.object(runner,'_service_state',return_value={'active':False}), \
                 patch.object(runner,'_stop_backend_locked',side_effect=lambda *a:order.append('cleanup')):
                runner._execute(metadata,policy)
            wait.assert_called_once_with(run_id)
            self.assertEqual(order,['wait','cleanup'])
            self.assertTrue(metadata['ssh_transport_recovered'])


if __name__ == '__main__':
    unittest.main()
