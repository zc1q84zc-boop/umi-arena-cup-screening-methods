import importlib
import unittest
from unittest.mock import patch
from pathlib import Path

import native_4090_console as native
import native_4090_transport as transport


class NativeDeploymentTest(unittest.TestCase):
    def test_wrong_host_is_rejected(self):
        with patch('socket.gethostname', return_value='not-4090'):
            with self.assertRaises(RuntimeError):
                native.configure()

    def test_relocation_preserves_policy_ids_and_action_semantics(self):
        obj = {'id': 'pi05_10000_rtx5090', 'waypoints': [1., .2],
               'path': native.OLD_MODELS + '/pi05/10000/inference_export'}
        out = native.relocate(obj)
        self.assertEqual(out['id'], obj['id'])
        self.assertEqual(out['waypoints'], obj['waypoints'])
        self.assertEqual(out['path'], str(native.MODELS / 'pi05/10000/inference_export'))

    def test_transport_has_no_foreign_host_execution(self):
        with patch('subprocess.call') as command:
            self.assertEqual(transport.execute(['command', 'squirrel_5090', 'true']), 2)
            command.assert_not_called()

    def test_transport_cannot_copy_sensitive_files_or_escape_runs(self):
        for source in ('squirrel_4090_2:/etc/passwd', 'squirrel_4090_2:/home/claude/dual-franka-yubi-isaac-sim-deploy/runs/../secret'):
            self.assertEqual(transport.execute(['copy', source, str(transport.CONSOLE/'sim_runs/test')]), 2)

    def test_command_native_and_retains_return_code(self):
        with patch('subprocess.call', return_value=3) as command:
            self.assertEqual(transport.execute(['command', transport.HOST, 'false']), 3)
            command.assert_called_once_with(['bash', '-lc', 'false'])

    def test_unvalidated_lingbot_is_not_advertised_as_ready(self):
        with patch.object(native, 'validated_lingbot', return_value=False):
            status = native.deployment_backend_status('lingbot_5000_rtx5090',
                                                    {'active': False, 'deploy_ready': True})
        self.assertFalse(status['deploy_ready'])
        self.assertTrue(status['weights_installed'])
        self.assertEqual(status['migration_validation'], 'adapter_pending')

    def test_other_model_readiness_is_preserved(self):
        status = {'active': False, 'deploy_ready': True}
        self.assertEqual(native.deployment_backend_status('rtx5090', status), status)

    def test_intersection_missing_target_probe_is_fail_closed(self):
        with patch.object(Path,'read_text',side_effect=FileNotFoundError):
            for key in native.INTERSECTION_KEYS:
                self.assertFalse(native.validated_intersection(key))
                state=native.deployment_backend_status(key,{'deploy_ready':True,'ready':True})
                self.assertFalse(state['deploy_ready'])
                self.assertFalse(state['ready'])

    def test_intersection_relocates_to_its_separate_verified_root(self):
        old=native.OLD_INTERSECTION+'/pi05/30000'
        self.assertEqual(native.relocate(old),str(native.INTERSECTION/'pi05/30000'))

    def test_openwam_readiness_checks_cpu_offload_code(self):
        import hashlib
        import json
        digest = hashlib.sha256(b'verified').hexdigest()
        model = 'openwam-cup-intersection-fullpass-5069'
        record = {'status':'ok', 'model':model, 'hardware':'RTX4090 GPU1',
                  'verified_requests':2, 'model_service_cleanup':'stopped',
                  **{key:digest for key in ('server_sha256','adapter_sha256','entry_sha256',
                      'launcher_sha256','simulation_launcher_sha256','cpu_offload_wrapper_sha256',
                      'cpu_text_helper_sha256')}}
        integrity = {'model':model, 'all_sha256_verified':True,
            'manifest_sha256':'f4b34ab3fa3236a5a1b0cec8b42005b8faa9e902721ff17a718051790a4e0662'}
        def read(path):
            return json.dumps(integrity if path.name.endswith('_checkpoint_integrity.json') else record)
        with patch.object(Path,'read_text',read), patch.object(Path,'read_bytes',return_value=b'verified'):
            self.assertTrue(native.validated_intersection('openwam_intersection_rtx5090'))
            record['cpu_offload_wrapper_sha256'] = 'changed'
            self.assertFalse(native.validated_intersection('openwam_intersection_rtx5090'))
            record['cpu_offload_wrapper_sha256'] = digest
            record['cpu_text_helper_sha256'] = 'changed'
            self.assertFalse(native.validated_intersection('openwam_intersection_rtx5090'))

    def test_validation_cannot_enable_foreign_or_missing_models(self):
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            for key in ('lingbot_official_rtx5090', 'lingbot_5000_rtx5090', 'a100'):
                self.assertFalse(native.validated_lingbot(key))

    def test_changed_normalization_or_code_invalidates_approval(self):
        import hashlib
        import json
        digest = hashlib.sha256(b'verified').hexdigest()
        record = {'status': 'causal_server_probe_passed', 'hostname': 'benyun-workstation',
                  'checkpoint': '5000', 'model_id': 'lingbot-cup-clean-5000',
                  'model_service_cleanup': 'stopped',
                  'saved_sim_steps': [{'step': i, 'action_rows': 3} for i in range(3)],
                  **{key: digest for key in ('server_sha256', 'adapter_sha256', 'launcher_sha256', 'normalization_sha256')}}
        with patch.object(Path, 'read_text', side_effect=lambda: json.dumps(record)), \
             patch.object(Path, 'read_bytes', return_value=b'verified'):
            self.assertTrue(native.validated_lingbot('lingbot_5000_rtx5090'))
            record['normalization_sha256'] = 'unverified'
            self.assertFalse(native.validated_lingbot('lingbot_5000_rtx5090'))

    def test_hash_verified_lingbot_preserves_service_state(self):
        status = {'active': False, 'deploy_ready': True}
        with patch.object(native, 'validated_lingbot', return_value=True):
            self.assertEqual(native.deployment_backend_status('lingbot_5000_rtx5090', status), status)

    def test_official_requires_pinned_provenance_and_wrapper_hash(self):
        import hashlib
        import json
        digest = hashlib.sha256(b'verified').hexdigest()
        record = {'status': 'causal_server_probe_passed', 'hostname': 'benyun-workstation',
                  'checkpoint': 'official', 'model_id': 'lingbot-vla2-official-pretrained',
                  'model_service_cleanup': 'stopped',
                  'saved_sim_steps': [{'step': i, 'action_rows': 3} for i in range(3)],
                  'model_provenance': {'repo_id': 'robbyant/lingbot-vla-v2-6b',
                      'revision': '11c703bf6a5c1f45b3b69168482da11fdbba53d7',
                      'fine_tuned': False, 'all_files_sha256_verified': True},
                  **{key: digest for key in ('server_sha256', 'adapter_sha256', 'launcher_sha256',
                    'normalization_sha256', 'official_adapter_sha256', 'official_loader_sha256',
                    'official_manifest_sha256', 'robot_config_sha256')}}
        with patch.object(Path, 'read_text', side_effect=lambda: json.dumps(record)), \
             patch.object(Path, 'read_bytes', return_value=b'verified'):
            self.assertTrue(native.validated_lingbot('lingbot_official_rtx5090'))
            record['model_provenance']['fine_tuned'] = True
            self.assertFalse(native.validated_lingbot('lingbot_official_rtx5090'))
            record['model_provenance']['fine_tuned'] = False
            record['official_adapter_sha256'] = 'changed'
            self.assertFalse(native.validated_lingbot('lingbot_official_rtx5090'))

    def test_another_consoles_service_cannot_be_taken_over(self):
        from types import SimpleNamespace
        runner = SimpleNamespace(_service_state=lambda key: {'active': True, 'pid': 123})
        with patch.dict(native.app.INFERENCE_BACKENDS, {'rtx5090': {}}, clear=True):
            with self.assertRaises(ValueError):
                native.check_service_ownership(runner)
            runner._4090_owned_model_pids = {'rtx5090': 123}
            native.check_service_ownership(runner)
            runner._4090_owned_model_pids = {'rtx5090': 124}
            with self.assertRaises(ValueError):
                native.check_service_ownership(runner)


if __name__ == '__main__':
    unittest.main()
