import tempfile
from pathlib import Path
from unittest import TestCase, mock

from sim_console import SimulationRunner, TUNED_REPLAY_ID


class TunedSimulatorProfileTests(TestCase):
    def request(self, **changes):
        return dict(policy=TUNED_REPLAY_ID, setup_index=0, seed=42, steps=215,
                    camera='overview', inference_backend='rtx5090', **changes)

    def test_tuned_profile_does_not_start_a_model_and_uses_private_replay_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = SimulationRunner(Path(folder))
            with mock.patch.object(runner, 'gpu_status', return_value={'available':True}), \
                 mock.patch.object(runner, '_start_backend_locked') as start, \
                 mock.patch('sim_console.threading.Thread.start'):
                result = runner.start(self.request())
                start.assert_not_called()
                self.assertIsNone(result['inference_backend'])
                self.assertEqual(result['steps'], 215)
                metadata = runner._metadata(result['id'])
                self.assertTrue(metadata['remote_dir'].startswith('/home/lrl/umi-tuned-replay-private/runs/console_'))

    def test_tuned_recipe_cannot_silently_change_demo_seed_or_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = SimulationRunner(Path(folder))
            for field, value in [('seed',99), ('steps',150), ('camera','head'), ('setup_index',1)]:
                request = self.request()
                request[field] = value
                with self.assertRaisesRegex(ValueError, 'requires scene 0'):
                    runner.start(request)
