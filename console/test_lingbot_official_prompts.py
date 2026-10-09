import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from deploy_servers.official_cup_prompts import (
    RIGHT_PLACE, LEFT_RETURN, PROMPT_PROTOCOL, PROMPT_SOURCE, cup_instruction, validate_instruction,
)
from deploy_servers import lingbot_sim_server as server
import lingbot_isaac_online_adapter as adapter

sys.path.insert(0, str(Path(__file__).parent / 'simulator_profiles/tuned_v1'))
from yubi_isaac_sim_env.two_stage_task import CupPlateReturnEvaluator


def response(prompt, step=0):
    action = np.tile([0., 0., 0., 0., 0., 0., 1.] * 2 + [.4, .5], (3, 1))
    return {'episode': 0, 'step': step, 'actions': action.tolist(), 'latency_ms': 1.,
            'prompt': prompt, 'prompt_source': PROMPT_SOURCE, 'prompt_protocol': PROMPT_PROTOCOL,
            'inference_mode': 'native_lingbot_vla_v2_chunk',
            'pose_mapping': 'first_live_left_tool_to_training_episode_61164_frame0_SE3_provisional',
            'action_timing': {'pose_rows': [1, 2, 3], 'gripper_rows': [0, 1, 2]}}


class OfficialPromptsTest(unittest.TestCase):
    def test_importlib_loading_works_outside_the_console_checkout(self):
        # Match Isaac's dynamic policy load, without relying on the test
        # runner's cwd or the console's deploy_servers Python package.
        root = Path(__file__).parent
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'adapters'
            target.mkdir()
            for name in ('lingbot_isaac_online_adapter.py', 'pi05_isaac_online_adapter.py',
                         'online_calibration.py'):
                shutil.copy2(root/name, target/name)
            shutil.copy2(root/'deploy_servers/official_cup_prompts.py', target/'official_cup_prompts.py')
            env = dict(os.environ, UMI_ONLINE_CALIBRATION='')
            result = subprocess.run([sys.executable, '-I', '-c',
                'import importlib.util,sys; s=importlib.util.spec_from_file_location("cup_policy",sys.argv[1]); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert callable(m.predict)',
                str(target/'lingbot_isaac_online_adapter.py')], cwd=folder, env=env,
                capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_primitives_not_a_combined_or_custom_instruction(self):
        self.assertEqual(cup_instruction('place_on_plate'), RIGHT_PLACE)
        self.assertEqual(cup_instruction('return_to_origin'), LEFT_RETURN)
        for prompt in (None, '', RIGHT_PLACE + '. ' + LEFT_RETURN, RIGHT_PLACE.lower()):
            with self.assertRaises(ValueError): validate_instruction(prompt)
        for stage in (None, '', 'complete', 'unknown'):
            with self.assertRaises(ValueError): cup_instruction(stage)

    def test_evaluator_does_not_switch_while_right_hand_still_holds_cup(self):
        obs = {'objects': {'cup': {'position_m': [0., 0., .75],
                   'quaternion_wxyz': [1., 0., 0., 0.],
                   'linear_velocity_m_s': [0., 0., 0.], 'angular_velocity_rad_s': [0., 0., 0.]}},
               'robots': {'left': {'gripper_open_fraction': .9},
                          'right': {'gripper_open_fraction': .3}}}
        evaluator = CupPlateReturnEvaluator(obs)
        self.assertEqual(evaluator.update(obs, plate_success=True)['stage'], 'place_on_plate')
        obs['robots']['right']['gripper_open_fraction'] = .9
        self.assertEqual(evaluator.update(obs, plate_success=False)['stage'], 'place_on_plate')
        self.assertEqual(evaluator.update(obs, plate_success=True)['stage'], 'return_to_origin')
        obs['objects']['cup']['position_m'] = [.3, .2, .75]
        self.assertEqual(evaluator.update(obs, plate_success=False)['stage'], 'return_to_origin')

    def test_native_engine_forwards_both_prompts_without_resetting_between_primitives(self):
        class Model:
            def __init__(self): self.calls = []
            def infer(self, obs):
                self.calls.append(obs)
                if obs.get('reset'): return {}
                return {**{key: np.tile([0., 0., 0., 0., 0., 0., 1.], (5, 1))
                            for key in server.POSE_KEYS},
                        'action.joint_states': np.tile([.4, .5], (5, 1))}
        engine = server.Engine.__new__(server.Engine)
        engine.model, engine.lock = Model(), threading.Lock()
        engine.model_id = 'lingbot-vla2-official-pretrained'
        engine.provenance = {'fine_tuned': False}
        payload = {'left_jpeg': '', 'right_jpeg': '',
                   'left_pose_wxyz': [0., 0., 1., 1., 0., 0., 0.],
                   'right_pose_wxyz': [.3, 0., 1., 1., 0., 0., 0.],
                   'relative_pose_xyzw': [.3, 0., 0., 0., 0., 0., 1.],
                   'gripper_rad': [.4, .5], 'episode': 0}
        with patch.object(server, 'image', return_value=np.zeros((480, 640, 3), np.uint8)):
            for step, prompt in enumerate((RIGHT_PLACE, LEFT_RETURN)):
                result = engine.infer({**payload, 'step': step, 'prompt': prompt})
                self.assertEqual(result['prompt'], prompt)
                self.assertEqual(result['model_id'], engine.model_id)
                self.assertFalse(result['model_provenance']['fine_tuned'])
                self.assertEqual(result['prompt_protocol'], PROMPT_PROTOCOL)
                self.assertEqual(engine.model.calls[-1]['task'], prompt)
                self.assertEqual(engine.model.calls[-1]['prompt'], prompt)
                self.assertNotIn('objects', engine.model.calls[-1])
        self.assertEqual(sum(bool(c.get('reset')) for c in engine.model.calls), 1)
        self.assertEqual(engine.next_step, 2)

    def _call_adapter(self, stage, returned_prompt=None):
        prompt = cup_instruction(stage)
        received = []
        def urlopen(request, **kwargs):
            received.append(json.loads(request.data))
            return io.BytesIO(json.dumps(response(returned_prompt or prompt)).encode())
        obs = {'task_stage': stage, 'images': {'left_wrist': object(), 'right_wrist': object()},
               'robots': {s: {'tool_pose': {}} for s in ('left', 'right')},
               'objects': {'cup': {'position_m': [99, 99, 99]}}}
        original = copy.deepcopy(obs['objects'])
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(adapter, 'AUDIT_DIR', Path(folder)), \
             patch.object(adapter, 'CALIBRATION', None), \
             patch.object(adapter, 'current_robot', return_value=(np.zeros(3), np.array([1.,0.,0.,0.]), .5)), \
             patch.object(adapter, 'encode_wrist', return_value=b'jpeg'), \
             patch.object(adapter, '_source_from_fraction', side_effect=lambda f: f), \
             patch.object(adapter, '_waypoint', return_value=({'unchanged': True}, {})) as waypoint, \
             patch.object(adapter, 'urlopen', side_effect=urlopen), \
             patch.dict(os.environ, {'UMI_EXECUTE_30HZ': '0'}):
            result = adapter.predict(obs, 0, 0)
            row = json.loads((Path(folder)/'online_adapter.jsonl').read_text())
            self.assertEqual(result['waypoints'], [{'unchanged': True}])
            np.testing.assert_array_equal(waypoint.call_args.args[1], response(prompt)['actions'])
            self.assertEqual(row['prompt'], row['server_prompt'])
            self.assertEqual(row['task_stage'], stage)
        self.assertEqual(obs['objects'], original)
        self.assertEqual(received[0]['prompt'], prompt)
        self.assertEqual(set(received[0]), {'left_jpeg','right_jpeg','left_pose_wxyz','right_pose_wxyz',
                         'relative_pose_xyzw','gripper_rad','episode','step','prompt'})

    def test_adapter_forwards_official_prompt_and_native_actions_without_object_state(self):
        for stage in ('place_on_plate', 'return_to_origin'):
            with self.subTest(stage=stage): self._call_adapter(stage)

    def test_adapter_refuses_a_server_that_keeps_the_old_prompt(self):
        with self.assertRaisesRegex(ValueError, 'official primitive prompt/native inference'):
            self._call_adapter('return_to_origin', returned_prompt=RIGHT_PLACE)

    def test_artifact_audit_rejects_early_switch_and_wrong_server_prompt(self):
        from verify_tuned_online import verify_official_lingbot_prompts
        rows = [{**response(prompt, step), 'task_stage': stage}
                for step, (stage, prompt) in enumerate((('place_on_plate', RIGHT_PLACE),
                                                        ('return_to_origin', LEFT_RETURN)))]
        for row in rows: row['server_prompt'] = row['prompt']
        episode = {'transitions': [
            {'requested_task_stage': 'place_on_plate', 'task_stage': 'return_to_origin'},
            {'requested_task_stage': 'return_to_origin', 'task_stage': 'return_to_origin'}]}
        result = verify_official_lingbot_prompts(rows, episode)
        self.assertEqual(result['requests_per_primitive'], {'place_on_plate': 1, 'return_to_origin': 1})
        rows[1]['server_prompt'] = RIGHT_PLACE
        with self.assertRaises(AssertionError): verify_official_lingbot_prompts(rows, episode)
        rows[1]['server_prompt'] = LEFT_RETURN
        episode['transitions'][0]['task_stage'] = 'place_on_plate'
        with self.assertRaises(AssertionError): verify_official_lingbot_prompts(rows, episode)


if __name__ == '__main__': unittest.main()
