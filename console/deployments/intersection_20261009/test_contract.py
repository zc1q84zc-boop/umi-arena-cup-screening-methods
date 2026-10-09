"""Check the changed temporal contract and current-observation HTTP boundary."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),os.environ.get('INTERSECTION_REFERENCE_ADAPTER_DIR',str(HERE.parents[1]))]
os.environ.update(INTERSECTION_MODEL_ID='pi05-cup-intersection-30000',
    INTERSECTION_ONLINE_URL='http://127.0.0.1:18861/infer',SIM_ADAPTER_AUDIT_DIR=tempfile.mkdtemp(),
    UMI_ONLINE_CALIBRATION='')
import intersection_adapter as adapter
entry=HERE/('intersection_server.py' if (HERE/'intersection_server.py').exists() else 'server.py')
spec=importlib.util.spec_from_file_location('intersection_server_contract',entry)
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.current={side:(np.zeros(3),np.array([1.,0,0,0]),.5) for side in ('left','right')}
        self.action=np.zeros((1,16));self.action[:,6]=self.action[:,13]=1;self.action[:,14:]=.2

    def test_future_100ms_action_is_integrated_once_in_current_body_frame(self):
        self.current['left']=(np.array([.1,.2,.3]),np.array([math.sqrt(.5),0,0,math.sqrt(.5)]),.5)
        self.action[0,0]=.06
        endpoint,diagnostic=adapter.waypoint10hz(self.current,self.action)
        np.testing.assert_allclose(endpoint['left']['position_m'],[.1,.26,.3],atol=1e-9)
        self.assertEqual(diagnostic['left']['future_rows_consumed'],1)
        self.assertAlmostEqual(diagnostic['left']['world_position_delta_m'],.06)

    def test_bounds_apply_to_100ms_rotation_and_translation(self):
        self.action[0,3:7]=[0,0,math.sin(.3),math.cos(.3)]
        adapter.waypoint10hz(self.current,self.action)
        self.action[0,0]=.081
        with self.assertRaisesRegex(ValueError,'100ms translation'):adapter.waypoint10hz(self.current,self.action)

    def test_http_prompt_images_and_identical_servo_targets(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def read(inner):
                return json.dumps({'model':adapter.MODEL,'episode':7,'step':0,'prompt':server.validate_instruction(prompt),
                    'prompt_source':server.PROMPT_SOURCE,'prompt_protocol':server.PROMPT_PROTOCOL,
                    'future_observation_used':False,'latency_ms':1,'action_timing':server.TIMING,
                    'actions':self.action.tolist()}).encode()
        prompt='Pick up the cup with your right hand and set it on the plate'
        self.action[0,0]=.009
        pixels=np.zeros((480,640,3),np.uint8);pixels[::2]=255
        obs={'task_stage':'place_on_plate','images':{s+'_wrist':pixels for s in ('left','right')},
            'robots':{s:{'tool_pose':{'position_m':[0,0,0],'quaternion_wxyz':[1,0,0,0]},
                         'gripper_open_fraction':.5} for s in ('left','right')}}
        with patch.object(adapter,'urlopen',return_value=Response()) as client:
            result=adapter.predict(obs,0,7)
        payload=json.loads(client.call_args.args[0].data)
        official=server.observation(payload)
        self.assertEqual(set(official),{'observation.image.left','observation.image.right',
            'observation.pose.left_hand_root_to_right_hand_root.absolute','observation.joint_states','prompt'})
        self.assertEqual(payload['prompt'],prompt)
        self.assertEqual(result['execute_steps'],3)
        self.assertTrue(all(item==result['waypoints'][0] for item in result['waypoints']))
        self.assertAlmostEqual(result['waypoints'][0]['left']['position_m'][0],.009)
        payload['prompt']='combined task'
        with self.assertRaises(ValueError):server.observation(payload)


if __name__=='__main__':unittest.main()
