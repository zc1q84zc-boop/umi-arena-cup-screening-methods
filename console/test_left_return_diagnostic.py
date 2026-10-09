import copy
import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).parent/'simulator_profiles/tuned_v1/yubi_isaac_sim_env/left_return_diagnostic.py'
spec = importlib.util.spec_from_file_location('left_diagnostic_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def observation():
    robot = {'joint_names': [f'panda_joint{i}' for i in range(1,8)],
             'joint_positions': [i*.1 for i in range(7)], 'gripper_open_fraction': .96,
             'tool_pose': {'position_m':[.041,.2548,.82], 'quaternion_wxyz':[1,0,0,0]}}
    return {'scenario': {'diagnostic_return_origin_m':[-.0165,-.0226,.75]},
            'robots': {'left':copy.deepcopy(robot),'right':copy.deepcopy(robot)},
            'objects': {'cup': {'position_m':[.041,.2548,.753], 'quaternion_wxyz':[1,0,0,0],
                        'linear_velocity_m_s':[0,0,0], 'angular_velocity_rad_s':[0,0,0]},
                        'plate': {'position_m':[.041,.2548,.75]}}}


class LeftReturnTests(unittest.TestCase):
    def test_extra_closure_is_left_only_bounded_and_preserves_release_and_poses(self):
        diagnostic = module.LeftReturnDiagnostic(observation(), extra_closure_fraction=.05)
        chunk = {'waypoints':[{'left':{'position_m':[1,2,3], 'quaternion_wxyz':[1,0,0,0],
                                      'gripper_open_fraction': f},'right':{}} for f in (.5,.6,.9)]}
        original = copy.deepcopy(chunk)
        filtered = diagnostic.filter_chunk(chunk)
        self.assertEqual(chunk, original)
        for source, applied, expected in zip(chunk['waypoints'], filtered['waypoints'], (.45,.575,.9)):
            self.assertEqual(source['left']['position_m'], applied['left']['position_m'])
            self.assertEqual(source['left']['quaternion_wxyz'], applied['left']['quaternion_wxyz'])
            self.assertAlmostEqual(applied['left']['gripper_open_fraction'], expected)
            self.assertEqual(applied['right'], diagnostic.right_pose)
        self.assertEqual(module.closure_target(.65,.05), .65)
        self.assertEqual(module.closure_target(0.,.05), 0.)
        self.assertAlmostEqual(module.closure_target(.65-1e-8,.05), .65, places=7)
        self.assertTrue(diagnostic.update(observation())['closure_bias_is_diagnostic_assistance'])

    def test_default_unchanged_and_invalid_extra_closure_rejected(self):
        for model in (0.,.5,.6,.65,.99,1.):
            self.assertEqual(module.closure_target(model,0.), model)
        for extra in (-.01,.051,float('nan'),float('inf'),True,'0.05'):
            with self.assertRaises(ValueError):
                module.LeftReturnDiagnostic(observation(), extra_closure_fraction=extra)
        for model in (-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): module.closure_target(model,.05)

    def test_closure_audit_rejects_hidden_force_or_release_changes(self):
        from verify_tuned_online import verify_closure_trace
        diagnostic = module.LeftReturnDiagnostic(observation(), extra_closure_fraction=.05)
        chunk = {'waypoints':[{'left':{'gripper_open_fraction': f},'right':{}} for f in (.5,.6,.9)]}
        diagnostic.filter_chunk(chunk)
        trace = [{'action_index':i, 'closure_command':copy.deepcopy(a),
                  'closure_bias_is_diagnostic_assistance':True}
                 for i,a in enumerate(diagnostic.closure_audit)]
        adapter = [{'substep_waypoints':chunk['waypoints']}]
        verify_closure_trace(trace,adapter,.05)
        trace[0]['closure_command']['force_limit_changed'] = True
        with self.assertRaises(AssertionError): verify_closure_trace(trace,adapter,.05)
        trace[0]['closure_command']['force_limit_changed'] = False
        trace[2]['closure_command']['command_open_fraction'] = .85
        with self.assertRaises(AssertionError): verify_closure_trace(trace,adapter,.05)

    def test_initial_plate_cannot_count_as_model_success(self):
        obs = observation()
        diagnostic = module.LeftReturnDiagnostic(obs)
        for _ in range(100):
            result = diagnostic.update(obs, plate_success=True)
            self.assertFalse(result['diagnostic_success'])
            self.assertFalse(result['full_task_success'])
        self.assertTrue(result['plate_placed_by_reset'])

    def test_right_hold_preserves_model_left_action_and_original_chunk(self):
        obs = observation()
        diagnostic = module.LeftReturnDiagnostic(obs)
        chunk = {'waypoints':[{'left':{'position_m':[1,2,3]},'right':{'position_m':[4,5,6]}}]*3,
                 'execute_steps':3}
        original = copy.deepcopy(chunk)
        filtered = diagnostic.filter_chunk(chunk)
        self.assertEqual(chunk, original)
        self.assertEqual(filtered['waypoints'][0]['left'], chunk['waypoints'][0]['left'])
        self.assertEqual(filtered['waypoints'][0]['right'], diagnostic.right_pose)
        action = diagnostic.filter_action({'left':{'gripper_open_fraction':.3},'right':{}})
        self.assertEqual(action['right']['arm_joint_targets_rad'], obs['robots']['right']['joint_positions'])

    def test_return_requires_stable_lift_and_release(self):
        obs = observation()
        diagnostic = module.LeftReturnDiagnostic(obs)
        obs['objects']['cup']['position_m'] = [-.0165,-.0226,.75]
        for _ in range(20):
            self.assertFalse(diagnostic.update(obs)['diagnostic_success'])
        obs['objects']['cup']['position_m'] = [.041,.2548,.804]
        obs['robots']['left']['gripper_open_fraction'] = .4
        for _ in range(9):
            self.assertFalse(diagnostic.update(obs)['stable_lift_confirmed'])
        self.assertTrue(diagnostic.update(obs)['stable_lift_confirmed'])
        obs['objects']['cup']['position_m'] = [-.0165,-.0226,.75]
        for _ in range(20):
            self.assertFalse(diagnostic.update(obs)['diagnostic_success'])
        obs['robots']['left']['gripper_open_fraction'] = .9
        for _ in range(14):
            self.assertFalse(diagnostic.update(obs)['diagnostic_success'])
        result = diagnostic.update(obs)
        self.assertTrue(result['diagnostic_success'])
        self.assertFalse(result['full_task_success'])

    def test_invalid_reset_rejected(self):
        obs = observation()
        obs['objects']['cup']['position_m'][2] = .75
        with self.assertRaises(ValueError): module.LeftReturnDiagnostic(obs)

    def test_verifier_accepts_only_governed_reset_to_hold_transition(self):
        from verify_tuned_online import verify_right_hold_corridor
        reset = [0.]*7
        hold = [.0006]*7
        rows = [{'q':[[0.]*9, [value]*7+[.6,-.6]]} for value in [0.,.0003,.0006]]
        verify_right_hold_corridor(reset, hold, rows)
        rows[1]['q'][1][0] = .001
        with self.assertRaises(AssertionError): verify_right_hold_corridor(reset, hold, rows)
        rows[1]['q'][1][0] = -.001
        with self.assertRaises(AssertionError): verify_right_hold_corridor(reset, hold, rows)
        rows[1]['q'][1][0] = .0003
        rows[-1]['q'][1][0] = 0.
        with self.assertRaises(AssertionError): verify_right_hold_corridor(reset, hold, rows)
        obs = observation()
        obs['scenario']['diagnostic_return_origin_m'] = [.041,.2548,.75]
        with self.assertRaises(ValueError): module.LeftReturnDiagnostic(obs)


if __name__ == '__main__': unittest.main()
