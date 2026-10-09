import importlib.util
import unittest
from pathlib import Path
import numpy as np

PROFILE = Path(__file__).parent / 'simulator_profiles/tuned_v1'
def module(name):
    spec = importlib.util.spec_from_file_location('tuned_test_'+name, PROFILE/'yubi_isaac_sim_env'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
mapping = module('umi_gripper_mapping')
open_fraction_to_source_angle = mapping.open_fraction_to_source_angle
ContinuousTargets = module('continuous_targets').ContinuousTargets
from online_calibration import TunedOnlineV1, matrix


class TunedOnlineTests(unittest.TestCase):
    def test_aperture_roundtrip_and_zero_action(self):
        calibration = TunedOnlineV1(mapping)
        for q in np.linspace(.1, .6, 100):
            fraction = calibration.sim_gripper(q)
            self.assertAlmostEqual(calibration.source_gripper(fraction), q, places=5)
        for side in ('left', 'right'):
            p, q = calibration.tool_to_hand(side, [.1,.2,.9], [.5,.5,.5,.5])
            wp, wq = calibration.hand_to_world_tool(side, p, q)
            np.testing.assert_allclose(matrix(wp,wq), matrix([.1,.2,.9],[.5,.5,.5,.5]), atol=1e-12)
        self.assertFalse(calibration.audit()['oracle_action_feedback'])
        with self.assertRaises(ValueError): open_fraction_to_source_angle(float('nan'))

    def test_safety_bounds_and_mirrored_jaw(self):
        bounds = np.broadcast_to([-.1,.7], (2,8,2)).copy()
        governor = ContinuousTargets(np.zeros((2,8)), bounds, 1/60)
        previous_v = governor.v.copy()
        for i in range(1000):
            q = governor.step(np.full((2,8), .7 if i//40%2 else -.1))
            self.assertLessEqual(np.abs(governor.v).max(), .8+1e-9)
            self.assertLessEqual(np.abs(governor.v-previous_v).max()*60, 1.5+1e-8)
            self.assertTrue((q>=-.1).all() and (q<=.7).all())
            command = np.concatenate((q,-q[:,-1:]),axis=1)
            np.testing.assert_allclose(command[:,-2]+command[:,-1],0)
            previous_v = governor.v.copy()

    def test_online_wrapper_has_no_replay_policy(self):
        wrapper = (Path(__file__).parent/'deploy_servers/run_tuned_online_on_squirrel.sh').read_text()
        self.assertNotIn('umi_left_second_height_replay.py', wrapper)
        self.assertIn('unset UMI_REPLAY_PATH', wrapper)
        self.assertIn('--online-chunk-30hz', wrapper)
        self.assertIn('TimeoutStopSec=10', wrapper)
        self.assertIn('ExecMainStatus', wrapper)
        self.assertIn('cgroup.procs', wrapper)
        self.assertIn('UMI_ONLINE_CALIBRATION=tuned_online_v1', wrapper)
        self.assertIn('task_objective=${5:-plate_return}', wrapper)
        self.assertIn('--task-objective "$task_objective"', wrapper)

    def test_live_verifier_rejects_wrong_arm_and_stale_camera(self):
        import shutil
        import tempfile
        import json
        from verify_tuned_online import verify
        source = Path(__file__).parent/'sim_validation/tuned_online_v1_lingbot10000'
        if not source.exists():
            self.skipTest('private smoke artifact unavailable')
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            for path in source.iterdir():
                if path.is_file(): shutil.copy2(path,target/path.name)
            verify(target)
            path = target/'wrist_camera_poses.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['views']['right_wrist']['robot_side'] = 'left'
            path.write_text('\n'.join(map(json.dumps,rows))+'\n')
            with self.assertRaises(AssertionError): verify(target)

    def test_live_verifier_rejects_invalid_model_waypoint(self):
        import shutil
        import tempfile
        import json
        from verify_tuned_online import verify
        source = Path(__file__).parent/'sim_validation/tuned_online_v1_lingbot10000_verified'
        if not source.exists():
            self.skipTest('private smoke artifact unavailable')
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            for path in source.iterdir():
                if path.is_file(): shutil.copy2(path,target/path.name)
            path = target/'online_adapter.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['substep_waypoints'][0]['right']['gripper_open_fraction'] = 1.1
            path.write_text('\n'.join(map(json.dumps,rows))+'\n')
            with self.assertRaises(AssertionError): verify(target)


if __name__ == '__main__': unittest.main()
