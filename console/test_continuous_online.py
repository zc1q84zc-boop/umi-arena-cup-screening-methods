import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/"simulator_profiles/tuned_v1"))
from yubi_isaac_sim_env.continuous_targets import ContinuousTargets


class GovernorTests(unittest.TestCase):
    def test_body_delta_to_forward_and_down_both_arms(self):
        import pi05_isaac_online_adapter as adapter
        from online_calibration import MirroredReplayPrior
        old = adapter.CALIBRATION
        adapter.CALIBRATION = MirroredReplayPrior()
        try:
            for direction in ([.01,0,0],[0,0,-.01]):
                obs = {"robots": {s: {"tool_pose": {"position_m": [.1, .3 if s=='left' else -.3, 1.],
                    "quaternion_wxyz": [.5,.5,.5,.5]}, "gripper_open_fraction": 1.} for s in ('left','right')}}
                current={s:adapter.current_robot(obs,s) for s in ('left','right')}
                a=np.zeros((3,16));a[:,6]=a[:,13]=1.;a[:,14:]=.78
                source_delta=adapter.CALIBRATION.a.T@direction
                for s,k in [('left',0),('right',7)]:
                    a[:,k:k+3]=adapter.rotate(adapter.conj(current[s][1]),source_delta)/3
                out,_=adapter._waypoint(current,a)
                for s in out:
                    np.testing.assert_allclose(np.array(out[s]['position_m'])-obs['robots'][s]['tool_pose']['position_m'],direction,atol=1e-12)
        finally: adapter.CALIBRATION=old

    def test_limits_acceleration_and_continuity_reversals(self):
        limits = np.broadcast_to([-1.,1.], (2,8,2))
        g = ContinuousTargets(np.zeros((2,8)), limits, 1/60)
        last_v = g.v.copy()
        last_q = g.q.copy()
        for k in range(6000):
            goal = np.full((2,8), 2 if (k//83)%2 else -2)
            q = g.step(goal)
            self.assertLessEqual(np.abs(g.v).max(), .8+1e-9)
            self.assertLessEqual(np.abs(g.v-last_v).max()/g.dt, 1.5+1e-8)
            self.assertTrue((q >= -1).all() and (q <= 1).all())
            np.testing.assert_allclose((q-last_q)/g.dt, g.v, atol=1e-12)
            last_v, last_q = g.v.copy(), q.copy()
        for _ in range(3000): g.step(np.ones((2,8)))
        np.testing.assert_allclose(g.q, 1., atol=1e-8)

    def test_invalid_goals_fail_closed(self):
        g = ContinuousTargets([0.], [[-1,1]], 1/60)
        with self.assertRaises(ValueError): g.step([float('nan')])

    def test_zero_action_preserves_observed_world_pose(self):
        import pi05_isaac_online_adapter as adapter
        from online_calibration import MirroredReplayPrior, matrix
        old = adapter.CALIBRATION
        adapter.CALIBRATION = MirroredReplayPrior()
        try:
            obs = {"robots": {s: {"tool_pose": {"position_m": [.3, .4 if s=='left' else -.4, .9],
                "quaternion_wxyz": [1,0,0,0]}, "gripper_open_fraction": .8} for s in ('left','right')}}
            current = {s: adapter.current_robot(obs,s) for s in ('left','right')}
            actions = np.zeros((3,16)); actions[:,6] = actions[:,13] = 1
            actions[:,14:] = adapter._source_from_fraction(.8)
            output,_ = adapter._waypoint(current, actions)
            for s in output:
                np.testing.assert_allclose(matrix(output[s]['position_m'],output[s]['quaternion_wxyz']),
                    matrix(obs['robots'][s]['tool_pose']['position_m'],[1,0,0,0]), atol=1e-12)
                self.assertAlmostEqual(output[s]['gripper_open_fraction'],.8)
        finally: adapter.CALIBRATION = old
