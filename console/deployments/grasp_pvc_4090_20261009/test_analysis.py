import unittest
from analyze_pvc import endpoints, timing
from verify_pvc_stiffness import stable_grasp


def row(action, time, bottom=.81, shape=.001):
    return dict(action_index=action,physics_time_s=time,min_node_world_z_m=bottom,
                cup_quaternion_wxyz=[1.,0.,0.,0.],cup_linear_velocity_m_s=[0.,0.,0.],
                cup_angular_velocity_rad_s=[0.,0.,0.],max_nodal_shape_change_m=shape)


class AnalysisTests(unittest.TestCase):
    def test_duplicates_not_counted_as_stable_steps(self):
        rows = [row(0,i/240) for i in range(20)]
        self.assertFalse(stable_grasp(endpoints(rows))['stable_grasp_verified'])

    def test_identical_startup_capture_allowed_but_not_counted(self):
        a = row(0,0)
        self.assertEqual(endpoints([a,dict(a),row(0,1/30)]),[row(0,1/30)])
        with self.assertRaises(ValueError):endpoints([a,row(0,0,bottom=.75)])

    def test_last_capture_not_earlier_rigid_fit_height(self):
        rows = [r for i in range(12) for r in
                (row(i,i/30,bottom=.84), row(i,i/30+1/240,bottom=.7505))]
        self.assertFalse(stable_grasp(endpoints(rows))['stable_grasp_verified'])

    def test_ten_real_upright_slow_steps(self):
        rows = [row(i,i/30) for i in range(10)]
        self.assertTrue(timing(rows))
        self.assertTrue(stable_grasp(rows)['stable_grasp_verified'])

    def test_gap_or_wrong_period(self):
        self.assertFalse(timing([row(0,0),row(2,1/30)]))
        self.assertFalse(timing([row(0,0),row(1,1/240)]))

    def test_nonfinite_time_or_reversal(self):
        for rows in ([row(0,float('nan'))],[row(1,0),row(0,1/30)],
                     [row(0,0),row(1,0)]):
            with self.assertRaises(ValueError):endpoints(rows)

    def test_shape_or_tilt_rejects(self):
        rows = [row(i,i/30,shape=.016) for i in range(12)]
        self.assertFalse(stable_grasp(rows)['stable_grasp_verified'])
        for r in rows:
            r['max_nodal_shape_change_m'] = 0
            r['cup_quaternion_wxyz'] = [0.,1.,0.,0.]
        self.assertFalse(stable_grasp(rows)['stable_grasp_verified'])


if __name__ == '__main__':unittest.main()
