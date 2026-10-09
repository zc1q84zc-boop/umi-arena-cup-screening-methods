import copy
import unittest
from verify_intersection_grasp import evaluate

def rows(height=.81, count=12):
    contact={p:{'positive_normal_force_sum_N':0. if p=='base' else .5,'buffer_truncated':False}
             for p in ('left_finger','right_finger','base')}
    return [{'request_step':i//3,'action_index':i,'physics_time_s':i/30,
             'cup_position_m':[0,0,height],'cup_quaternion_wxyz':[1,0,0,0],
             'robots':{s:{'cup_contacts':copy.deepcopy(contact)} for s in ('left','right')}} for i in range(count)]

class GraspTests(unittest.TestCase):
    def test_stable_bilateral_elevated_passes(self):
        self.assertTrue(evaluate(rows())['numeric_stable_grasp'])
    def test_table_contact_or_one_finger_cannot_pass(self):
        self.assertFalse(evaluate(rows(height=.75))['numeric_stable_grasp'])
        data=rows()
        for r in data:
            for s in r['robots']:r['robots'][s]['cup_contacts']['right_finger']['positive_normal_force_sum_N']=0
        self.assertFalse(evaluate(data)['numeric_stable_grasp'])
    def test_duplicate_physics_samples_do_not_inflate_dwell(self):
        data=rows(count=9)
        self.assertFalse(evaluate([r for r in data for _ in range(8)])['numeric_stable_grasp'])
    def test_base_support_or_tilt_cannot_pass(self):
        data=rows()
        for r in data:
            for s in r['robots']:r['robots'][s]['cup_contacts']['base']['positive_normal_force_sum_N']=1
        self.assertFalse(evaluate(data)['numeric_stable_grasp'])
        for r in data:r['cup_quaternion_wxyz']=[.70710678,.70710678,0,0]
        self.assertFalse(evaluate(data)['numeric_stable_grasp'])

if __name__=='__main__':unittest.main()
