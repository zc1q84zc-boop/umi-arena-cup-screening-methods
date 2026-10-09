from copy import deepcopy
import math
import unittest
from margin_core import closure_target, filtered_waypoints, decode_friction


class MarginTests(unittest.TestCase):
    def test_bounded_and_release_unchanged(self):
        for i in range(1001):
            f = i/1000
            out = closure_target(f, .02)
            self.assertTrue(0 <= out <= f)
            self.assertLessEqual(f-out, .020000001)
            if f >= .65: self.assertEqual(f, out)
        self.assertAlmostEqual(closure_target(.52, .02), .5)
        self.assertAlmostEqual(closure_target(.60, .02), .59)

    def test_monotone_and_no_bias_baseline(self):
        outputs = [closure_target(i/1000, .02) for i in range(1001)]
        self.assertEqual(outputs, sorted(outputs))
        for i in range(1001): self.assertEqual(closure_target(i/1000, 0), i/1000)

    def test_bad_values_rejected(self):
        for pair in [(math.nan, .02), (.5, math.inf), (.5, -.01), (.5, .031), (1.1, .02), (True, .02)]:
            with self.assertRaises(ValueError): closure_target(*pair)

    def test_only_right_jaw_changes(self):
        target = {s: {'position_m': [0,0,1], 'quaternion_wxyz': [1,0,0,0],
                     'gripper_open_fraction': .52} for s in ('left','right')}
        original = [deepcopy(target) for _ in range(3)]
        old = deepcopy(original)
        out = filtered_waypoints(original, .02)
        self.assertEqual(original, old)
        for a,b in zip(out,old):
            self.assertEqual(a['left'], b['left'])
            for k in ('position_m','quaternion_wxyz'):self.assertEqual(a['right'][k], b['right'][k])
            self.assertAlmostEqual(a['right']['gripper_open_fraction'], .5)

    def test_friction_counts_starts_and_vectors(self):
        out = decode_friction(([[9,9,9],[1,2,3],[-1,-2,-3]], [[0,0,0]]*3, [[2]], [[1]]))
        self.assertEqual(out['net_tangential_force_world_N'], [0,0,0])
        self.assertAlmostEqual(out['sum_anchor_force_norms_N'], 2*math.sqrt(14))
        self.assertEqual(out['reported_anchor_count'], 2)
        self.assertFalse(out['buffer_truncated'])

    def test_friction_overflow_and_invalid(self):
        self.assertTrue(decode_friction(([[1,2,3]],[[0,0,0]],[[2]],[[0]]))['buffer_truncated'])
        for data in [([],[],[[-1]],[[0]]), ([[math.nan,0,0]],[[0,0,0]],[[1]],[[0]])]:
            with self.assertRaises(ValueError):decode_friction(data)


if __name__ == '__main__':unittest.main()
