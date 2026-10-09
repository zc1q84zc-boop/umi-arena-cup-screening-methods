import math
import unittest

from inspect_cup_pickup import evaluate


def fixture(tilt_deg=0, contact=True, count=10):
    angle = math.radians(tilt_deg)/2
    rows = [{'policy_step': i, 'cup_position_m': [0, 0, .815],
             'spatial_alignment': {'cup_quaternion_wxyz': [math.cos(angle), math.sin(angle), 0, 0]},
             'tool_poses': {s: {'position_m': [0, 0, .85]} for s in ('left', 'right')}}
            for i in range(count)]
    contacts = [{'request_step': i, 'robots': {s: {'cup_contacts': {
        f: {'positive_normal_force_sum_N': .2 if contact and s == 'right' else 0.}
        for f in ('left_finger', 'right_finger')}} for s in ('left', 'right')}} for i in range(count)]
    return rows, contacts


class PickupTest(unittest.TestCase):
    def test_held_cup_is_verified(self):
        value = evaluate(*fixture())
        self.assertTrue(value['physical_pickup_verified'])
        self.assertEqual(value['side'], 'right')

    def test_height_without_contact_is_not_pickup(self):
        self.assertFalse(evaluate(*fixture(contact=False))['physical_pickup_verified'])

    def test_tipped_cup_and_short_contact_are_rejected(self):
        self.assertFalse(evaluate(*fixture(tilt_deg=60))['physical_pickup_verified'])
        self.assertFalse(evaluate(*fixture(count=9))['physical_pickup_verified'])

    def test_sample_gaps_cannot_extend_hold(self):
        rows, contacts = fixture(count=12)
        rows.pop(6)
        self.assertFalse(evaluate(rows, contacts)['physical_pickup_verified'])

    def test_cup_leaving_the_hand_is_rejected(self):
        rows, contacts = fixture()
        for row in rows:
            row['tool_poses']['right']['position_m'] = [1, 0, .85]
        self.assertFalse(evaluate(rows, contacts)['physical_pickup_verified'])


if __name__ == '__main__':
    unittest.main()
