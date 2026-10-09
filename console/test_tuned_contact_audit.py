import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).parent/'simulator_profiles/tuned_v1/yubi_isaac_sim_env/contact_audit.py'
spec = importlib.util.spec_from_file_location('contact_audit_test', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ContactAuditTests(unittest.TestCase):
    def pair(self, count=2, start=1):
        return audit.decode_pair(([[99.], [2.], [3.]], [[0,0,0]]*3,
            [[1,0,0]]*3, [[0.]]*3, [[count]], [[start]]), [[[0.,0.,0.]]])

    def test_counts_and_starts_are_not_swapped(self):
        result = self.pair()
        self.assertEqual(result['reported_contact_count'], 2)
        self.assertEqual(result['positive_normal_force_sum_N'], 5.)
        self.assertFalse(result['buffer_truncated'])

    def test_cancelling_force_does_not_look_contact_free(self):
        result = self.pair()
        self.assertEqual(result['net_force_norm_N'], 0)
        self.assertEqual(result['positive_normal_force_sum_N'], 5.)

    def test_overflow_is_explicit(self):
        self.assertTrue(self.pair(count=8)['buffer_truncated'])
        self.assertEqual(self.pair(count=8)['buffered_contact_count'], 2)

    def test_negative_index_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError): self.pair(start=-1)
        with self.assertRaises(ValueError):
            audit.decode_pair(([[float('nan')]], [[0,0,0]], [[1,0,0]], [[0]], [[1]], [[0]]), [[[0,0,0]]])

    def test_summary_and_support_positive_control(self):
        summary = audit.ContactSummary(1/60)
        contact = self.pair()
        empty = self.pair(count=0)
        row = dict(cup_net_contact_force_world_N=[0,0,.98], robots={s: dict(cup_contacts={
            'left_finger': contact, 'right_finger': contact, 'base': empty}) for s in ('left','right')})
        summary.update(row)
        summary.update(row)
        row['robots']['right']['cup_contacts']['right_finger'] = empty
        summary.update(row)
        result = summary.result()
        self.assertTrue(result['cup_contact_signal_observed'])
        self.assertEqual(result['sample_hz'],60)
        self.assertEqual(result['arms']['right']['bilateral_samples'],2)
        self.assertEqual(result['arms']['right']['longest_bilateral_samples'],2)

    def test_force_dt_and_no_solver_change(self):
        env = (PATH.parent/'env.py').read_text()
        self.assertIn("'disable_stablization': False",env)
        self.assertIn('get_contact_force_data(clone=False, dt=dt)',env)
        self.assertIn('get_contact_force_matrix(clone=False, dt=dt)',env)
        self.assertIn('get_net_contact_forces(clone=False, dt=dt)',env)


if __name__ == '__main__': unittest.main()
