import unittest
from profiles import PROFILE_IDS, trial_profile


BASE = dict(id='verified',youngs_modulus_Pa=3e9,poissons_ratio=.38,
            surface_bend_stiffness_Pa=3e9/(12*(1-.38**2)),thickness_m=.001,
            solver_position_iterations=128,mass_kg=.0317,dynamic_friction=.4,
            diagnostic_max_shape_change_m=.06)


class ProfileTests(unittest.TestCase):
    def test_only_modulus_and_derived_bending_differ_between_trials(self):
        profiles = [trial_profile(e,BASE) for e in (3,2,1)]
        for p in profiles:
            self.assertEqual(p['solver_position_iterations'],128)
            self.assertEqual(p['diagnostic_max_shape_change_m'],.015)
            self.assertEqual(p['thickness_m'],.001)
        for a,b in zip(profiles,profiles[1:]):
            self.assertEqual({k for k in a if a[k]!=b[k]},
                             {'id','youngs_modulus_Pa','surface_bend_stiffness_Pa'})

    def test_shared_profile_not_mutated(self):
        old = dict(BASE);trial_profile(2,BASE);self.assertEqual(BASE,old)

    def test_unsupported_or_changed_baseline_rejected(self):
        for value in [.5,4,True,float('nan')]:
            with self.assertRaises(ValueError):trial_profile(value,BASE)
        with self.assertRaises(ValueError):trial_profile(3,{**BASE,'solver_position_iterations':32})

    def test_ids(self):
        self.assertEqual(PROFILE_IDS,tuple(trial_profile(e,BASE)['id'] for e in (3,2,1)))


if __name__=='__main__':unittest.main()
