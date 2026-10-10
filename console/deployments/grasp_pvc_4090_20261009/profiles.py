"""Matched-precision elastic profiles; no change to shared/default registry."""
VALID_GPA = (3., 2., 1.)
PROFILE_IDS = tuple(f'diagnostic_grasp_pvc_e{int(e)}_i128_h240' for e in VALID_GPA)


def trial_profile(gpa, base_e3):
    if type(gpa) not in (int, float) or gpa not in VALID_GPA:
        raise ValueError('Only bounded 3/2/1 GPa trials supported')
    if (base_e3['youngs_modulus_Pa']!=3e9 or base_e3['thickness_m']!=.001
            or base_e3['solver_position_iterations']!=128 or base_e3['mass_kg']!=.0317):
        raise ValueError('Expected the verified 3 GPa, 1 mm, 128-iteration baseline')
    profile = dict(base_e3)
    profile.update(id=f'diagnostic_grasp_pvc_e{int(gpa)}_i128_h240',
                   youngs_modulus_Pa=gpa*1e9,
                   surface_bend_stiffness_Pa=gpa*1e9/(12*(1-profile['poissons_ratio']**2)),
                   diagnostic_max_shape_change_m=.015)
    return profile
