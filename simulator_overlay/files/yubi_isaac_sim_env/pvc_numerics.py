"""Independent solver-accuracy trial; original material profiles stay intact."""
PRECISION_ID = 'pvc_shell_e3000mpa_i128_h240_v2'


def precision_profile(base_e3):
    if base_e3['youngs_modulus_Pa'] != 3e9 or base_e3['thickness_m'] != .001:
        raise ValueError('High-accuracy trial requires the unchanged 3GPa/1mm material')
    profile = dict(base_e3)
    profile.update(id=PRECISION_ID, solver_position_iterations=128)
    return profile


def physics_hz_for(profile_id):
    return 240 if profile_id == PRECISION_ID else 120
