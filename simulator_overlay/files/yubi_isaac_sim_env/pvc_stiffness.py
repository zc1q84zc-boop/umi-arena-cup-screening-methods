"""Unmeasured hard-to-soft elastic-modulus trials; not Shore hardness."""

STIFFNESS_SPECS = (
    ('pvc_shell_e3000mpa_v1', 3.e9),
    ('pvc_shell_e2000mpa_v1', 2.e9),
    ('pvc_shell_e1000mpa_v1', 1.e9),
    ('pvc_shell_e0500mpa_v1', .5e9),
    ('pvc_shell_e0200mpa_v1', .2e9),
)
SHELL_PROFILE_IDS = ('pvc_elastic_shell_v1', *(key for key, _ in STIFFNESS_SPECS))


def profile_for(profile_id, base=None):
    if profile_id not in SHELL_PROFILE_IDS:
        raise ValueError('Unknown elastic shell profile')
    if base is None:
        from .pvc_shell import PROFILE
        base = PROFILE
    profile = dict(base)
    if profile_id != 'pvc_elastic_shell_v1':
        modulus = dict(STIFFNESS_SPECS)[profile_id]
        profile.update(id=profile_id, youngs_modulus_Pa=modulus,
                       surface_bend_stiffness_Pa=modulus/(12*(1-profile['poissons_ratio']**2)))
    return profile
