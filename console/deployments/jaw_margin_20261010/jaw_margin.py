"""Apply a small, left-only closure margin while retaining commanded release."""
import copy
import math


def adjust(action, margin_rad, q_closed=-.1, q_open=.7):
    if not math.isfinite(margin_rad) or not 0 <= margin_rad <= .02:
        raise ValueError('Closure margin must be between 0 and .02 radians')
    if not q_open > q_closed:
        raise ValueError('Expected increasing angle to open the jaw')
    out = copy.deepcopy(action)
    fraction = float(out['left']['gripper_open_fraction'])
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid aperture')
    # Full margin through .60; taper to zero by .65 so release stays available.
    weight = min(1., max(0., (.65 - fraction) / .05))
    applied = min(margin_rad * weight, fraction * (q_open - q_closed))
    out['left']['gripper_open_fraction'] = max(0., fraction - applied / (q_open - q_closed))
    info = dict(raw_left_jaw_goal_rad=q_closed + fraction * (q_open - q_closed),
                adjusted_left_jaw_goal_rad=q_closed + out['left']['gripper_open_fraction'] * (q_open - q_closed),
                applied_closure_margin_rad=applied, commanded_release_unmodified=fraction >= .65)
    return out, info
