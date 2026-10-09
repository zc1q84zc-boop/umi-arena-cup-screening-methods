"""Bounded jaw-only replay diagnostic; never observes objects or contact forces."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

MAX_EXTRA_FRACTION = .03


def closure_target(fraction, extra):
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (fraction, extra)):
        raise ValueError('Finite numeric fractions required')
    if not 0 <= fraction <= 1 or not 0 <= extra <= MAX_EXTRA_FRACTION:
        raise ValueError('Fraction outside bounded diagnostic range')
    # Same .55 closed/.65 release fade as the existing left-return diagnostic.
    # No latch: intentional opening beyond .65 is never held or tightened.
    weight = min(1., max(0., (.65 - fraction) / .10))
    return max(0., fraction - extra * weight)


def filtered_waypoints(waypoints, extra):
    out = deepcopy(waypoints)
    for waypoint in out:
        waypoint['right']['gripper_open_fraction'] = closure_target(
            waypoint['right']['gripper_open_fraction'], extra)
    return out


def load_source(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Frozen action source SHA256 mismatch')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 600 or [r['step'] for r in rows] != list(range(600)):
        raise ValueError('Expected the frozen 600-request episode')
    for row in rows:
        if row['episode'] != 0 or len(row['substep_waypoints']) != 3:
            raise ValueError('Expected episode 0, three 30Hz targets per request')
        for waypoint in row['substep_waypoints']:
            if set(waypoint) != {'left', 'right'}:
                raise ValueError('Expected both anatomical hands')
            for target in waypoint.values():
                values = [*target['position_m'], *target['quaternion_wxyz'], target['gripper_open_fraction']]
                if len(values) != 8 or not all(math.isfinite(float(v)) for v in values):
                    raise ValueError('Nonfinite or malformed world target')
                if abs(sum(v*v for v in target['quaternion_wxyz']) - 1) > .001:
                    raise ValueError('Nonunit world quaternion')
                closure_target(target['gripper_open_fraction'], 0.)
    return rows


def _list(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, 'tolist') else list(value)


def decode_friction(data):
    """Isaac5.1 four buffers; caller supplies physical dt to obtain N.

    Tangential anchors are NOT paired one-to-one with normal contact records.
    Summed anchor norms are not a Coulomb-utilization ratio.
    """
    forces, points, counts, starts = map(_list, data)
    if len(counts) != 1 or len(starts) != 1 or len(counts[0]) != 1 or len(starts[0]) != 1:
        raise ValueError('Expected one sensor body and one cup filter')
    count, start = int(counts[0][0]), int(starts[0][0])
    if count < 0 or start < 0:
        raise ValueError('Invalid friction buffer indices')
    end = min(start+count, len(forces), len(points))
    anchors = []
    for i in range(start, end):
        f, p = list(map(float, forces[i])), list(map(float, points[i]))
        if len(f) != 3 or len(p) != 3 or not all(math.isfinite(v) for v in [*f, *p]):
            raise ValueError('Nonfinite or malformed friction anchor')
        anchors.append({'force_world_N': f, 'point_world_m': p})
    net = [sum(a['force_world_N'][j] for a in anchors) for j in range(3)]
    return {'reported_anchor_count': count, 'buffered_anchor_count': len(anchors),
            'buffer_truncated': count != len(anchors), 'anchors': anchors,
            'net_tangential_force_world_N': net,
            'net_tangential_force_norm_N': math.sqrt(sum(v*v for v in net)),
            'sum_anchor_force_norms_N': sum(math.sqrt(sum(v*v for v in a['force_world_N'])) for a in anchors),
            'anchor_to_normal_point_pairing': 'not_available'}
