"""Evaluate a rigid cup lift from simulator telemetry; never a policy input."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_rows(path):
    if not path.is_file():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A running process can still be writing its last row.
            continue
    return result


def bottom_clearance(row, table_z=.75, base_radius=.027):
    q = row['spatial_alignment']['cup_quaternion_wxyz']
    norm = math.sqrt(sum(float(value)**2 for value in q))
    if norm < 1e-9:
        raise ValueError('Invalid cup orientation')
    w, x, y, z = [value / norm for value in q]
    tilt = math.acos(max(-1., min(1., 1 - 2*(x*x+y*y))))
    # For an upright or moderately tilted frustum, the lowest point is on
    # its circular base. This is specific to the current rigid 27 mm base.
    clearance = row['cup_position_m'][2] - table_z - base_radius*math.sin(tilt)
    return clearance, math.degrees(tilt)


def evaluate(rows, contact_rows, *, required_steps=10):
    contacts = {}
    for row in contact_rows:
        for side in ('left', 'right'):
            values = row['robots'][side]['cup_contacts']
            both = all(values[finger]['positive_normal_force_sum_N'] >= .02
                       for finger in ('left_finger', 'right_finger'))
            contacts.setdefault((row['request_step'], side), []).append(both)
    longest = {side: [] for side in ('left', 'right')}
    current = {side: [] for side in longest}
    max_clearance = 0.
    for row in rows:
        step = row['policy_step']
        clearance, tilt = bottom_clearance(row)
        max_clearance = max(max_clearance, clearance)
        for side in longest:
            samples = contacts.get((step, side), [])
            contact = bool(samples) and sum(samples) >= len(samples)/2
            # A tossed cup leaving the gripper must not count as held.
            tool = row['tool_poses'][side]['position_m']
            center = [*row['cup_position_m'][:2], row['cup_position_m'][2]+.0375]
            follows_hand = math.dist(tool, center) <= .12
            valid = clearance >= .05 and tilt <= 25 and contact and follows_hand
            if not valid or (current[side] and step != current[side][-1]+1):
                current[side] = []
            if valid:
                current[side].append(step)
                if len(current[side]) > len(longest[side]):
                    longest[side] = current[side].copy()
    winner = max(longest, key=lambda side: len(longest[side]))
    interval = longest[winner]
    return {'physical_pickup_verified': len(interval) >= required_steps,
            'side': winner if interval else None,
            'max_bottom_clearance_mm': max_clearance*1000,
            'longest_held_request_steps': len(interval),
            'held_interval_request_steps': [interval[0], interval[-1]] if interval else None,
            'held_duration_sim_s': len(interval)/10,
            'criteria': {'minimum_bottom_clearance_mm': 50, 'maximum_tilt_deg': 25,
                         'minimum_bilateral_contact_N': .02,
                         'minimum_hold_sim_s': required_steps/10,
                         'maximum_tool_to_cup_center_m': .12},
            'geometry': 'rigid 75 mm frustum with 27 mm base radius; table Z=0.75 m',
            'contact_force_calibration': 'unmeasured PhysX telemetry'}


def inspect(directory):
    rows = read_rows(directory/'evaluation.jsonl')
    contacts = read_rows(directory/'gripper_contact_audit.jsonl')
    result = evaluate(rows, contacts)
    actions = read_rows(directory/'online_adapter.jsonl')
    official = bool(actions) and all(
        row.get('model') == 'lingbot-vla2-official-pretrained'
        and row.get('model_provenance', {}).get('fine_tuned') is False
        and row.get('model_provenance', {}).get('all_files_sha256_verified') is True
        for row in actions)
    result.update(requests=len(actions), completed_telemetry_steps=len(rows),
                  official_model_identity_verified=official,
                  official_model_pickup_verified=result['physical_pickup_verified'] and official)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.run_dir), indent=2))
