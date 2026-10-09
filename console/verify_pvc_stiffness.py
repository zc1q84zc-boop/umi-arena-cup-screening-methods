"""Evidence-only evaluation of an elastic cup lift, never an action controller."""
import json
import math
from pathlib import Path


def stable_grasp(rows, table_z=.75, thickness_m=.001):
    """Require upright, slow, undeformed-enough bottom clearance for 10 unique steps."""
    longest = current = 0
    last_action = None
    peak_clearance = float('-inf')
    for row in rows:
        action = row.get('action_index')
        if action == last_action:
            continue
        consecutive = last_action is not None and action == last_action + 1
        last_action = action
        try:
            # FEM nodes are the shell midsurface. Its collision rest offset
            # is half the wall thickness, not an extra physical lift.
            clearance = float(row['min_node_world_z_m']) - thickness_m/2 - table_z
            q = row['cup_quaternion_wxyz']
            linear = row['cup_linear_velocity_m_s']
            angular = row['cup_angular_velocity_rad_s']
            values = [clearance, *q, *linear, *angular, row['max_nodal_shape_change_m']]
            finite = all(math.isfinite(v) for v in values)
            upright = len(q) == 4 and abs(sum(v*v for v in q)-1) < .001 and 1-2*(q[1]**2+q[2]**2) >= math.cos(math.radians(15))
            held = (finite and clearance >= .05 and upright
                    and len(linear) == len(angular) == 3
                    and math.sqrt(sum(v*v for v in linear)) <= .05
                    and math.sqrt(sum(v*v for v in angular)) <= .3
                    and row['max_nodal_shape_change_m'] < .015)
            if finite:
                peak_clearance = max(peak_clearance, clearance)
        except (KeyError, TypeError, ValueError, IndexError):
            held = False
        current = (current+1 if consecutive else 1) if held else 0
        longest = max(longest, current)
    return dict(stable_grasp_verified=longest >= 10, longest_stable_action_steps=longest,
                max_bottom_clearance_m=peak_clearance if math.isfinite(peak_clearance) else None,
                bottom_clearance_threshold_m=.05, minimum_hold_action_steps=10,
                upright_tolerance_deg=15, rigid_fit_root_lift_is_not_used=True)


def evaluate(directory):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    rows = [json.loads(x) for x in (directory/'cup_deformation.jsonl').read_text().splitlines()]
    return dict(run_id=directory.name, status=report['status'],
                profile=report['cup_physics_profile'],
                requests=report['episodes'][0]['policy_steps'],
                task_success=report['episodes'][0].get('success', False),
                max_shape_change_m=max(r['max_nodal_shape_change_m'] for r in rows),
                max_rim_compression_fraction=max(r['rim_compression_fraction'] for r in rows),
                **stable_grasp(rows, thickness_m=report['cup_physics_profile']['thickness_m']))


if __name__ == '__main__':
    import sys
    print(json.dumps(evaluate(sys.argv[1]), indent=2, allow_nan=False))
