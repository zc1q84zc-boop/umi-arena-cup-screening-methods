"""Offline provenance and actual shell-bottom checks; never control the robot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verify_pvc_stiffness import stable_grasp

SOURCE_SHA = '7bfba23843272f78ea8a2278fea976d75add22588143d42ba511928ee9e4608b'


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def endpoints(rows):
    """Keep the last observation per independent action, not duplicate captures."""
    result = {}
    last_action, last_time, last_row = -1, -math.inf, None
    for row in rows:
        action, time = row['action_index'], row['physics_time_s']
        if type(action) is not int or action < 0 or action < last_action:
            raise ValueError('Invalid or reversed action index')
        if time == last_time and row == last_row:
            # Startup capture writes the same restored state twice. Do not
            # count it twice, and reject conflicting same-time observations.
            continue
        if not math.isfinite(time) or time <= last_time:
            raise ValueError('Nonfinite or nonincreasing simulation time')
        result[action] = row
        last_action, last_time, last_row = action, time, row
    return list(result.values())


def timing(rows):
    complete = True
    for a, b in zip(rows, rows[1:]):
        complete &= b['action_index'] == a['action_index'] + 1
        # FEM is sampled at 30 Hz even though physics advances at 240 Hz.
        complete &= abs(b['physics_time_s'] - a['physics_time_s'] - 1/30) < 2e-5
    return bool(complete)


def command_check(commands, source):
    for index, command in enumerate(commands):
        assert command['episode'] == 0 and command['step'] == index
        assert command['source_sha256'] == SOURCE_SHA
        assert command['source_waypoints'] == source[index]['substep_waypoints']
        assert command['applied_waypoints'] == command['source_waypoints']
        assert command['execute_steps'] == 3 and command['action_dt_s'] == 1/30
        assert command['extra_closure_fraction'] == 0
        assert not command['model_inference_called'] and not command['object_feedback_used']
    return len(commands)*3


def analyze(path, source):
    report = json.loads((path/'report.json').read_text())
    assert report['status'] in ('completed', 'failed', 'stopped')
    assert report['model_inference_requests'] == 0
    provenance = report['frozen_pvc_diagnostic']
    assert provenance['source_sha256'] == SOURCE_SHA
    assert provenance['original_world_pose_and_jaw_commands_unchanged']
    profile = report['cup_physics_profile']
    assert profile['physics_hz'] == 240 and profile['solver_position_iterations'] == 128
    assert profile['thickness_m'] == .001 and profile['mass_kg'] == .0317
    emitted = command_check(jsonl(path/'frozen_command_audit.jsonl'), source)
    raw = jsonl(path/'cup_deformation.jsonl')
    rows = endpoints(raw)
    assert timing(rows), 'Action endpoints must remain on the independent 30 Hz grid'
    control = jsonl(path/'continuous_targets.jsonl')
    previous_v = None
    for row in control:
        assert row['dt_s'] == 1/240 and row['vmax'] == .8 and row['amax'] == 1.5
        assert row['mirrored_jaw']
        assert all(math.isfinite(v) and abs(v) <= .800001 for arm in row['v'] for v in arm)
        assert all(abs(arm[-1]+arm[-2]) < 1e-6 for arm in row['q'])
        if previous_v is not None:
            assert all(abs(v-old) <= 1.5/240+1e-6 for arm,old_arm in zip(row['v'],previous_v)
                       for v,old in zip(arm,old_arm)), 'Acceleration governor exceeded'
        previous_v = row['v']
    guard = path/'safety_abort.json'
    abort = json.loads(guard.read_text()) if guard.exists() else None
    screen = stable_grasp(rows, thickness_m=profile['thickness_m'])
    peak = max(rows, key=lambda r:r['min_node_world_z_m']) if rows else None
    result = dict(
        report_status=report['status'], error=report.get('error'),
        material_profile=profile, composed_material_readback=provenance['material_readback'],
        source_targets_numbers_order_timing_verified=True, servo_targets_emitted=emitted,
        independent_30hz_endpoints=len(rows), complete_requested_prefix=(emitted == 1080 and len(rows) == 1080),
        model_inference_requests=0, unchanged_arms_and_jaws=True,
        force_telemetry_available=False, contact_force_not_inferred_from_missing_data=True,
        max_shape_change_mm=max((r['max_nodal_shape_change_m'] for r in raw), default=0)*1000,
        max_rim_compression_fraction=max((r['rim_compression_fraction'] for r in raw), default=0),
        numeric_stable_grasp_screen=screen, physical_stable_grasp_verified=False,
        visual_review_required=True, safety_abort=abort, peak_clearance_sample=peak,
        no_claim_of_online_model_or_full_task_success=True,
        raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (path/'report.json',path/'manifest.json',path/'cup_deformation.jsonl',
                     path/'frozen_command_audit.jsonl',path/'continuous_targets.jsonl')})
    # This evidence is observational. A numerical pass still needs video/node review.
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == SOURCE_SHA
    source = jsonl(args.source)
    results = {p.name:analyze(p,source) for p in args.runs}
    first = json.loads((args.runs[0]/'report.json').read_text())
    for path in args.runs[1:]:
        report = json.loads((path/'report.json').read_text())
        for field in ('seed','setup_requested','task_objective','physics_hz','action_execution_hz',
                      'model_request_hz','trajectory_controller_profile','joint_command_profile','camera'):
            assert report[field] == first[field], field
        p, q = first['cup_physics_profile'], report['cup_physics_profile']
        assert {k for k in p if p[k] != q[k]} <= {'id','youngs_modulus_Pa','surface_bend_stiffness_Pa'}
        assert report['input_sha256'] == first['input_sha256']
        assert report['frozen_pvc_diagnostic']['files_sha256'] == first['frozen_pvc_diagnostic']['files_sha256']
    value = dict(classification='matched_precision_frozen_action_FEM_material_comparison',
                 source_run='c3b23931ada5', results=results,
                 comparison_to_rigid_also_changes_contact_backend_and_numerics=True,
                 measured_PVC_material=False, default_scene_or_weights_modified=False)
    with args.output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print(json.dumps({k:{f:r[f] for f in ('report_status','servo_targets_emitted',
         'independent_30hz_endpoints','max_shape_change_mm','numeric_stable_grasp_screen','safety_abort')}
         for k,r in results.items()},indent=2,allow_nan=False))


if __name__ == '__main__':main()
