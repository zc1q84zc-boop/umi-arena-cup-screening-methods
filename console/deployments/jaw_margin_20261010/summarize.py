"""Summarize executed closure margins and hold/deformation metrics."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from analyze_gpu import analyze


def summarize(root, package):
    with contextlib.redirect_stdout(io.StringIO()):
        analyze(root, package)
    result = json.loads((root / 'result.json').read_text())
    analysis = json.loads((root / 'analysis.json').read_text())
    rows = [json.loads(x) for x in (root / 'states.jsonl').read_text().splitlines()]
    active = [r for r in rows if r['jaw_margin']['applied_closure_margin_rad'] > 1e-9]
    peak = max(rows, key=lambda r: r['cup']['deformation']['max_nodal_shape_change_m'])
    margins = [r['jaw_margin']['applied_closure_margin_rad'] for r in rows]
    motion = []
    with (root / 'continuous_targets.jsonl').open() as stream:
        for line in stream:
            tick = json.loads(line)
            assert tick['dt_s'] == 1 / 240
            assert tick['response_gain'] == tick['jaw_response_gain'] == 4
            motion.append(np.array(tick['v']))
    motion = np.array(motion)
    velocity = float(abs(motion).max())
    acceleration = float(abs(np.diff(motion, axis=0) * 240).max())
    assert velocity <= .8 + 1e-9 and acceleration <= 1.5 + 1e-8
    summary = dict(
        classification=result['classification'],status=result['status'],
        requested_margin_rad=result['left_jaw_margin_rad'],
        actual_max_applied_margin_rad=max(margins),margin_active_servo_frames=len(active),
        gain_arm=4,gain_jaw=4,physics_hz=240,solver_iterations=128,
        longest_upright_clearance_above10mm_s=analysis['longest_upright_clearance_above10mm_s'],
        peak_clearance_mm=analysis['peak_left_clearance_mm'],
        peak_shape_change_mm=analysis['peak_shape_change_mm'],
        peak_rim_compression_percent=analysis['peak_rim_compression_percent'],
        first_tilt_above30_time_s=analysis['first_left_tilt_over30_time_s'],
        windows=analysis['windows'],peak_deformation_model_step=peak['model_step'],
        peak_deformation_actual_jaw_rad=peak['left']['joint_positions'][7],
        peak_deformation_raw_goal_rad=peak['jaw_margin']['raw_left_jaw_goal_rad'],
        peak_deformation_adjusted_goal_rad=peak['jaw_margin']['adjusted_left_jaw_goal_rad'],
        motion_velocity_max_rad_s=velocity,motion_acceleration_max_rad_s2=acceleration,
        stop_reason=result.get('stop_reason'),left_return_success=result.get('left_return_success',False),
        right_and_arm_cartesian_targets_unchanged=True,
        measured_grasp_force=False,full_pipeline_evaluation=False,
        source_time_offset_s=result.get('reference_time_offset_s',0))
    (root / 'margin_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--package', type=Path, required=True)
    args = p.parse_args()
    summarize(args.root, args.package)
