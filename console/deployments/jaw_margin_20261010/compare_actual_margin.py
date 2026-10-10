"""Check matched source goals and quantify the measured jaw angle change."""
import argparse
import json
from pathlib import Path
import numpy as np


def rows(root):
    with (root / 'states.jsonl').open() as stream:
        return {r['servo_step']: r for r in (json.loads(line) for line in stream)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline', type=Path)
    p.add_argument('trial', type=Path)
    a = p.parse_args()
    base, trial = rows(a.baseline), rows(a.trial)
    requested=json.loads((a.trial / 'result.json').read_text())['left_jaw_margin_rad']
    assert set(base) == set(trial)
    deltas, tracking, grasp_deltas = [], [], []
    for key, r in trial.items():
        b = base[key]
        assert r['goal'] == b['goal'], 'Source Cartesian and aperture goals changed'
        assert r['model_step'] == b['model_step']
        if r['jaw_margin']['applied_closure_margin_rad'] >= .999 * requested:
            deltas.append(r['left']['joint_positions'][7] - b['left']['joint_positions'][7])
            tracking.append(r['left']['joint_positions'][7] - r['jaw_margin']['adjusted_left_jaw_goal_rad'])
            if 1450 <= r['model_step'] <= 1510:
                grasp_deltas.append(deltas[-1])
    result = dict(
        matched_source_goal_rows=len(trial),original_model_goals_identical=True,
        physical_actual_jaw_delta_median_rad=float(np.median(deltas)),
        physical_actual_jaw_delta_p05_rad=float(np.percentile(deltas,5)),
        physical_actual_jaw_delta_p95_rad=float(np.percentile(deltas,95)),
        physical_actual_jaw_delta_min_rad=float(min(deltas)),
        actual_to_adjusted_goal_error_median_rad=float(np.median(tracking)),
        grasp_window_actual_jaw_delta_median_rad=float(np.median(grasp_deltas)),
        grasp_window_actual_jaw_delta_p05_rad=float(np.percentile(grasp_deltas,5)),
        grasp_window_actual_jaw_delta_p95_rad=float(np.percentile(grasp_deltas,95)),
        positive_delta_means_more_open=True,
        no_measured_force_claim=True)
    (a.trial / 'paired_jaw_angles.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
