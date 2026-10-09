"""Validate recorded online commands, not grasp success or real robot safety."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def validate(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    result = json.loads((directory/'report.json').read_text())
    obs = [json.loads(x) for x in (directory/'online_adapter.jsonl').read_text().splitlines()]
    rows = [json.loads(x) for x in (directory/'continuous_targets.jsonl').read_text().splitlines()]
    assert meta['status'] == 'completed' and meta['inference_cleanup'] == 'stopped'
    assert len(obs) == meta['steps']
    assert [r['step'] for r in obs] == list(range(len(obs)))
    assert len(rows) == len(obs)*6
    assert [(r['policy_step'],r['substep']) for r in rows] == [(i,j) for i in range(len(obs)) for j in range(1,7)]
    assert all(r['calibration']['id'] == 'mirrored_replay_prior_20260929' for r in obs)
    assert all(r['observation_origin'] == 'current_simulator_render_and_robot_state' for r in obs)
    q = np.array([r['q'] for r in rows]); v = np.array([r['v'] for r in rows]); dt = rows[0]['dt_s']
    assert np.isfinite(q).all() and np.isfinite(v).all()
    a = np.diff(np.concatenate([np.zeros_like(v[:1]), v]), axis=0)/dt
    assert abs(v).max() <= .8+1e-8 and abs(a).max() <= 1.5+1e-8
    np.testing.assert_allclose(np.diff(q,axis=0)/dt, v[1:], atol=1e-5)
    for row in obs:
        for side in ('left','right'):
            goal=row['waypoint'][side]
            assert np.isfinite(goal['position_m']).all()
            assert abs(np.linalg.norm(goal['quaternion_wxyz'])-1)<1e-6
            assert 0<=goal['gripper_open_fraction']<=1
    with (directory/'joints.csv').open() as file:
        actual=[r for r in csv.DictReader(file) if r['joint_kind']=='arm']
    return {'run':meta['id'], 'policy':meta['policy'], 'steps':len(obs), 'physics_commands':len(rows),
        'max_command_velocity_rad_s':float(abs(v).max()), 'max_command_acceleration_rad_s2':float(abs(a).max()),
        'max_observed_arm_velocity_rad_s':max(abs(float(r['velocity'])) for r in actual),
        'input_unique_images':{side:len({r['image_sha256'][side] for r in obs}) for side in obs[0]['image_sha256']},
        'calibration':obs[0]['calibration'], 'task_success':result['episodes'][0]['success'],
        'scope':'command continuity / causal adapter smoke, not task success or physical acceleration certification'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+'); parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    report=[validate(p) for p in args.runs]
    payload=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output: args.output.write_text(payload+'\n')
    print(payload)
