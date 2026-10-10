"""Identical recorded joint-target sequence; this is not a physics replay."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from continuous_targets_candidate import ContinuousTargets

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'sim_validation/left_failure_612620c03d8a_20261010'
window = json.loads((SOURCE/'tracking_window.json').read_text())
reference = json.loads((SOURCE/'geometry_reference.json').read_text())
initial = window['samples']['870']['continuous']['start']
limits = np.array([reference['arm_joint_limits_rad'] + [[-.1, .7]]] * 2)
position = np.array(initial['q'])[:, :8]
velocity = np.array(initial['v'])
source_module_path = HERE.parents[1]/'simulator_profiles/tuned_v1/yubi_isaac_sim_env/continuous_targets.py'
spec = importlib.util.spec_from_file_location('production_governor',source_module_path)
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
results = []
for gain in (4., 8., 12., 20.):
    governor = ContinuousTargets(position, limits, 1/240, response_gain=gain)
    governor.v = velocity.copy()
    original = module.ContinuousTargets(position, limits, 1/240)
    original.v = velocity.copy()
    errors, velocities, accelerations = [], [], []
    for key, row in window['samples'].items():
        goal = np.array([row['transition']['action'][side]['arm_joint_targets_rad']+
                         [-.1+.8*row['transition']['action'][side]['gripper_open_fraction']]
                         for side in ('left','right')])
        for _ in range(24):
            prev = governor.v.copy()
            q = governor.step(goal)
            if gain == 4:
                # The isolated implementation must reproduce the production
                # kernel exactly at the baseline parameter.
                np.testing.assert_array_equal(q,original.step(goal))
                np.testing.assert_array_equal(governor.v,original.v)
            velocities.append(governor.v.copy());accelerations.append((governor.v-prev)*240)
            assert np.all(q >= limits[:,:,0]-1e-9) and np.all(q <= limits[:,:,1]+1e-9)
        errors.append(goal-q)
    error, v, a = np.array(errors),np.array(velocities),np.array(accelerations)
    assert abs(v).max() <= .8+1e-9 and abs(a).max() <= 1.5+1e-9
    reversals = ((v[1:] * v[:-1] < 0) & (abs(v[1:])>.02) & (abs(v[:-1])>.02)).sum()
    results.append(dict(response_gain=gain,linear_time_constant_s=1/gain,
                        left_arm_mae_rad=float(abs(error[:,0,:7]).mean()),
                        left_arm_absolute_error_p95_rad=float(np.percentile(abs(error[:,0,:7]),95)),
                        left_jaw_mae_rad=float(abs(error[:,0,7]).mean()),
                        max_speed_rad_s=float(abs(v).max()),max_acceleration_rad_s2=float(abs(a).max()),
                        above_002_rad_s_adjacent_velocity_reversals=int(reversals)))
result = dict(source_run=window['run_id'], model_steps=[870,1599], results=results,
              classification='offline fixed final-IK joint-target sequence; no physics, no model inference',
              baseline_kernel_bitwise_equal=True,
              target_sequence='last recorded IK goal of each model request held for 24 physics ticks; original first/second servo IK targets are unavailable',
              selected_for_physics_comparison=8.,
              selection_reason='Lowest arm mean/P95 error and jaw mean error in this fixed-sequence sweep; faster gains worsen acceleration-limited tracking')
(HERE/'offline_sweep.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
