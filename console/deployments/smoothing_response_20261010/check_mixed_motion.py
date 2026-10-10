"""Regression against gain4 and bound checks for arm8/jaw4."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from continuous_targets_candidate import ContinuousTargets

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('old_governor',here/'production_before/continuous_targets.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
limits=np.broadcast_to([-1.,1.],(2,8,2))
original=old.ContinuousTargets(np.zeros((2,8)),limits,1/240)
candidate=ContinuousTargets(np.zeros((2,8)),limits,1/240)
rng=np.random.default_rng(10)
for i in range(6000):
    if i%24==0:goal=rng.uniform(-2,2,(2,8))
    assert np.array_equal(original.step(goal),candidate.step(goal))
g=ContinuousTargets(np.zeros((2,8)),limits,1/240,response_gain=8,jaw_response_gain=4)
max_v=max_a=0.
for i in range(18000):
    goal=np.full((2,8),2. if (i//173)%2 else -2.)
    before=g.v.copy();q=g.step(goal)
    max_v=max(max_v,float(abs(g.v).max()));max_a=max(max_a,float(abs(g.v-before).max()/g.dt))
    assert max_v<=.8+1e-9 and max_a<=1.5+1e-8
    assert (q>=-1-1e-9).all() and (q<=1+1e-9).all()
for _ in range(9000):g.step(np.ones((2,8)))
np.testing.assert_allclose(g.q,1.,atol=1e-8)
x=dict(bitwise_default_gain4_regression_steps=6000,arm_response_gain=8,jaw_response_gain=4,
       envelope_steps=27000,max_command_velocity_rad_s=max_v,max_command_acceleration_rad_s2=max_a,
       joint_bounds_preserved=True)
(here/'mixed_motion_envelope.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
