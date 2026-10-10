"""Stress the optional response parameter at 240 Hz near joint boundaries."""
import importlib.util
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[2]
path=root/'simulator_profiles/tuned_v1/yubi_isaac_sim_env/continuous_targets.py'
spec=importlib.util.spec_from_file_location('governor',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checks=[]
for gain in (4.,8.):
    bounds=np.broadcast_to([-1.,1.],(2,8,2))
    g=module.ContinuousTargets(np.zeros((2,8)),bounds,1/240,response_gain=gain)
    max_v=max_a=0.
    for k in range(18000):
        goal=np.full((2,8),2. if (k//173)%2 else -2.)
        old_v=g.v.copy();old_q=g.q.copy();q=g.step(goal)
        max_v=max(max_v,float(abs(g.v).max()));max_a=max(max_a,float(abs(g.v-old_v).max()/g.dt))
        assert max_v<=.8+1e-9 and max_a<=1.5+1e-8
        assert (q>=-1-1e-9).all() and (q<=1+1e-9).all()
        np.testing.assert_allclose((q-old_q)/g.dt,g.v,atol=1e-12)
    for _ in range(9000):g.step(np.ones((2,8)))
    np.testing.assert_allclose(g.q,1.,atol=1e-8)
    checks.append(dict(response_gain=gain,steps=27000,max_v_rad_s=max_v,max_a_rad_s2=max_a,
                       joint_bounds_preserved=True,settled_at_upper_bound=True))
for gain in (0.,-1.,float('nan'),241.):
    try:module.ContinuousTargets([0.],[[-1,1]],1/240,response_gain=gain)
    except ValueError:pass
    else:raise AssertionError('invalid gain accepted')
out=dict(physics_hz=240,checks=checks,invalid_gain_rejected=True)
Path(__file__).with_name('motion_envelope.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
