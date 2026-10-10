"""Read partial diagnostic logs; no simulation or model mutation."""
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
last = None
streak = max_streak = 0
peak_clearance = peak_shape = 0.
first_lift = None
with (root / 'states.jsonl').open() as stream:
    for line in stream:
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        c = r['cup'];q = c['quaternion_wxyz'];norm = sum(v*v for v in q)
        tilt = math.degrees(math.acos(max(-1., min(1., 1 - 2*(q[1]**2+q[2]**2)/norm))))
        clearance = 1000 * (c['deformation']['min_node_world_z_m'] - .7505)
        streak = streak+1 if clearance > 10 and tilt <= 15 else 0
        max_streak = max(max_streak,streak)
        peak_shape = max(peak_shape,c['deformation']['max_nodal_shape_change_m']*1000)
        peak_clearance = max(peak_clearance,clearance)
        if first_lift is None and clearance > 10 and tilt <= 15:
            first_lift = r['physics_time_s']
        last = r
if last:
    print(json.dumps(dict(model_step=last['model_step'],physics_time_s=last['physics_time_s'],
        cup_tilt_deg=tilt,clearance_mm=clearance,peak_clearance_mm=peak_clearance,
        peak_shape_mm=peak_shape,longest_upright_clearance_s=max_streak/30,
        first_upright_clearance_over10mm_s=first_lift,
        stable_lift_confirmed=last['stable_lift_confirmed'],
        actual_jaw_rad=last['left']['joint_positions'][7],jaw_margin=last['jaw_margin'])))
if (root / 'result.json').exists():
    r = json.loads((root / 'result.json').read_text())
    print(json.dumps({k:r[k] for k in ('status','stop_reason','model_requests','left_return_success','stable_lift_confirmed','error') if k in r}))
