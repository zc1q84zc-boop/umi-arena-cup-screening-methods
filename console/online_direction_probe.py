"""Bounded axis/gripper diagnostic through the SAME online body-action adapter.

Not a learned policy and not a grasp performance test. World +X is robot
forward; world -Z is down. We never interpret a raw body-delta X as forward.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi05_isaac_online_adapter import current_robot, _waypoint, _source_from_fraction, rotate, conj, CALIBRATION

ROOT = Path(os.environ['SIM_ADAPTER_AUDIT_DIR'])
BASE = None
PHASES = [('settle',[0,0,0],1.), ('forward',[.02,0,0],1.),
          ('return_forward',[0,0,0],1.), ('down',[0,0,-.02],1.),
          ('return_down',[0,0,0],1.), ('close',[0,0,0],0.), ('open',[0,0,0],1.)]


def predict(observation, step, episode):
    global BASE
    if not CALIBRATION: raise ValueError('probe requires active online calibration')
    if step == 0:
        BASE = {s: observation['robots'][s]['tool_pose'] for s in ('left','right')}
    if step == 20:
        BASE = {s: observation['robots'][s]['tool_pose'] for s in ('left','right')}
    name, delta, fraction = PHASES[min(step//20,6)]
    current = {s: current_robot(observation,s) for s in ('left','right')}
    actions = np.zeros((3,16)); actions[:,6]=actions[:,13]=1.
    targets = {}
    for side,offset,gi in [('left',0,14),('right',7,15)]:
        world_goal=np.asarray(BASE[side]['position_m'])+delta
        # Fixed current orientation isolates translational direction.
        goal,_=CALIBRATION.tool_to_hand(side,world_goal,observation['robots'][side]['tool_pose']['quaternion_wxyz'])
        actions[:,offset:offset+3]=rotate(conj(current[side][1]),goal-current[side][0])/3
        actions[:,gi]=_source_from_fraction(fraction)
        targets[side]=world_goal.tolist()
    waypoints,diagnostic=_waypoint(current,actions)
    ROOT.mkdir(parents=True,exist_ok=True)
    row={'step':step,'phase':name,'requested_world_delta':delta,'world_targets':targets,
         'body_actions':actions.tolist(),'calibration':CALIBRATION.audit(),
         'robots':observation['robots'],'waypoint':waypoints,'mapping':diagnostic}
    row['image_metadata']=observation.get('image_metadata',{})
    with (ROOT/'direction_probe.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    if step%20 in (0,19):
        for side in ('left','right'):
            Image.fromarray(observation['images'][side+'_wrist']).save(ROOT/f'{name}_{step}_{side}.jpg')
    return {'action_dt_s':.1,'waypoints':[waypoints],'execute_steps':1}
