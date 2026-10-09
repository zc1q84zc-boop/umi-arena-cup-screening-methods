"""No-model, bounded initialization fitting through normal differential IK."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(os.environ['SIM_ADAPTER_AUDIT_DIR'])
PROFILE = json.loads(Path(os.environ['UMI_INITIAL_POSE_PROFILE']).read_text())
TARGET = PROFILE['selected']['targets']

def errors(observation):
    result = {}
    positions = []
    for side in ['left','right']:
        actual = observation['robots'][side]['tool_pose']
        p = np.array(actual['position_m']); positions.append(p)
        q = np.array(actual['quaternion_wxyz']); q /= np.linalg.norm(q)
        goal = TARGET[side]
        result[side] = dict(position_m=float(np.linalg.norm(p-goal['position_m'])),
            orientation_deg=float(np.degrees(2*np.arccos(np.clip(abs(q @ goal['quaternion_wxyz']),0,1)))),
            gripper_fraction=observation['robots'][side]['gripper_open_fraction'])
    result['tool_separation_m'] = float(np.linalg.norm(positions[0]-positions[1]))
    result['minimum_tool_height_m'] = float(min(p[2] for p in positions))
    return result

def predict(observation, step, episode):
    ROOT.mkdir(parents=True,exist_ok=True)
    error = errors(observation)
    with (ROOT/'initial_pose_fit.jsonl').open('a') as f:
        f.write(json.dumps(dict(step=step, errors=error, robots=observation['robots']))+'\n')
    if step in [0,39,79,119,159]:
        for name, image in observation.get('images', {}).items():
            Image.fromarray(np.asarray(image)).save(ROOT/f'{name}_{step:03d}.jpg')
    if error['tool_separation_m'] < .18 or error['minimum_tool_height_m'] < .86:
        raise RuntimeError('initial pose fit safety clearance failed; do not start learned policy')
    return dict(action_dt_s=.1, waypoints=[TARGET], execute_steps=1)

def rotation_error(goal, current):
    from scipy.spatial.transform import Rotation
    a=Rotation.from_quat(np.asarray(goal)[[1,2,3,0]])
    b=Rotation.from_quat(np.asarray(current)[[1,2,3,0]])
    return (a*b.inv()).as_rotvec()

def act(observation, step, episode):
    # Position-priority, damped null-space orientation and joint-centering,
    # adapted from the prior successful replay tracking diagnostic.
    predict(observation, step, episode)  # audit and clearance checks
    result = {}
    for side in ['left', 'right']:
        robot=observation['robots'][side]; goal=TARGET[side]
        indices=[robot['joint_names'].index(f'panda_joint{i}') for i in range(1,8)]
        q=np.asarray(robot['joint_positions'])[indices]; limits=np.asarray(robot['arm_joint_limits_rad'])
        jac=np.asarray(robot['tool_jacobian']); jp=jac[:3]
        inverse=jp.T @ np.linalg.inv(jp @ jp.T + .06**2*np.eye(3))
        null=np.eye(7)-inverse@jp
        error=np.asarray(goal['position_m'])-robot['tool_pose']['position_m']
        error*=min(1., .025/max(np.linalg.norm(error),1e-9))
        re=rotation_error(goal['quaternion_wxyz'],robot['tool_pose']['quaternion_wxyz'])
        re*=min(1., .12/max(np.linalg.norm(re),1e-9))
        jn=jac[3:]@null
        secondary=null@jn.T@np.linalg.solve(jn@jn.T+.12**2*np.eye(3), .35*re)
        center=-.012*(q-limits.mean(axis=1))/np.maximum(((limits[:,1]-limits[:,0])/2)**2,.2)
        dq=np.clip(inverse@(.55*error)+secondary+null@center,-.04,.04)
        target=np.clip(q+dq,limits[:,0]+.06,limits[:,1]-.06)
        result[side]=dict(arm_joint_targets_rad=target.tolist(),gripper_open_fraction=goal['gripper_open_fraction'])
    return result
