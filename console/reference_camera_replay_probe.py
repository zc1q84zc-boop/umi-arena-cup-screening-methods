"""Bounded no-model reference replay. Keep frame/pose residuals with images."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(os.environ['SIM_ADAPTER_AUDIT_DIR'])
DATA = json.loads(Path(os.environ['UMI_REFERENCE_EPISODE']).read_text())
A = np.array([[0,1,0],[-1,0,0],[0,0,1.]])
C = np.array([[0,0,1],[1,0,0],[0,1,0.]])
T = np.array([-.209307083410,.202121290786,.749006156995])

def target(frame):
    obs = DATA['frames'][frame]['observation']
    goals = {}
    for i, side in enumerate(['left','right']):
        v = obs['poses_xyzw'][i]
        r = Rotation.from_quat(v[3:]).as_matrix()
        xyz = A @ (np.array(v[:3]) + r @ np.array([.09343,0,0])) + T
        q = Rotation.from_matrix(A @ r @ C).as_quat()[[3,0,1,2]]
        # Camera geometry probe only: don't invent the missing CAD lookup table.
        goals[side] = {'position_m':xyz.tolist(), 'quaternion_wxyz':q.tolist(),
                      'gripper_open_fraction':1.0}
    return goals

def predict(observation, step, episode):
    ROOT.mkdir(parents=True,exist_ok=True)
    frame = min(step*3, 87)
    observed_frame = max(0, frame-3)
    expected = target(observed_frame)
    errors = {}
    for side in ['left','right']:
        actual = observation['robots'][side]['tool_pose']
        q=np.array(actual['quaternion_wxyz']); q /= np.linalg.norm(q)
        errors[side] = {
          'position_error_m':float(np.linalg.norm(np.array(actual['position_m'])-expected[side]['position_m'])),
          'orientation_error_deg':float(np.degrees(2*np.arccos(np.clip(abs(q@expected[side]['quaternion_wxyz']),0,1))))}
    record={'step':step,'command_source_frame':frame,'previous_target_source_frame':observed_frame,
            'errors':errors,'robots':observation['robots'],
            'purpose':'camera correspondence only; grippers held open, not grasp evaluation'}
    with (ROOT/'correspondence.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    if step in [1,6,11,16,21,26]:
        for name,rgb in observation['images'].items():
            Image.fromarray(np.asarray(rgb)).save(ROOT/f'{name}_source_{observed_frame:03d}.png')
    return {'action_dt_s':.1,'waypoints':[target(frame)],'execute_steps':1}
