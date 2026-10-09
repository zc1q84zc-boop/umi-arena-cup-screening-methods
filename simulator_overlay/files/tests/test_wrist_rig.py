import json
from pathlib import Path
import numpy as np
import pytest
from yubi_isaac_sim_env.wrist_rig import camera_pose, follow_wrist_cameras, multiply

MOUNT = json.loads((Path(__file__).parents[1] / 'yubi_isaac_sim_env/wrist_camera_model.json').read_text())['nominal_mount_in_yubi_frame']

def test_rigid_camera_mount_rotates_and_translates_with_base():
    base = {'position_m': [1,2,3], 'quaternion_wxyz': [1,0,0,0]}
    p, q = camera_pose(base, MOUNT)
    np.testing.assert_allclose(p, np.array([1,2,3]) + MOUNT['translation_m'])
    rot = np.array([np.sqrt(.5),0,0,np.sqrt(.5)])
    p2, q2 = camera_pose({**base, 'quaternion_wxyz': rot}, MOUNT)
    t = MOUNT['translation_m']
    np.testing.assert_allclose(p2, [1-t[1], 2+t[0], 3+t[2]])
    np.testing.assert_allclose(q2, multiply(rot, q))

def test_each_camera_follows_its_own_anatomical_arm():
    class Camera:
        def set_world_pose(self, **kw): self.pose = kw
    cameras = {n: Camera() for n in ('left_wrist', 'right_wrist')}
    specs = {n: {'robot_side': n.split('_')[0], 'rigid_mount': MOUNT} for n in cameras}
    obs = {'robots': {side: {'link_poses': {'base': {'position_m': [x,0,0], 'quaternion_wxyz': [1,0,0,0]}}} for side,x in [('left',1),('right',-1)]}}
    follow_wrist_cameras(cameras, specs, obs)
    assert cameras['left_wrist'].pose['position'][0] == 1
    assert cameras['right_wrist'].pose['position'][0] == -1
    assert cameras['right_wrist'].pose['camera_axes'] == 'usd'

def test_invalid_base_is_rejected():
    with pytest.raises(ValueError):
        camera_pose({'position_m': [0,0,0], 'quaternion_wxyz': [0,0,0,0]}, MOUNT)
