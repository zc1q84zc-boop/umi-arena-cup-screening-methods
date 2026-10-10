"""Exact official model inputs; invert the shared CAD hand/TCP and jaw maps."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from ..umi_pose_mapping import HAND_FROM_TOOL, HAND_TCP_M
from ..umi_gripper_mapping import open_fraction_to_source_angle, source_angles_to_open_fraction
from .evaluation import rotation

IMAGE_LEFT='observation.image.left';IMAGE_RIGHT='observation.image.right'
POSE='observation.pose.left_hand_root_to_right_hand_root.absolute'
JOINTS='observation.joint_states'
KEYS={IMAGE_LEFT,IMAGE_RIGHT,POSE,JOINTS,'prompt'}


def hands(state):
    result={}
    for side in ('left','right'):
        robot=state['robots'][side]
        if robot['tool_body_name']!='yubi_tool':raise ValueError('Explicit YUBI TCP is required')
        pose=robot['tool_pose'];r=rotation(pose['quaternion_wxyz'])@HAND_FROM_TOOL.T
        p=np.asarray(pose['position_m'])-r@HAND_TCP_M
        result[side]=(p,r,float(robot['gripper_open_fraction']))
    return result


def observation(state,images,prompt):
    h=hands(state);lp,lr,lg=h['left'];rp,rr,rg=h['right']
    rel=lr.T@rr;q=Rotation.from_matrix(rel).as_quat()
    result={POSE:np.asarray([*(lr.T@(rp-lp)),*q],dtype=np.float32),
            JOINTS:np.asarray([open_fraction_to_source_angle(g) for g in (lg,rg)],dtype=np.float32),
            'prompt':str(prompt)}
    for side,key in [('left',IMAGE_LEFT),('right',IMAGE_RIGHT)]:
        image=np.asarray(images[side+'_wrist'])
        if image.shape!=(480,640,3) or image.dtype!=np.uint8:raise ValueError('Native 480x640 HWC uint8 RGB required')
        result[key]=image.copy()
    return result


def action_chunk(actions,state,adopt_rows=16):
    value=np.asarray(actions)
    if value.ndim!=2 or value.shape[0]<16 or value.shape[1]!=16 or not np.isfinite(value).all():
        raise ValueError('Official action output needs finite (N>=16,16) rows')
    if not 1<=adopt_rows<=min(16,len(value)):raise ValueError('Adopt between 1 and 16 rows')
    h=hands(state);waypoints=[]
    for row in value[:adopt_rows]:
        waypoint={}
        for side,offset,jaw in [('left',0,14),('right',7,15)]:
            p,r,g=h[side];delta=row[offset:offset+3];q=row[offset+3:offset+7]
            if np.linalg.norm(delta)>.08 or abs(np.linalg.norm(q)-1)>.05:raise ValueError('Invalid 100 ms pose delta')
            dq=Rotation.from_quat(q).as_matrix()
            if Rotation.from_matrix(dq).magnitude()>1.05:raise ValueError('Rotation exceeds 1.05 rad/100 ms')
            newp=p+r@delta;newr=r@dq
            grip=float(source_angles_to_open_fraction([row[jaw]],-.1,.7)[0])
            targetq=Rotation.from_matrix(newr@HAND_FROM_TOOL).as_quat()
            waypoint[side]=dict(position_m=(newp+newr@HAND_TCP_M).tolist(),
                                quaternion_wxyz=targetq[[3,0,1,2]].tolist(),gripper_open_fraction=grip)
            h[side]=(newp,newr,grip)
        waypoints.append(waypoint)
    return dict(action_dt_s=.1,waypoints=waypoints,execute_steps=adopt_rows)
