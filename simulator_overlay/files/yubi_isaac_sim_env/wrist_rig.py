"""Rigid CAD-mounted wrist optics, driven by current PhysX body poses.

Use separate sensor prims to avoid stale USD articulation transforms in GPU
physics. This reproduces the authored mount; it is not measured hand-eye
calibration and performs no image-space flip.
"""
import math
import numpy as np


def multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw])


def camera_pose(base_pose, mount):
    q = np.asarray(base_pose['quaternion_wxyz'], dtype=float)
    p = np.asarray(base_pose['position_m'], dtype=float)
    t = np.asarray(mount['translation_m'], dtype=float)
    if q.shape != (4,) or p.shape != (3,) or t.shape != (3,) or not np.isfinite(np.r_[q,p,t]).all():
        raise ValueError('Invalid wrist/base transform')
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        raise ValueError('Zero base quaternion')
    q = q / norm
    rotated = multiply(multiply(q, np.r_[0., t]), q * [1,-1,-1,-1])[1:]
    if 'quaternion_wxyz' in mount:
        local = np.asarray(mount['quaternion_wxyz'], dtype=float)
        if local.shape != (4,) or not np.isfinite(local).all() or np.linalg.norm(local) < 1e-9:
            raise ValueError('Invalid fixed wrist mount quaternion')
        local = local / np.linalg.norm(local)
    else:
        x = math.radians(mount['rotation_x_deg']) / 2
        z = math.radians(mount['roll_about_optical_axis_deg']) / 2
        local = multiply([math.cos(x),math.sin(x),0,0], [math.cos(z),0,0,math.sin(z)])
    return p + rotated, multiply(q, local)


def follow_wrist_cameras(cameras, specs, observation):
    for name, camera in cameras.items():
        spec = specs[name]
        if 'rigid_mount' not in spec:
            continue
        side = spec['robot_side']
        base = observation['robots'][side]['link_poses']['base']
        p, q = camera_pose(base, spec['rigid_mount'])
        camera.set_world_pose(position=p, orientation=q, camera_axes='usd')
