"""Bidirectional rigid calibration shared by online policies and replay.

T_world_tool = T_world_source @ T_source_hand @ T_hand_tool.
No episode anchors, reflection matrices, image flips, or forced grasp timing.
Profiles must be explicitly verified before use; this module cannot estimate
missing extrinsics from an episode-specific visual correction.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def mirror_sagittal_rotation(rotation):
    """Mirror between anatomical arms about XZ, in matching local bases.

    A reflection alone is not a robot rotation; conjugation preserves SO(3).
    Thus world Rx(+17) becomes Rx(-17), while local Ry(+22) stays Ry(+22).
    """
    r = np.asarray(rotation, float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("expected proper rotation, not a reflection")
    s = np.diag([1., -1., 1.])
    return s @ r @ s


def matrix(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("invalid finite xyz / wxyz pose")
    if np.linalg.norm(q) < 1e-8:
        raise ValueError("zero quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                 [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                 [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]
    t[:3, 3] = p
    return t


def pose(t):
    # Eigenvector method stays stable at 180 degrees.
    r = t[:3, :3]
    k = np.array([[r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
                  [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
                  [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
                  [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)]]) / 3
    _, vectors = np.linalg.eigh(k)
    xyzw = vectors[:, -1]
    q = xyzw[[3,0,1,2]]
    return t[:3,3].copy(), q if q[0] >= 0 else -q


class Calibration:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        d = self.data
        if set(d) != {"schema_version", "id", "verified", "world_from_source", "hand_to_tool", "gripper", "camera_files_sha256"}:
            raise ValueError("profile schema excludes episode-specific corrections")
        if d["schema_version"] != 1 or d["verified"] is not True:
            raise ValueError("calibration has not been verified; no silent identity fallback")
        def transform(v):
            if not isinstance(v, dict) or set(v) != {"position_m", "quaternion_wxyz"}:
                raise ValueError("missing explicit rigid transform")
            return matrix(v["position_m"], v["quaternion_wxyz"])
        self.world = transform(d["world_from_source"])
        if set(d["hand_to_tool"]) != {"left", "right"}:
            raise ValueError("both anatomical arms need tool transforms")
        self.tool = {s: transform(d["hand_to_tool"][s]) for s in ("left", "right")}
        g = d["gripper"]
        self.source_open, self.source_closed = float(g["source_open_rad"]), float(g["source_closed_rad"])
        self.sim_open, self.sim_closed = float(g["sim_open_fraction"]), float(g["sim_closed_fraction"])
        if not np.isfinite([self.source_open,self.source_closed,self.sim_open,self.sim_closed]).all() or self.source_open == self.source_closed or not 0 <= self.sim_closed < self.sim_open <= 1:
            raise ValueError("invalid gripper endpoints")
        if not d["camera_files_sha256"]:
            raise ValueError("camera configuration hashes required")
        for filename, expected in d["camera_files_sha256"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"camera configuration changed: {filename}")

    def tool_to_hand(self, side, p, q):
        return pose(np.linalg.inv(self.world) @ matrix(p,q) @ np.linalg.inv(self.tool[side]))

    def hand_to_world_tool(self, side, p, q):
        return pose(self.world @ matrix(p,q) @ self.tool[side])

    def source_gripper(self, fraction):
        a = np.clip((fraction-self.sim_closed)/(self.sim_open-self.sim_closed), 0, 1)
        return float(self.source_closed + a*(self.source_open-self.source_closed))

    def sim_gripper(self, source):
        a = np.clip((source-self.source_closed)/(self.source_open-self.source_closed), 0, 1)
        return float(self.sim_closed + a*(self.sim_open-self.sim_closed))

    def audit(self):
        return {"id": self.data["id"], "sha256": self.digest, "pose_frame": "source_hand",
                "camera_files_sha256": self.data["camera_files_sha256"]}


class Reference259632:
    """User-selected simulation reference; not measured camera calibration."""
    def __init__(self):
        self.gripper = {'source_closed_rad': 0.0, 'source_open_rad': .78,
                        'sim_closed_fraction': 0.0, 'sim_open_fraction': 1.0,
                        'direction_verified': True, 'aperture_scale_measured': False}
        self.a = np.array([[0.,1,0],[-1,0,0],[0,0,1]])
        self.c = np.array([[0.,0,1],[1,0,0],[0,1,0]])
        self.t = np.array([-.209307083410,.202121290786,.749006156995])
        self.offset = np.array([.09343,0,0])
    def hand_to_world_tool(self, side, p, q):
        r=matrix(p,q)[:3,:3]; out=np.eye(4)
        out[:3,:3]=self.a@r@self.c
        out[:3,3]=self.a@(np.asarray(p)+r@self.offset)+self.t
        return pose(out)
    def tool_to_hand(self, side, p, q):
        r=self.a.T@matrix(p,q)[:3,:3]@self.c.T; out=np.eye(4)
        out[:3,:3]=r; out[:3,3]=self.a.T@(np.asarray(p)-self.t)-r@self.offset
        return pose(out)
    def source_gripper(self, fraction):
        return float(np.clip(fraction,0,1)*self.gripper['source_open_rad'])
    def sim_gripper(self, source):
        return float(np.clip(source/self.gripper['source_open_rad'],0,1))
    def audit(self):
        return {'id':'reference_259632_v1','measured':False,'pose_frame':'source_hand',
                'gripper_mapping':'existing approximate 0..0.78, CAD lookup not supplied',
                'camera_status':'approximate; multi-frame mismatch remains'}


class MirroredReplayPrior:
    """User-authorized simulation prior, NOT measured calibration.

    Matches the fixed replay position and orientation maps independently.
    World-roll correction affects orientation only, as in the replay; it must
    not rotate the table or introduce a second height offset.
    """
    def __init__(self):
        raw = Path(__file__).with_name('calibration_mirrored_provisional.json').read_bytes()
        self.profile_sha256 = hashlib.sha256(raw).hexdigest()
        profile = json.loads(raw)
        self.gripper = profile['gripper']
        if profile['id'] != 'mirrored_replay_prior_20260929' or profile['measured'] is not False:
            raise ValueError('unexpected provisional profile')
        if profile['left'] != {'world_x_roll_deg': -profile['right']['world_x_roll_deg'],
                              'tool_local_y_deg': profile['right']['tool_local_y_deg']}:
            raise ValueError('left profile is not the authorized sagittal mirror')
        self.a = matrix([0,0,0], [2**-.5,0,0,-2**-.5])[:3,:3]
        # Only the authorized shared height; no episode-specific XY placement.
        self.t = np.array([0.,0.,profile['shared_source_to_world_height_offset_m']])
        source0 = matrix([0,0,0],[.42073740109757896,-.1781199279793403,.08241493135351714,.8856980917131329])[:3,:3]
        anchor = matrix([0,0,0],[.23923011258380583,.35578242624599926,.7522329356729917,.5003333177956902])[:3,:3]
        c = (self.a @ source0).T @ anchor
        self.c = {'right': c, 'left': mirror_sagittal_rotation(c)}
        self.offset = np.array([.08927,0,0])
        self.angles = [profile['right']['world_x_roll_deg'],profile['right']['tool_local_y_deg']]
        a,b=np.deg2rad(self.angles)/2
        rx=matrix([0,0,0],[np.cos(a),np.sin(a),0,0])[:3,:3]
        self.w={'right':rx,'left':mirror_sagittal_rotation(rx)}
        self.y=matrix([0,0,0],[np.cos(b),0,np.sin(b),0])[:3,:3]

    def tool_to_hand(self, side, p, q):
        rt=matrix(p,q)[:3,:3]
        rh=self.a.T @ self.w[side].T @ rt @ self.y.T @ self.c[side].T
        ph=self.a.T @ (np.asarray(p)-self.t)-rh @ self.offset
        t=np.eye(4);t[:3,:3]=rh;t[:3,3]=ph
        return pose(t)

    def hand_to_world_tool(self, side, p, q):
        rh=matrix(p,q)[:3,:3]
        t=np.eye(4)
        t[:3,3]=self.a @ (np.asarray(p)+rh @ self.offset)+self.t
        t[:3,:3]=self.w[side] @ self.a @ rh @ self.c[side] @ self.y
        return pose(t)

    def source_gripper(self, fraction):
        return float(self.gripper['source_closed_rad'] + np.clip(fraction,0,1)*
                     (self.gripper['source_open_rad']-self.gripper['source_closed_rad']))

    def sim_gripper(self, source):
        return float(np.clip((source-self.gripper['source_closed_rad'])/
                    (self.gripper['source_open_rad']-self.gripper['source_closed_rad']),0,1))

    def audit(self):
        return {'id':'mirrored_replay_prior_20260929','measured':False,
                'profile_sha256':self.profile_sha256,
                'gripper':self.gripper,
                'pose_frame':'source_hand','height_offset_m':float(self.t[2]),
                'world_roll_deg':{'right':self.angles[0],'left':-self.angles[0]},'local_y_deg':self.angles[1]}
