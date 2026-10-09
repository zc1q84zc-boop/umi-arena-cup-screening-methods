"""Shared preprocessing for the official YUBI observation/action contract."""
import numpy as np
from PIL import Image, ImageOps

INTERHAND = "observation.pose.left_hand_root_to_right_hand_root.absolute"
CAMERAS = ["observation.image.left", "observation.image.right"]


def unit_quat(q):
    q = np.asarray(q, dtype=np.float32).copy()
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-6) or not np.isfinite(q).all():
        raise ValueError("invalid xyzw quaternion")
    q /= norm
    return q * np.where(q[..., 3:] < 0, -1, 1)


def image_pair(left, right, width=640, height=256):
    """Fixed left/right tiling; no center camera or simulated absolute pose."""
    canvas = Image.new("RGB", (width, height))
    for side, arr in enumerate((left, right)):
        arr = np.asarray(arr)
        if arr.shape != (480, 640, 3) or arr.dtype != np.uint8:
            raise ValueError("expected native 480x640 HWC uint8 RGB wrist image")
        image = ImageOps.pad(Image.fromarray(arr), (width//2, height),
                             method=Image.Resampling.BILINEAR, color=(0,0,0))
        canvas.paste(image, (side*(width//2), 0))
    return canvas


def observation(obs):
    pose = np.asarray(obs[INTERHAND], dtype=np.float32)
    joints = np.asarray(obs["observation.joint_states"], dtype=np.float32)
    if pose.shape != (7,) or joints.shape != (2,) or not np.isfinite(pose).all() or not np.isfinite(joints).all():
        raise ValueError("official state requires inter-hand pose(7) + finger joints(2)")
    if not isinstance(obs["prompt"], str): raise ValueError("prompt must be a string")
    state = np.r_[pose, joints].astype(np.float32)
    state[3:7] = unit_quat(state[3:7])
    return image_pair(obs[CAMERAS[0]], obs[CAMERAS[1]]), state, obs["prompt"]


def normalize(values, stats):
    lo, hi = [np.asarray(stats[k], dtype=np.float32) for k in ("min","max")]
    return np.clip(2*(values-lo)/np.maximum(hi-lo,1e-6)-1,-1,1).astype(np.float32)


def actions_to_wire(normalized, stats):
    values = np.asarray(normalized, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 16 or values.shape[1] not in (16,80):
        raise ValueError("expected at least 16 action rows, with 16 or 80 columns")
    lo, hi = [np.asarray(stats[k], dtype=np.float32) for k in ("min","max")]
    action = (values[:,:16]+1)/2*np.maximum(hi-lo,1e-6)+lo
    action[:,3:7] = unit_quat(action[:,3:7])
    action[:,10:14] = unit_quat(action[:,10:14])
    if not np.isfinite(action).all(): raise ValueError("non-finite action")
    return action.astype(np.float32)
