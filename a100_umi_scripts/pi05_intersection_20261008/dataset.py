"""Independent single-prompt episodes, using only official YUBI observations."""
from functools import lru_cache
import json
from pathlib import Path

import av
import numpy as np

CAMERAS = ["observation.image.left", "observation.image.right"]
INTERHAND = "observation.pose.left_hand_root_to_right_hand_root.absolute"


def read_image(ref, frame_index):
    target = ref["start_s"] + int(frame_index)/30
    with av.open(ref["path"]) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        container.seek(int(target/stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is None: continue
            if abs(frame.time-target) <= 1/60:
                image = frame.to_ndarray(format="rgb24")
                assert image.dtype == np.uint8 and image.shape == (480, 640, 3)
                return image
            if frame.time > target+1/30: break
    raise ValueError(f"Missing video frame at {target}: {ref}")


class CupDataset:
    def __init__(self, prepared, split="train", validation_samples=None):
        self.root = Path(prepared)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        assert self.manifest["version"] == "pi05_intersection_independent_v2"
        self.samples = [(idx, t) for idx, row in sorted(self.manifest["episodes"].items(), key=lambda x:int(x[0]))
                        if row["split"] == split for t in range(row["samples"])]
        if validation_samples is not None:
            indices = np.random.default_rng(42).choice(len(self.samples), validation_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]
        assert self.samples

    def __len__(self): return len(self.samples)

    @lru_cache(maxsize=16)
    def episode(self, idx):
        with np.load(self.root/"episodes"/f"{idx}.npz") as f:
            return {k:f[k] for k in f.files}

    def __getitem__(self, index):
        idx, t = self.samples[index]
        data, meta = self.episode(idx), self.manifest["episodes"][idx]
        action = np.repeat(data["hold"][None], 32, axis=0)
        valid = min(32, len(data["action"])-t)
        action[:valid] = data["action"][t:t+valid]
        state = data["state"][t]
        out = {c:read_image(meta["videos"][c], data["raw_frame"][t]) for c in CAMERAS}
        out.update({INTERHAND:state[:7], "observation.joint_states":state[7:],
                    "observation.pose.left_hand_root.relative":action[:,:7],
                    "observation.pose.right_hand_root.relative":action[:,7:14],
                    "action.joint_states":action[:,14:], "prompt":meta["prompt"]})
        assert action.shape == (32,16) and state.shape == (9,) and np.isfinite(action).all()
        return out
