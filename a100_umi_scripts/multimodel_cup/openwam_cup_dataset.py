"""Causal OpenWAM-Alpha reader for the verified UMI cup selection."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageOps
import torch


TASK = "Place the cup on the plate, then put it back to its original position"


class CupDataset(torch.utils.data.Dataset):
    def __init__(self, prepared_dir: str | Path, height: int = 384, width: int = 320):
        self.prepared_dir = Path(prepared_dir)
        self.manifest = json.loads((self.prepared_dir / "manifest.json").read_text())
        self.norms = json.loads((self.prepared_dir / "normalization.json").read_text())
        self.normalization_stats_path = str(self.prepared_dir / "normalization_stats.npy")
        if (self.manifest["total_episodes"], self.manifest["total_frames"]) != (3423, 746914):
            raise ValueError("OpenWAM dataset is not the audited clean cup selection")
        if (self.manifest.get("video_target_frames"), self.manifest.get("action_horizon")) != (33, 32):
            raise ValueError("OpenWAM video/action alignment is not the audited t..t+32 recipe")
        if self.manifest.get("active_action_slots") != [*range(10), *range(34, 44)]:
            raise ValueError("OpenWAM bimanual slot map is not left0-9/right34-43")
        if not Path(self.normalization_stats_path).is_file():
            raise FileNotFoundError(self.normalization_stats_path)
        self.height, self.width = int(height), int(width)
        self.action_dim = 80
        self.samples = [(episode_id, frame)
                        for episode_id in self.manifest["train_episodes"]
                        for frame in range(self.manifest["episodes"][str(episode_id)]["frames"] - 32)]
        if not self.samples:
            raise ValueError("no aligned 33-frame/32-action windows")

    @classmethod
    def from_config(cls, config, split: str = "train") -> "CupDataset":
        if split != "train":
            raise ValueError("this qualitative comparison has no independent validation split")
        if int(config.num_frames) != 33 or int(config.video_stride) != 4:
            raise ValueError("OpenWAM-Alpha requires 33-frame/stride-4 video history")
        if config.normalize_mode != "min-max" or not config.unify_action:
            raise ValueError("unexpected OpenWAM normalization/action configuration")
        return cls(config.prepared_dir, config.height, config.width)

    def __len__(self) -> int:
        return len(self.samples)

    @lru_cache(maxsize=3)
    def episode(self, episode_id: int) -> dict[str, np.ndarray]:
        with np.load(self.prepared_dir / f"episode_{episode_id:06d}.npz") as file:
            return {key: file[key] for key in file.files}

    def normalize(self, values: np.ndarray, key: str) -> np.ndarray:
        record = self.norms[key]
        low = np.asarray(record["min"], dtype=np.float32)
        high = np.asarray(record["max"], dtype=np.float32)
        normalized = np.clip(2 * (values - low) / np.maximum(high - low, 1e-6) - 1, -1, 1)
        normalized[..., 3:9] = values[..., 3:9]
        normalized[..., 13:19] = values[..., 13:19]
        return normalized.astype(np.float32)

    def video_targets(self, episode_id: int, frame_index: int) -> list:
        reference = self.manifest["episodes"][str(episode_id)]
        # The first frame is the current observation. Subsequent frames are
        # future VIDEO TARGETS aligned with the action horizon, not inputs to
        # the deployed policy at this timestep.
        targets = [reference["video_start_s"] + (frame_index + i * 4) / 30 for i in range(9)]
        frames = []
        with av.open(reference["video"]) as container:
            stream = container.streams.video[0]
            container.seek(int(targets[0] / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.time is None or frame.time < targets[len(frames)] - 1 / 60:
                    continue
                if abs(frame.time - targets[len(frames)]) > 1 / 60:
                    raise ValueError(f"video frame timing mismatch: episode {episode_id} at {targets[len(frames)]}")
                frames.append(ImageOps.pad(frame.to_image(), (self.width, self.height),
                                           method=Image.Resampling.BILINEAR, color=(0, 0, 0)))
                if len(frames) == 9:
                    break
        if len(frames) != 9:
            raise ValueError(f"missing aligned video targets: episode {episode_id} frame {frame_index}")
        return frames

    def __getitem__(self, index: int) -> dict:
        episode_id, frame = self.samples[index]
        data = self.episode(episode_id)
        poses = data["poses"]
        gripper = data["observation_gripper"]
        gripper_action = data["action_gripper"]
        from scipy.spatial.transform import Rotation

        def eef(p: np.ndarray, g: np.ndarray) -> np.ndarray:
            matrix = Rotation.from_quat(p[..., 3:].reshape(-1, 4)).as_matrix().reshape(*p.shape[:-1], 3, 3)
            output = np.empty((*p.shape[:-1], 10), dtype=np.float32)
            output[..., :3] = p[..., :3]
            output[..., 3:6] = matrix[..., :, 0]
            output[..., 6:9] = matrix[..., :, 1]
            output[..., 9] = g
            return output

        state20 = np.concatenate([eef(poses[frame, 0], gripper[frame, 0]),
                                  eef(poses[frame, 1], gripper[frame, 1])])
        future = poses[frame + 1:frame + 33]
        target20 = np.concatenate([
            eef(future[:, 0], gripper_action[frame:frame + 32, 0]),
            eef(future[:, 1], gripper_action[frame:frame + 32, 1]),
        ], axis=1)
        action = np.zeros((32, 80), dtype=np.float32)
        proprio = np.zeros((1, 80), dtype=np.float32)
        normalized_action = self.normalize(target20, "action")
        normalized_state = self.normalize(state20, "state")
        action[:, :10] = normalized_action[:, :10]
        action[:, 34:44] = normalized_action[:, 10:20]
        proprio[0, :10] = normalized_state[:10]
        proprio[0, 34:44] = normalized_state[10:20]
        action_mask = np.zeros_like(action, dtype=bool)
        proprio_mask = np.zeros_like(proprio, dtype=bool)
        action_mask[:, :10] = True
        action_mask[:, 34:44] = True
        proprio_mask[:, :10] = True
        proprio_mask[:, 34:44] = True
        video = self.video_targets(episode_id, frame)
        return {"video": video, "first_frame_image": [video[0]], "vace_video": None,
                "action": torch.from_numpy(action), "action_mask": torch.from_numpy(action_mask),
                "proprio": torch.from_numpy(proprio), "proprio_mask": torch.from_numpy(proprio_mask),
                "video_mask": torch.ones(9, dtype=torch.bool), "prompt": TASK}
