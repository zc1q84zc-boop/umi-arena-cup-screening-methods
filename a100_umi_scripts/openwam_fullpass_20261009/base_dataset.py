"""Paired-recording OpenWAM samples with official observations and phase prompts."""
from functools import lru_cache
import json
from pathlib import Path

import av
import numpy as np
import torch

from contract import CAMERAS, image_pair, normalize


class MergedCupDataset(torch.utils.data.Dataset):
    def __init__(self, prepared_dir, height=256, width=640, split="train", balance=True):
        self.root = Path(prepared_dir)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        assert self.manifest["version"] == "openwam_merged_official_yubi_v1"
        assert (self.manifest["state_dim"],self.manifest["action_dim"],self.manifest["control_fps"]) == (9,16,10)
        self.norms = json.loads((self.root/"normalization.json").read_text())
        self.normalization_stats_path = str(self.root/"normalization_stats.npy")
        self.height, self.width, self.split = int(height), int(width), split
        assert (self.height,self.width)==(256,640)
        groups = [[],[]]
        for uuid, row in sorted(self.manifest["records"].items()):
            if row["split"] != split: continue
            data = self.record(uuid)
            for frame,p in enumerate(data["phase"]): groups[int(p)].append((uuid,frame))
        if not all(groups): raise ValueError("both phases are required in this split")
        self.unique_windows = sum(map(len,groups))
        if balance and split=="train":
            self.samples = [group[i%len(group)] for i in range(max(map(len,groups))) for group in groups]
        else: self.samples = sorted(groups[0]+groups[1])

    @classmethod
    def from_config(cls, config, split="train"):
        return cls(config.prepared_dir,config.height,config.width,split)

    def __len__(self): return len(self.samples)

    @lru_cache(maxsize=4)
    def record(self, uuid):
        with np.load(self.root/"records"/f"{uuid}.npz") as f:
            return {key:f[key] for key in f.files}

    def video_targets(self, uuid, indices):
        meta = self.manifest["records"][uuid]
        data = self.record(uuid)
        pairs = []
        # A window is clipped to one prompt, so all requested frames use one
        # source episode; the merged record retains the exact phase boundary.
        phase = int(data["phase"][indices[0]])
        local = data["raw_frame"][indices] - (meta["phase_boundary_raw_frame"] if phase else 0)
        for camera in CAMERAS:
            ref = meta["videos"][phase][camera]
            times = ref["start_s"] + local / 30
            decoded = {}
            with av.open(ref["path"]) as container:
                stream = container.streams.video[0]
                container.seek(int(times[0]/stream.time_base),stream=stream,backward=True)
                for frame in container.decode(stream):
                    if frame.time is None: continue
                    for n,target in enumerate(times):
                        if n not in decoded and abs(frame.time-target) <= 1/60:
                            decoded[n] = np.asarray(frame.to_image().convert("RGB"))
                    if len(decoded)==len(times): break
                    if frame.time > times[-1]+1/30: break
            if len(decoded)!=len(times): raise ValueError(f"missing wrist video frames: {uuid} {camera} {times.tolist()}")
            pairs.append([decoded[i] for i in range(len(times))])
        return [image_pair(a,b,self.width,self.height) for a,b in zip(*pairs)]

    def __getitem__(self,index):
        uuid,frame = self.samples[index]
        data = self.record(uuid)
        phase = int(data["phase"][frame])
        end = int(np.flatnonzero(data["phase"]==phase)[-1])+1
        action_indices = np.minimum(frame+np.arange(32),end-1)
        video_indices = np.minimum(frame+4*np.arange(9),end-1)
        valid_action = frame+np.arange(32)<end
        valid_video = frame+4*np.arange(9)<end
        action, proprio = np.zeros((32,80),np.float32), np.zeros((1,80),np.float32)
        action[:,:16] = normalize(data["action"][action_indices],self.norms["action"])
        proprio[0,:9] = normalize(data["state"][frame],self.norms["state"])
        amask, pmask = np.zeros_like(action,dtype=bool),np.zeros_like(proprio,dtype=bool)
        amask[:,:16] = valid_action[:,None]
        pmask[:,:9] = True
        video = self.video_targets(uuid,video_indices)
        prompt = self.manifest["records"][uuid]["prompts"][phase]
        return {"video":video,"first_frame_image":[video[0]],"vace_video":None,
                "action":torch.from_numpy(action),"action_mask":torch.from_numpy(amask),
                "proprio":torch.from_numpy(proprio),"proprio_mask":torch.from_numpy(pmask),
                "video_mask":torch.from_numpy(valid_video),"prompt":prompt}
