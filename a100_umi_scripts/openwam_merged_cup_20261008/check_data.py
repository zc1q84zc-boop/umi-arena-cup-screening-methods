#!/usr/bin/env python3
"""CPU checks for official-input preprocessing, merged phases, and control timing."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from contract import INTERHAND, actions_to_wire, observation
from dataset import MergedCupDataset
from prepare import compose_deltas


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",type=Path,required=True)
    args=parser.parse_args()
    left=np.full((480,640,3),(255,0,0),dtype=np.uint8)
    right=np.full((480,640,3),(0,0,255),dtype=np.uint8)
    obs={"observation.image.left":left,"observation.image.right":right,
         INTERHAND:np.array([.2,.1,.3,0,0,0,1]),"observation.joint_states":np.array([.2,.4]),"prompt":"first"}
    image,state,prompt=observation(obs)
    assert state.shape==(9,) and prompt=="first"
    assert np.array_equal(np.asarray(image)[128,160],[255,0,0])
    assert np.array_equal(np.asarray(image)[128,480],[0,0,255])
    obs["prompt"]="second"
    assert observation(obs)[2]=="second"
    obs["observation.image.center"]="unused"
    assert observation(obs)[1].shape==(9,)
    # A translated step followed by a rotated local step must not be added
    # directly in the world frame.
    quarter=Rotation.from_euler("z",90,degrees=True).as_quat()
    delta=compose_deltas([[1,0,0,*quarter],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1]])
    assert np.allclose(delta[:3],[1,2,0],atol=1e-6)
    assert np.allclose(Rotation.from_quat(delta[3:]).as_matrix(),Rotation.from_quat(quarter).as_matrix(),atol=1e-6)
    norms={"min":[-1]*16,"max":[1]*16}
    action=np.zeros((32,80),np.float32);action[:,6]=1;action[:,13]=1
    assert actions_to_wire(action,norms).shape==(32,16)
    train=MergedCupDataset(args.prepared)
    val=MergedCupDataset(args.prepared,split="val",balance=False)
    assert not ({u for u,_ in train.samples}&{u for u,_ in val.samples})
    examples=[]
    for phase in (0,1):
        idx=next(i for i,(u,f) in enumerate(train.samples) if train.record(u)["phase"][f]==phase)
        sample=train[idx]
        assert sample["action"].shape==(32,80) and sample["proprio"].shape==(1,80)
        assert not sample["action_mask"][:,16:].any() and not sample["proprio_mask"][:,9:].any()
        assert sample["proprio_mask"][:,:9].all() and np.isfinite(sample["action"].numpy()).all()
        assert sample["first_frame_image"][0] is sample["video"][0]
        assert sample["video"][0].size==(640,256)
        u,_=train.samples[idx]
        end=int(np.flatnonzero(train.record(u)["phase"]==phase)[-1])
        near=next(i for i,pair in enumerate(train.samples) if pair==(u,end))
        tail=train[near]
        assert tail["action_mask"][:,:16].sum().item()==16
        assert tail["video_mask"].sum().item()==1
        examples.append({"uuid":u,"phase":phase,"prompt":sample["prompt"],"tail_supervised_steps":1})
    result={"status":"ok","training_unique_windows":train.unique_windows,
            "training_balanced_windows":len(train),"validation_windows":len(val),
            "state_dim":9,"action_dim":16,"control_fps":10,"examples":examples}
    (args.prepared/"data_checks.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
