"""CPU gate for selection, finite targets, and complete distributed sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--intersection",type=Path,required=True)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.repo))
    import numpy as np
    import torch
    from accelerate.data_loader import prepare_data_loader
    from openwam.train.utils.seeding import wire_sampler_seed
    from dataset import MergedCupDataset,CoverageSampler
    train=MergedCupDataset(args.prepared)
    val=MergedCupDataset(args.prepared,split="val",balance=False)
    manifest=train.manifest
    selected=json.loads(args.intersection.read_text())
    allowed={int(r["episode_index"]) for r in selected["episodes"]}
    train_uuids={u for u,_ in train.samples};val_uuids={u for u,_ in val.samples}
    assert not train_uuids & val_uuids
    assert len(train_uuids)==1068 and len(val_uuids)==118
    assert len(train)==162196 and train.unique_windows==160338 and len(val)==17366
    digest=hashlib.sha256();source_ids=set();negative_state_quaternions=0
    train_states=[];train_actions=[]
    for uuid,meta in sorted(manifest["records"].items()):
        assert set(meta["episode_indices"])<=allowed
        source_ids.update(meta["episode_indices"])
        path=args.prepared/"records"/(uuid+".npz")
        digest.update(uuid.encode());digest.update(hashlib.sha256(path.read_bytes()).digest())
        data=train.record(uuid)
        assert data["state"].shape==(meta["control_frames"],9)
        assert data["action"].shape==(meta["control_frames"],16)
        assert np.isfinite(data["state"]).all() and np.isfinite(data["action"]).all()
        assert np.all(np.diff(data["raw_frame"])>0)
        assert np.all(np.diff(data["phase"])>=0) and set(data["phase"])=={0,1}
        for lo,hi in [(3,7),(10,14)]:
            assert np.allclose(np.linalg.norm(data["action"][:,lo:hi],axis=1),1,atol=2e-6)
        assert np.allclose(np.linalg.norm(data["state"][:,3:7],axis=1),1,atol=2e-6)
        negative_state_quaternions+=int((data["state"][:,6]<0).sum())
        for phase in (0,1):
            for ref in meta["videos"][phase].values():assert Path(ref["path"]).is_file()
        if meta["split"]=="train":
            train_states.append(data["state"]);train_actions.append(data["action"])
    assert len(source_ids)==2372 and len(manifest["records"])==1186
    # Norms were computed before phase balancing, on training UUIDs only.
    for kind,values,slices in [("state",np.concatenate(train_states),[(3,7)]),
                               ("action",np.concatenate(train_actions),[(3,7),(10,14)])]:
        for bound,op in [("min",np.min),("max",np.max)]:
            computed=op(values,axis=0)
            for lo,hi in slices:computed[lo:hi]=-1 if bound=="min" else 1
            assert np.allclose(computed,train.norms[kind][bound],atol=1e-6)

    class IDs(torch.utils.data.Dataset):
        def __len__(self):return len(train)
        def __getitem__(self,index):return index

    sampling=[]
    for batch,accum in [(8,2),(16,1),(4,4),(2,8)]:
        sampler=CoverageSampler(len(train),batch,2,accum)
        ranks=[]
        for rank in (0,1):
            loader=torch.utils.data.DataLoader(IDs(),batch_size=batch,
                sampler=CoverageSampler(len(train),batch,2,accum),collate_fn=list,
                generator=torch.Generator().manual_seed(42+rank))
            wire_sampler_seed(loader,42,rank=rank)
            ranks.append(prepare_data_loader(loader,num_processes=2,process_index=rank,
                device=torch.device("cpu"),put_on_device=False,even_batches=True))
        actual=[index for pair in zip(*ranks) for rank_batch in pair for index in rank_batch]
        assert actual==sampler.order()
        assert len(set(actual))==len(train)
        assert len({train.samples[i] for i in actual})==train.unique_windows
        sampling.append({"world_size":2,"batch":batch,"accum":accum,"effective_batch":32,
            "slots":len(actual),"tail_padding":len(actual)-len(train),
            "micro_steps_per_rank":len(actual)//(batch*2),"optimizer_updates":len(actual)//32,
            "all_balanced_entries_covered":True,"all_unique_windows_covered":True,
            "prepared_cross_rank_order_matches_plan":True})
    # Actual RGB decoding and loss masks for both phase tails.
    examples=[]
    for phase in (0,1):
        index=next(i for i,(u,f) in enumerate(train.samples) if train.record(u)["phase"][f]==phase)
        sample=train[index];uuid,_=train.samples[index]
        tail_frame=int(np.flatnonzero(train.record(uuid)["phase"]==phase)[-1])
        tail_index=next(i for i,pair in enumerate(train.samples) if pair==(uuid,tail_frame))
        tail=train[tail_index]
        assert sample["video"][0].size==(640,256) and sample["first_frame_image"][0] is sample["video"][0]
        assert sample["proprio_mask"][:,:9].all() and not sample["proprio_mask"][:,9:].any()
        assert tail["action_mask"].sum()==16 and tail["video_mask"].sum()==1
        assert sample["_coverage_index"]==index
        examples.append({"phase":phase,"prompt":sample["prompt"],"tail_loss_steps":1})
    result={"status":"ok","training_records":1068,"validation_records":118,
        "source_episodes":len(source_ids),"training_unique_windows":train.unique_windows,
        "balanced_entries":len(train),"validation_windows":len(val),"source_npz_combined_sha256":digest.hexdigest(),
        "manifest_sha256":hashlib.sha256((args.prepared/"manifest.json").read_bytes()).hexdigest(),
        "normalization_sha256":hashlib.sha256((args.prepared/"normalization.json").read_bytes()).hexdigest(),
        "training_only_norm_bounds_verified":True,"state_quaternion_canonicalization_matches_saved_contract":negative_state_quaternions==0,
        "sampling":sampling,"phase_examples":examples}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
