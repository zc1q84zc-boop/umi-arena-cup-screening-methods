#!/usr/bin/env python3
"""OpenWAM full-pass training with actual cross-rank sample coverage proof."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ("repo","prepared","recipe","foundation","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--mode",choices=["smoke","train"],required=True)
    p.add_argument("--batch",type=int,required=True)
    p.add_argument("--accum",type=int,required=True)
    p.add_argument("--save-every-updates",type=int,default=1250)
    args=p.parse_args()
    world=int(os.environ.get("WORLD_SIZE",1))
    assert min(args.batch,args.accum)>0 and args.batch*args.accum*world==32
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.repo))
    import setuptools
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from dataset import MergedCupDataset, CoverageSampler
    from openwam.train.openwam_trainer import OpenWAMTrainer
    ds=MergedCupDataset(args.prepared)
    assert len(set(ds.samples))==ds.unique_windows
    sampler=CoverageSampler(len(ds),args.batch,world,args.accum)
    micro_steps=sampler.slots//(args.batch*world) if args.mode=="train" else 2*args.accum
    cfg=OmegaConf.load(args.recipe)
    cfg.model.freeze=["video_backbone"]
    cfg.dataloader.type="umi_cup_merged_official"
    cfg.dataloader.prepared_dir=str(args.prepared)
    cfg.dataloader.dataset_dir=str(args.prepared)
    cfg.dataloader.height=256; cfg.dataloader.width=640
    cfg.dataloader.unify_action_map=["0-15"]
    t=cfg.training
    t.batch_size=args.batch; t.gradient_accumulation_steps=args.accum
    t.max_steps=micro_steps; t.num_epochs=1 if args.mode=="train" else None
    t.finetune_ckpt_path=str(args.foundation); t.resume_ckpt_path=None
    t.mixed_precision="bf16"; t.zero_stage=2
    t.offload_optimizer_device="cpu"; t.initialize_model_on_cpu=True
    t.use_gradient_checkpointing=True; t.use_gradient_checkpointing_offload=True
    t.learning_rate=1e-5; t.lr_scheduler=None; t.dataset_num_workers=2
    t.save_steps=micro_steps if args.mode=="smoke" else args.save_every_updates*args.accum
    t.save_full_states_for_resume=False; t.keep_last_k_ckpts=5
    t.output_path=str(args.output/"checkpoints")
    cfg.project.output_dir=str(args.output/"checkpoints")
    cfg.project.wandb.project=None; cfg.project.seed=42
    spec=importlib.util.spec_from_file_location("upstream_entry",args.repo/"scripts/train.py")
    entry=importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    started=time.monotonic(); records=[]
    expected_order=sampler.order()
    unique_pairs=sorted(set(ds.samples)); unique_ids={pair:i for i,pair in enumerate(unique_pairs)}
    physical=np.array([unique_ids[pair] for pair in ds.samples],dtype=np.int32)
    index_counts=np.zeros(len(ds),dtype=np.int32)
    unique_counts=np.zeros(ds.unique_windows,dtype=np.int32)
    consumed=0; order_digest=hashlib.sha256()
    result={"mode":args.mode,"status":"running","world_size":world,
        "micro_batch":args.batch,"accum":args.accum,"effective_batch":32,
        "training_balanced_entries":len(ds),"training_unique_windows":ds.unique_windows,
        "planned_slots":sampler.slots,"tail_padding_entries":sampler.slots-len(ds),
        "planned_micro_steps":micro_steps,"planned_optimizer_updates":micro_steps//args.accum,
        "sampling":"seed-42 shuffled complete pass; tail-only padding; no replacement before tail",
        "time_cap_hours":None,"foundation":str(args.foundation),"prepared":str(args.prepared)}

    def save_json(name,data):
        target=args.output/name; temp=target.with_suffix(target.suffix+".tmp")
        temp.write_text(json.dumps(data,indent=2)+"\n");temp.replace(target)

    def coverage(step,final=False):
        row={"micro_step":step,"optimizer_updates":step//args.accum,
             "consumed_slots":consumed,"planned_slots":sampler.slots,
             "balanced_entries_seen":int(np.count_nonzero(index_counts)),"balanced_entries_total":len(ds),
             "unique_windows_seen":int(np.count_nonzero(unique_counts)),"unique_windows_total":ds.unique_windows,
             "all_training_windows_seen":bool(np.all(unique_counts>0)),
             "all_balanced_entries_seen":bool(np.all(index_counts>0)),
             "actual_order_matches_deterministic_plan":True,"order_sha256":order_digest.hexdigest(),
             "full_pass_completed":bool(final and consumed==sampler.slots and np.all(index_counts>0)),
             "updated_unix":time.time()}
        save_json("coverage.json",row)
        if final:
            np.savez_compressed(args.output/"coverage_counts.npz",balanced_entry_counts=index_counts,
                                unique_window_counts=unique_counts,balanced_to_unique=physical)
        return row

    class AuditedTrainer(OpenWAMTrainer):
        def build_dataloader(self,batch_size):
            from openwam.train.utils.seeding import dataloader_worker_init_fn, make_dataloader_generator
            return torch.utils.data.DataLoader(self.dataset,batch_size=batch_size,sampler=sampler,
                drop_last=False,num_workers=2,collate_fn=list,pin_memory=True,
                generator=make_dataloader_generator(42,rank=self._rank),worker_init_fn=dataloader_worker_init_fn)

        def prepare_accelerate(self,*a):
            out=super().prepare_accelerate(*a)
            loader=out[1]
            assert len(loader)==sampler.slots//(args.batch*world),(len(loader),sampler.slots)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            self.last_end=time.monotonic();result["setup_seconds"]=self.last_end-started
            if self.accelerator.is_main_process:
                OmegaConf.save(cfg,args.output/"requested_config.yaml")
                assets=args.output/"official_contract"
                assets.mkdir(exist_ok=False)
                for source in (Path(__file__).with_name("contract.py"),
                               args.prepared/"normalization.json",
                               args.prepared/"normalization_stats.npy"):
                    shutil.copy2(source,assets/source.name)
                (assets/"schema.json").write_text(json.dumps({
                    "camera_keys":["observation.image.left","observation.image.right"],
                    "camera_layout":"left then right, 640x256 total",
                    "state":"inter-hand pose7 + finger joints2, xyzw canonical quaternion",
                    "action":"left pose7 + right pose7 + finger joints2, 10Hz body-frame future SE3 delta",
                    "state_dim":9,"action_dim":16,"action_horizon":32,
                    "phase_prompts":next(iter(ds.manifest["records"].values()))["prompts"]
                },indent=2)+"\n")
                save_json("result.json",result)
            return out

        def compute_loss(self,batch):
            if not isinstance(batch,list):batch=[batch]
            self.batch_indices=[sample["_coverage_index"] for sample in batch]
            clean=[{k:v for k,v in sample.items() if k!="_coverage_index"} for sample in batch]
            inputs=self.architecture.prepare_inputs(clean)
            assert inputs["actions"].shape==(len(batch),32,80)
            assert inputs["proprio"].shape==(len(batch),80)
            value=self.architecture.compute_loss(**inputs,lambda_video=self.lambda_video,lambda_action=self.lambda_action)
            return {"total":value["loss"],"video":value.get("loss_video",torch.tensor(0.)),
                    "action":value.get("loss_action",torch.tensor(0.))}

        def log_step(self,**kw):
            nonlocal consumed
            torch.cuda.synchronize();now=time.monotonic()
            metrics={k:float(v) for k,v in kw["metrics"].items()}
            assert all(math.isfinite(v) for v in metrics.values()),metrics
            observed=self.accelerator.gather(torch.tensor(self.batch_indices,device=self.accelerator.device)).cpu().numpy()
            assert observed.tolist()==expected_order[consumed:consumed+len(observed)],"distributed sample order differs from plan"
            consumed+=len(observed)
            np.add.at(index_counts,observed,1);np.add.at(unique_counts,physical[observed],1)
            order_digest.update(observed.astype("<i4").tobytes())
            row={"micro_step":kw["global_step"],"optimizer_step":kw["opt_step"],"samples":len(observed),
                 "seconds":now-self.last_end,"metrics":metrics,
                 "unique_windows_seen":int(np.count_nonzero(unique_counts)),
                 "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                 "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20}
            records.append(row)
            if self.accelerator.is_main_process:
                with (args.output/"steps.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
                if row["micro_step"]<=3 or row["micro_step"]%100==0:
                    print("AUDIT_STEP "+json.dumps(row),flush=True);coverage(row["micro_step"])
            self.last_end=now;kw["pbar"].update(1)

    accelerator=None
    try:
        accelerator=entry._build_accelerator(cfg)
        assert accelerator.num_processes==world
        trainer=AuditedTrainer(cfg,accelerator=accelerator,dataset=ds)
        result["trainable_parameters"]=sum(x.numel() for x in trainer.architecture.parameters() if x.requires_grad)
        trainer.train()
        if args.mode=="train":
            assert consumed==sampler.slots and np.all(index_counts>0) and np.all(unique_counts>0)
        seconds=sum(r["seconds"] for r in records)
        result.update(status="ok",micro_steps=len(records),optimizer_updates=records[-1]["optimizer_step"],
            consumed_slots=consumed,unique_windows_seen=int(np.count_nonzero(unique_counts)),
            samples_per_second=consumed/seconds,median_micro_step_seconds=statistics.median(r["seconds"] for r in records),
            peak_reserved_mib=max(r["peak_reserved_mib"] for r in records))
        if accelerator.is_main_process:coverage(len(records),final=True)
    except BaseException as exc:
        result.update(status="error",error=repr(exc));raise
    finally:
        result["wall_seconds"]=time.monotonic()-started
        if int(os.environ.get("RANK",0))==0:
            save_json("result.json",result);print("RESULT "+json.dumps(result),flush=True)
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=="__main__":main()
