#!/usr/bin/env python3
"""Isolated OpenWAM official-input paired-data benchmark/training entry."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ("repo","prepared","recipe","foundation","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--mode",choices=["benchmark","smoke","train"],required=True)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--accum",type=int,default=1)
    p.add_argument("--steps",type=int,default=8)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--save-every",type=int,default=1000)
    args=p.parse_args()
    if min(args.batch,args.accum,args.steps)<1 or args.steps%args.accum:
        p.error("positive batch/accum/steps and complete optimizer cycles required")
    if args.mode=="benchmark" and (args.warmup>=args.steps or args.warmup%args.accum):
        p.error("benchmark needs complete warmup cycles and timed steps")
    # Both ranks share this directory; the launcher checks it is a fresh run.
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.repo))
    import setuptools  # DeepSpeed/Python 3.12 distutils compatibility.
    import torch
    from omegaconf import OmegaConf
    from dataset import MergedCupDataset
    from openwam.train.openwam_trainer import OpenWAMTrainer
    cfg=OmegaConf.load(args.recipe)
    cfg.model.freeze=["video_backbone"]
    cfg.dataloader.type="umi_cup_merged_official"
    cfg.dataloader.prepared_dir=str(args.prepared)
    cfg.dataloader.dataset_dir=str(args.prepared)
    cfg.dataloader.height=256; cfg.dataloader.width=640
    cfg.dataloader.unify_action_map=["0-15"]
    t=cfg.training
    t.batch_size=args.batch; t.gradient_accumulation_steps=args.accum
    t.max_steps=args.steps; t.num_epochs=None
    t.finetune_ckpt_path=str(args.foundation); t.resume_ckpt_path=None
    t.mixed_precision="bf16"; t.zero_stage=2
    t.offload_optimizer_device="cpu"; t.initialize_model_on_cpu=True
    t.use_gradient_checkpointing=True; t.use_gradient_checkpointing_offload=True
    t.learning_rate=1e-5; t.lr_scheduler=None
    t.dataset_num_workers=2
    t.save_steps=0 if args.mode=="benchmark" else (args.steps if args.mode=="smoke" else args.save_every)
    t.save_full_states_for_resume=False; t.keep_last_k_ckpts=3
    t.output_path=str(args.output/"checkpoints")
    cfg.project.output_dir=str(args.output/"checkpoints")
    cfg.project.wandb.project=None; cfg.project.seed=42
    if int(os.environ.get("RANK",0))==0:
        OmegaConf.save(cfg,args.output/"requested_config.yaml")
    ds=MergedCupDataset(args.prepared)
    spec=importlib.util.spec_from_file_location("upstream_entry",args.repo/"scripts/train.py")
    entry=importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    started=time.monotonic()
    records=[]
    result={"mode":args.mode,"micro_batch":args.batch,"accum":args.accum,
            "unique_training_windows":ds.unique_windows,"balanced_epoch_windows":len(ds),
            "official_state_dim":9,"official_action_dim":16,"camera_keys":["observation.image.left","observation.image.right"],
            "height":256,"width":640,"control_fps":10,"foundation":str(args.foundation),"status":"running"}
    class AuditedTrainer(OpenWAMTrainer):
        def prepare_accelerate(self,*a):
            out=super().prepare_accelerate(*a)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            self.last_end=time.monotonic()
            result["setup_seconds"]=self.last_end-started
            return out

        def compute_loss(self,batch):
            if not isinstance(batch,list): batch=[batch]
            self.cardinality=len(batch)
            inputs=self.architecture.prepare_inputs(batch)
            assert tuple(inputs["actions"].shape)==(len(batch),32,80)
            assert tuple(inputs["proprio"].shape)==(len(batch),80)
            value=self.architecture.compute_loss(**inputs,lambda_video=self.lambda_video,lambda_action=self.lambda_action)
            return {"total":value["loss"],"video":value.get("loss_video",torch.tensor(0.)),"action":value.get("loss_action",torch.tensor(0.))}

        def log_step(self,**kw):
            torch.cuda.synchronize(); now=time.monotonic()
            metrics={k:float(v) for k,v in kw["metrics"].items()}
            if not all(math.isfinite(v) for v in metrics.values()): raise RuntimeError("nonfinite training metric")
            row={"micro_step":kw["global_step"],"optimizer_step":kw["opt_step"],
                 "samples":self.cardinality*self.accelerator.num_processes,
                 "seconds":now-self.last_end,"metrics":metrics,
                 "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                 "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20}
            records.append(row)
            if self.accelerator.is_main_process:
                with (args.output/"steps.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
                print("AUDIT_STEP "+json.dumps(row),flush=True)
            self.last_end=now
            kw["pbar"].update(1)
    try:
        accelerator=entry._build_accelerator(cfg)
        trainer=AuditedTrainer(cfg,accelerator=accelerator,dataset=ds)
        result["trainable_parameters"]=sum(x.numel() for x in trainer.architecture.parameters() if x.requires_grad)
        trainer.train()
        timed=[r for r in records if r["micro_step"]>args.warmup] if args.mode=="benchmark" else records
        seconds=sum(r["seconds"] for r in timed)
        result.update(status="ok",world_size=accelerator.num_processes,effective_batch=args.batch*args.accum*accelerator.num_processes,
                      steps=len(records),optimizer_updates=records[-1]["optimizer_step"],
                      timed_seconds=seconds,timed_samples=sum(r["samples"] for r in timed),
                      samples_per_second=sum(r["samples"] for r in timed)/seconds,
                      median_micro_step_seconds=statistics.median(r["seconds"] for r in timed),
                      peak_allocated_mib=max(r["peak_allocated_mib"] for r in records),
                      peak_reserved_mib=max(r["peak_reserved_mib"] for r in records))
    except BaseException as exc:
        result.update(status="oom" if isinstance(exc,torch.cuda.OutOfMemoryError) else "error",error=repr(exc))
        raise
    finally:
        result["wall_seconds"]=time.monotonic()-started
        if int(os.environ.get("RANK",0))==0:
            (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
            print("RESULT "+json.dumps(result),flush=True)
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=="__main__": main()
