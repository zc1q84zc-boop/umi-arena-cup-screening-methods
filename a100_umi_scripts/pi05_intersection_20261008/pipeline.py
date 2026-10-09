#!/usr/bin/env python3
"""Guarded A100 launch: preparation, real batch checks, gradient/save/reload, train."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.work_dir
    lock = (root/"pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    status = {}

    def update(**values):
        status.update(values, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (root/"status.json.tmp").write_text(json.dumps(status,indent=2)+"\n")
        (root/"status.json.tmp").replace(root/"status.json")
        print(json.dumps(status),flush=True)

    def free_cards():
        rows = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
        busy = {s.split(",")[0].strip() for s in subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True).splitlines() if s.strip()}
        cards = []
        for row in rows.splitlines():
            idx,uuid,memory,util = [s.strip() for s in row.split(",")]
            # GPU 0 is reserved for other users, even if it appears idle.
            if int(idx) in range(1,8) and uuid not in busy and int(memory)<500 and int(util)<=2:
                cards.append({"index":int(idx),"uuid":uuid})
        return cards

    try:
        update(phase="waiting_preparation")
        prep_pid = int((root/"prepare.pid").read_text())
        while not (root/"prepared/manifest.json").is_file():
            proc = Path(f"/proc/{prep_pid}/stat")
            if not proc.exists() or proc.read_text().split()[2] == "Z":
                raise RuntimeError("Preparation stopped before manifest completion; inspect prepare.log")
            time.sleep(5)
        asset = root/"assets/pi05_cup_intersection/intersection_v2"
        asset.mkdir(parents=True,exist_ok=True)
        (asset/"norm_stats.json").write_bytes((root/"prepared/norm_stats.json").read_bytes())
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join([str(root/"scripts"),"/mnt/data/benyun/workspace/projects/openpi/src",str(root/"evaluation")]),
                   HF_HOME="/mnt/data/benyun/workspace/huggingface-cache",PYTHONUNBUFFERED="1",
                   XLA_PYTHON_CLIENT_MEM_FRACTION="0.93",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                   TOKENIZERS_PARALLELISM="false",WANDB_MODE="disabled")
        common = ["/home/benyun/.venvs/umi_arena_pi05/bin/python",str(root/"scripts/train.py"),
                  "--work-dir",str(root),"--openpi-root","/mnt/data/benyun/workspace/projects/openpi",
                  "--evaluation-root",str(root/"evaluation")]
        for name in ("data-smoke","gradient-smoke","verify-checkpoint","train"):
            cards = free_cards()
            if len(cards) < 2: raise RuntimeError("Fewer than two idle allowed A100 GPUs; stopped before allocating")
            cards = cards[:2]
            env["CUDA_VISIBLE_DEVICES"] = ",".join(c["uuid"] for c in cards)
            if name == "train":
                # Recheck immediately before the expensive formal run.
                if not {c["uuid"] for c in cards}.issubset({c["uuid"] for c in free_cards()}):
                    raise RuntimeError("Selected GPUs became busy before training")
                (root/"allocation.json").write_text(json.dumps(cards,indent=2)+"\n")
            logpath = root/(name+".log")
            assert not logpath.exists(), f"Refusing to overwrite {logpath}"
            with logpath.open("wb") as log:
                proc = subprocess.Popen(common+["--mode",name],stdout=log,stderr=subprocess.STDOUT,env=env)
            (root/(name+".pid")).write_text(str(proc.pid)+"\n")
            update(phase=name,pid=proc.pid,gpus=cards,log=str(logpath))
            code = proc.wait()
            update(last_completed=name,last_exit_code=code)
            if code: raise RuntimeError(f"{name} failed with exit {code}; inspect {logpath}")
            # CUDA teardown may outlive the child by a moment.
            for _ in range(12):
                if len(free_cards())>=2: break
                time.sleep(5)
        update(phase="complete",completed_steps=30000)
    except Exception as error:
        update(phase="failed",error=str(error))
        raise


if __name__ == "__main__": main()
