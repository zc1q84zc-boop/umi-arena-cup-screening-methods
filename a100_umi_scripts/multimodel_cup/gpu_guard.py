#!/usr/bin/env python3
"""Refuse launch if any selected physical A100 is already in use."""

import argparse
import subprocess


def query(fields: str) -> list[list[str]]:
    output = subprocess.check_output([
        "nvidia-smi", f"--query-{fields}", "--format=csv,noheader,nounits",
    ], text=True)
    return [[piece.strip() for piece in line.split(",")] for line in output.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if len(set(args.gpu)) != len(args.gpu):
        raise ValueError("duplicate GPU index")
    gpus = {int(row[0]): (row[1], int(row[2])) for row in query("gpu=index,uuid,memory.used")}
    selected = {gpus[index][0] for index in args.gpu}
    active = [row for row in query("compute-apps=gpu_uuid,pid,process_name") if row[0] in selected]
    reserved = {index: used for index, (_, used) in gpus.items() if index in args.gpu and used > 2048}
    if active or reserved:
        raise RuntimeError(f"selected GPU already occupied: processes={active}, memory_MiB={reserved}")
    print(f"GPU_GUARD_OK physical indices={args.gpu}", flush=True)


if __name__ == "__main__":
    main()
