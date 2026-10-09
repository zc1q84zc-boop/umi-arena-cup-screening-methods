#!/usr/bin/env python3
"""Check the isolated online Panda drive overlay inside an Isaac Sim runtime."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

from isaacsim import SimulationApp


app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics  # noqa: E402  (Kit must start first)

stage = Usd.Stage.Open(str(Path(sys.argv[1]).resolve()))
if stage is None:
    raise RuntimeError("could not open online smooth scene")
for side in ("LeftMount", "RightMount"):
    for index in range(1, 8):
        path = f"/World/Robots/{side}/Panda/joints/panda_joint{index}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"missing {path}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        stiffness = drive.GetStiffnessAttr().Get()
        damping = drive.GetDampingAttr().Get()
        if not (math.isclose(stiffness, 2500.0, abs_tol=1e-3)
                and math.isclose(damping, 70.710678, abs_tol=1e-3)):
            raise RuntimeError(f"{path} drive K={stiffness} D={damping}")
print("validated 14 online Panda drives: K=2500, D=70.710678", flush=True)
os._exit(0)  # Isaac GPU teardown is unreliable on this host after output flush.
