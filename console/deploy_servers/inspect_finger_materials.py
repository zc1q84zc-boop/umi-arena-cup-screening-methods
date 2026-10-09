#!/usr/bin/env python3
"""Read-only USD material inventory for the two YUBI fingers."""

import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
try:
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(sys.argv[1])
    if stage is None:
        raise RuntimeError("cannot open USD asset")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "yubi_leftfinger" not in path and "yubi_rightfinger" not in path:
            continue
        if prim.IsA(UsdGeom.Mesh):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            print("MESH", path, "material", material.GetPath() if material else "none", flush=True)
        elif prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader):
            print("SHADER", path, prim.GetTypeName(), flush=True)
    material = UsdShade.Material.Get(stage, "/panda/YubiLooks/Jaws")
    if material:
        shader, _, _ = material.ComputeSurfaceSource()
        print("JAW_SHADER", shader.GetPath() if shader else "none", flush=True)
        if shader:
            for parameter in shader.GetInputs():
                print("JAW_INPUT", parameter.GetBaseName(), parameter.Get(), flush=True)
finally:
    app.close()
