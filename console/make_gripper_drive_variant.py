"""Create an isolated Isaac USD scene overriding YUBI jaw drive tuning.

The source scene and composed robot asset remain untouched. This is a
simulation-only diagnostic; the numbers are not measured motor limits.
Run with the repository's ``pxr_python.sh`` in its Isaac Sim environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Sdf, Usd


JOINT_PATHS = (
    "/World/Robots/LeftMount/Panda/joints/yubi_finger_joint",
    "/World/Robots/RightMount/Panda/joints/yubi_finger_joint",
)
MATERIAL_PATHS = (
    "/World/Robots/LeftMount/Panda/YubiLooks/GripPhysics",
    "/World/Robots/RightMount/Panda/YubiLooks/GripPhysics",
)


def create(source: Path, output: Path, max_force_nm: float, stiffness: float,
           static_friction: float | None = None,
           dynamic_friction: float | None = None) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scene variant: {output}")
    if not source.is_file() or output.resolve() == source.resolve():
        raise ValueError("source must be an existing, separate scene")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths.append(str(source.resolve()))
    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        raise ValueError("source has no /World prim")
    stage.SetDefaultPrim(world)
    for path in JOINT_PATHS:
        if not stage.GetPrimAtPath(path).IsValid():
            raise ValueError(f"missing source jaw joint: {path}")
        prim = stage.OverridePrim(path)
        prim.CreateAttribute("drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float).Set(max_force_nm)
        prim.CreateAttribute("drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float).Set(stiffness)
    if static_friction is not None and dynamic_friction is not None:
        for path in MATERIAL_PATHS:
            if not stage.GetPrimAtPath(path).IsValid():
                raise ValueError(f"missing source grip material: {path}")
            prim = stage.OverridePrim(path)
            prim.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float).Set(static_friction)
            prim.CreateAttribute("physics:dynamicFriction", Sdf.ValueTypeNames.Float).Set(dynamic_friction)
    stage.GetRootLayer().Save()
    verified = Usd.Stage.Open(str(output))
    for path in JOINT_PATHS:
        prim = verified.GetPrimAtPath(path)
        if (abs(prim.GetAttribute("drive:angular:physics:maxForce").Get() - max_force_nm) > 1e-5
                or abs(prim.GetAttribute("drive:angular:physics:stiffness").Get() - stiffness) > 1e-5):
            raise RuntimeError(f"variant override not composed at {path}")
    if static_friction is not None and dynamic_friction is not None:
        for path in MATERIAL_PATHS:
            prim = verified.GetPrimAtPath(path)
            if (abs(prim.GetAttribute("physics:staticFriction").Get() - static_friction) > 1e-5
                    or abs(prim.GetAttribute("physics:dynamicFriction").Get() - dynamic_friction) > 1e-5):
                raise RuntimeError(f"material override not composed at {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-force-nm", type=float, default=2.0)
    parser.add_argument("--stiffness", type=float, default=0.05)
    parser.add_argument("--static-friction", type=float)
    parser.add_argument("--dynamic-friction", type=float)
    args = parser.parse_args()
    if not 0 < args.max_force_nm <= 5 or not 0 < args.stiffness <= 0.1:
        parser.error("diagnostic drive tuning must be within force (0,5] Nm and stiffness (0,0.1]")
    if (args.static_friction is None) != (args.dynamic_friction is None):
        parser.error("provide both friction coefficients or neither")
    if args.static_friction is not None and not (0 < args.dynamic_friction <= args.static_friction <= 3):
        parser.error("diagnostic friction must satisfy 0 < dynamic <= static <= 3")
    create(args.source, args.output, args.max_force_nm, args.stiffness,
           args.static_friction, args.dynamic_friction)
    print(args.output)
