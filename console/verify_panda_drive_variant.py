"""Read-only USD composition check for an isolated Panda drive scene."""

from __future__ import annotations

import argparse
import math

from isaacsim import SimulationApp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    args = parser.parse_args()
    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(args.scene)
        if stage is None:
            raise RuntimeError("USD stage could not be opened")
        for side in ("LeftMount", "RightMount"):
            for index in range(1, 8):
                path = f"/World/Robots/{side}/Panda/joints/panda_joint{index}"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise RuntimeError(f"missing Panda joint {path}")
                stiffness = prim.GetAttribute("drive:angular:physics:stiffness").Get()
                damping = prim.GetAttribute("drive:angular:physics:damping").Get()
                if not (math.isclose(stiffness, 2500, abs_tol=1e-3) and
                        math.isclose(damping, 70.710678, abs_tol=1e-3)):
                    raise RuntimeError(f"unexpected composed drive at {path}: {stiffness}, {damping}")
        print("verified both Panda articulations: 14 drives K=2500, D=70.710678")
    finally:
        app.close()


if __name__ == "__main__":
    main()
