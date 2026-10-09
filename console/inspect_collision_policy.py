"""Read-only Isaac diagnostic for the composed YUBI collision approximation."""

import json


def act(observation, step, episode):
    if step == 0:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        values = {}
        for mount in ("LeftMount", "RightMount"):
            for finger in ("yubi_leftfinger", "yubi_rightfinger"):
                path = f"/World/Robots/{mount}/Panda/{finger}/geometry"
                prim = stage.GetPrimAtPath(path)
                attr = prim.GetAttribute("physics:approximation") if prim.IsValid() else None
                values[path] = {"valid": prim.IsValid(), "approximation": attr.Get() if attr else None}
        print("YUBI_COLLISION_APPROXIMATION=" + json.dumps(values, sort_keys=True), flush=True)
    return {}
