"""CPU USD read of the composed finger convex colliders; no SimulationApp."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd,UsdGeom,UsdPhysics

root=Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1/yubi_isaac_sim_env')
p=argparse.ArgumentParser();p.add_argument('--mount',choices=('LeftMount','RightMount'),default='LeftMount');p.add_argument('--output',type=Path)
args=p.parse_args()
stage=Usd.Stage.Open(str(root/'scenes/dual_franka_yubi_official_fingertip_friction_trial.usda'))
result={}
for side in ('left','right'):
    prim=stage.GetPrimAtPath(f'/World/Robots/{args.mount}/Panda/yubi_{side}finger')
    items=[]
    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Mesh) and child.HasAPI(UsdPhysics.CollisionAPI) and child.GetAttribute('physics:collisionEnabled').Get() is not False:
            vertices=np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get())
            matrix=UsdGeom.Xformable(child).GetLocalTransformation()
            np.testing.assert_allclose(np.array(matrix),np.eye(4),atol=1e-12)
            hull=ConvexHull(vertices)
            indices={int(old):new for new,old in enumerate(hull.vertices)}
            items.append(dict(name=child.GetName(),point_count=len(vertices),equations=hull.equations.tolist(),
                vertices=vertices[hull.vertices].tolist(),
                triangles=[[indices[int(j)] for j in tri] for tri in hull.simplices],
                approximation=child.GetAttribute('physics:approximation').Get(),
                contact_offset_m=child.GetAttribute('physxCollision:contactOffset').Get(),
                rest_offset_m=child.GetAttribute('physxCollision:restOffset').Get()))
            print(side,child.GetName(),len(vertices),vertices.min(0).tolist(),vertices.max(0).tolist())
    result[side+'_finger']=items
output=args.output or Path(__file__).with_name('collider_hulls.json')
output.write_text(json.dumps(result))
print('written',output.stat().st_size)
