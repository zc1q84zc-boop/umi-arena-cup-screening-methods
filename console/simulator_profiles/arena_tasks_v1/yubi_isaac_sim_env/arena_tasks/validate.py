"""Inspect authored USD collision openings, joints and composed dependencies."""
import json
from pxr import Usd,UsdGeom,UsdPhysics
from .catalog import ROOT,TASK_IDS


def validate(key):
    path=ROOT/'scenes'/f'{key}.usda';stage=Usd.Stage.Open(str(path));meta=json.loads(path.with_suffix('.json').read_text())
    if not stage:raise ValueError('Cannot open stage')
    for name in ('Cup','Tray'):
        prim=stage.GetPrimAtPath('/World/Objects/'+name)
        if prim and prim.IsActive():raise ValueError('Legacy cup/plate must be inactive')
    for side in ('LeftMount','RightMount'):
        if not stage.GetPrimAtPath(f'/World/Robots/{side}/Panda/yubi_base'):raise ValueError('Shared robot missing')
    for obj in meta['objects']:
        prim=stage.GetPrimAtPath(obj['prim_path'])
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Task rigid body missing')
        collisions=[p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
        if not collisions:raise ValueError('Empty task collider')
        if obj['kind'] in ('open_box','lid','divider','sorter','socket') and len(collisions)<5:raise ValueError('Hollow objects cannot use a solid outer hull')
    joints=[p for p in stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]
    if key=='cable' and len(joints)!=meta['cable']['segments']:raise ValueError('Broken cable chain')
    if key=='phone':
        if any(p.IsA(UsdPhysics.RevoluteJoint) and 'Arena' in str(p.GetPath()) for p in stage.Traverse()):raise ValueError('The demo lid is detachable')
    return dict(task_id=key,status='ok',rigid_bodies=len(meta['objects']),spherical_joints=len(joints),primitive_count=len(meta['primitives']))

if __name__=='__main__':print(json.dumps([validate(k) for k in TASK_IDS],indent=2))
