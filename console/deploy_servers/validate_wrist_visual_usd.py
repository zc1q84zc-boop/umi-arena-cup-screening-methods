"""Check composed collision geometry, physics values and material bindings."""
import json
from pxr import Usd, UsdPhysics, UsdGeom, UsdShade
from yubi_isaac_sim_env.visual_alignment import apply_visual_alignment

stage=Usd.Stage.Open('yubi_isaac_sim_env/scenes/dual_franka_yubi_cup40k_cupfriction_trial.usda')

def snapshot():
    attributes={};relations={};collision={};bindings={}
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(('physics:','physx')):attributes[path+':'+attr.GetName()]=str(attr.Get())
        for rel in prim.GetRelationships():
            if 'physics' in rel.GetName().lower():relations[path+':'+rel.GetName()]=list(map(str,rel.GetTargets()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision[path]={a.GetName():str(a.Get()) for a in prim.GetAttributes()}
            mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            bindings[path]=str(mat.GetPath()) if mat else None
    return dict(attributes=attributes,relations=relations,collision=collision,bindings=bindings)

before=snapshot();visual=apply_visual_alignment(stage);after=snapshot()
render_visibility_changes={}
for path,attrs in before['collision'].items():
    if attrs.get('visibility')!=after['collision'][path].get('visibility'):
        render_visibility_changes[path]=[attrs.get('visibility'),after['collision'][path].get('visibility')]
    # Render visibility does not enable or disable the PhysX collider.
    attrs.pop('visibility',None);after['collision'][path].pop('visibility',None)
assert before==after,'Visual alignment changed a physical or collision geometry value'
extra=[str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith('/World/VisualAlignment') and
       (p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI))]
assert not extra
for p in stage.Traverse():
    if p.GetTypeName()=='GeomSubset' and p.GetName()=='RedDistalTrim':
        assert UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0].GetPath().name=='GloveRed'
result=dict(physics_attributes_checked=len(before['attributes']),physics_relationships_checked=len(before['relations']),
            collision_prims_checked=len(before['collision']),physics_material_bindings_unchanged=True,
            collision_render_visibility_changes=render_visibility_changes,
            physics_unchanged=True,new_collision_prims=extra,visual=visual)
stage.GetSessionLayer().Export('visual_session_validated.usda')
open('usd_visual_validation.json','w').write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
