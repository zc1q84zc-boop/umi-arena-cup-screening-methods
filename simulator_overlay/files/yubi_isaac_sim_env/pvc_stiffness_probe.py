"""Per-modulus physical platen verification; not a model grasp."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    from .pvc_stiffness import STIFFNESS_SPECS, profile_for
    parser.add_argument('--profile',choices=[key for key,_ in STIFFNESS_SPECS],required=True)
    args=parser.parse_args()
    PROFILE=profile_for(args.profile)
    args.output.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':0,'physics_gpu':0,'multi_gpu':False})
    from pxr import UsdGeom,UsdPhysics,UsdShade,UsdLux,PhysxSchema,Gf
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    from .pvc_shell import author_shell,ShellCupView
    from .recording import VideoWriter
    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    scene=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
    physics=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physics.CreateEnableGPUDynamicsAttr(True);physics.CreateBroadphaseTypeAttr('GPU')
    physics.CreateSolverTypeAttr('TGS');physics.CreateTimeStepsPerSecondAttr(120)
    world=World(physics_dt=1/120,rendering_dt=1/30,device='cuda:0')
    rest=author_shell(stage,PROFILE)
    def cube(path,half,p,kinematic=False):
        body=UsdGeom.Cube.Define(stage,path);body.CreateSizeAttr(2)
        move=body.AddTranslateOp();move.Set(Gf.Vec3d(*p));body.AddScaleOp().Set(Gf.Vec3f(*half))
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        offset=PhysxSchema.PhysxCollisionAPI.Apply(body.GetPrim())
        offset.CreateContactOffsetAttr(.0007);offset.CreateRestOffsetAttr(0.)
        if kinematic:
            api=UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());api.CreateKinematicEnabledAttr(True)
        return body,move
    cube('/World/Floor',(.2,.2,.01),(0,0,-.01))
    # A two-slider fixed-base press uses the same GPU articulation drive path
    # as the robot. PhysX107.3's GPU rigid-body view does NOT implement
    # set_kinematic_targets, despite that method existing in the Python API.
    UsdGeom.Xform.Define(stage,'/World/Press')
    base=UsdGeom.Xform.Define(stage,'/World/Press/Base')
    base.AddTranslateOp().Set(Gf.Vec3d(0,0,.052))
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr(1.)
    fixed=UsdPhysics.FixedJoint.Define(stage,'/World/Press/WorldJoint')
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(0,0,.052))
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
    left,_=cube('/World/Press/LeftPlaten',(.005,.02,.018),(-.06,0,.052))
    right,_=cube('/World/Press/RightPlaten',(.005,.02,.018),(.06,0,.052))
    for name,side,body in (('Left',-1,left),('Right',1,right)):
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.1)
        joint=UsdPhysics.PrismaticJoint.Define(stage,f'/World/Press/{name}Slide')
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateAxisAttr('X')
        joint.CreateLowerLimitAttr(-.065 if side<0 else .025)
        joint.CreateUpperLimitAttr(-.025 if side<0 else .065)
        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
        drive.CreateStiffnessAttr(1.e5);drive.CreateDampingAttr(200.)
        drive.CreateMaxForceAttr(20.);drive.CreateTargetPositionAttr(side*.06)
    fingertip=UsdShade.Material.Define(stage,'/World/FingerMaterial')
    material=UsdPhysics.MaterialAPI.Apply(fingertip.GetPrim())
    material.CreateStaticFrictionAttr(.8);material.CreateDynamicFrictionAttr(.8)
    PhysxSchema.PhysxMaterialAPI.Apply(fingertip.GetPrim()).CreateFrictionCombineModeAttr('max')
    for p in (left,right): UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(fingertip,materialPurpose='physics')
    UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(1200)
    usd_camera=UsdGeom.Camera.Define(stage,'/World/Camera')
    matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(.21,-.24,.16),Gf.Vec3d(0,0,.035),Gf.Vec3d(0,0,1)).GetInverse()
    usd_camera.AddTransformOp().Set(matrix)
    camera=Camera(prim_path='/World/Camera',resolution=(640,480),frequency=30)
    camera.set_clipping_range(.001,10.)
    camera.set_focal_length(.024)
    camera.set_horizontal_aperture(.036)
    video=None;report={'status':'failed','profile':PROFILE,'purpose':'physical_platen_compression_not_model_grasp',
                      'physics_hz':120,'render_hz':30,'prescribed_vertex_animation':False,
                      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'shell_sha256':hashlib.sha256(Path(__file__).with_name('pvc_shell.py').read_bytes()).hexdigest(),
                      'registry_sha256':hashlib.sha256(Path(__file__).with_name('pvc_stiffness.py').read_bytes()).hexdigest()}
    rows=[]
    try:
        world.reset();view=ShellCupView(rest,PROFILE);camera.initialize()
        # Drive ONLY the rigid press joints. The cup nodes remain dynamic.
        import torch
        press=view.sim.create_articulation_view('/World/Press/WorldJoint')
        if press.count != 1 or press.max_dofs != 2:
            raise RuntimeError('Expected one two-slider press articulation')
        platen_views=[view.sim.create_rigid_body_view(str(p.GetPath())) for p in (left,right)]
        if any(p.count != 1 for p in platen_views):
            raise RuntimeError('Expected two independent kinematic platen views')
        indices=torch.tensor([0],dtype=torch.int32,device=view.sim.device)
        targets=press.get_dof_positions().clone()
        dof_names=list(press.shared_metatype.dof_names)
        if set(dof_names) != {'LeftSlide','RightSlide'}:
            raise RuntimeError('Unexpected press joint names')
        for _ in range(8):world.step(render=True)
        video=VideoWriter(args.output/'video.mp4',640,480,30)
        with (args.output/'deformation.jsonl').open('x') as audit:
            for i in range(960):
                if i<120:phase='settle';distance=.06
                elif i<360:phase='compress';distance=.06-(.06-.04)*(i-120)/239
                elif i<480:phase='hold';distance=.04
                elif i<720:phase='release';distance=.04+(.06-.04)*(i-480)/239
                else:phase='recover';distance=.06
                for name,side in (('LeftSlide',-1),('RightSlide',1)):
                    targets[0,dof_names.index(name)]=side*distance
                press.set_dof_position_targets(targets,indices)
                world.step(render=(i%4==0))
                actual=[p.get_transforms()[0,:3].detach().cpu().numpy().copy() for p in platen_views]
                tracking=max(np.linalg.norm(p-[side*distance,0,.052]) for side,p in zip((-1,1),actual))
                if tracking > .005:
                    raise RuntimeError('Physical platen failed to track its target')
                metrics=view.metrics()
                if metrics['max_nodal_shape_change_m']>PROFILE['diagnostic_max_shape_change_m']:
                    raise RuntimeError('FEM cup instability/deformation safety limit')
                row=dict(physics_step=i,phase=phase,platen_gap_m=2*(distance-.005),
                         actual_platen_gap_m=float(actual[1][0]-actual[0][0]-.01),
                         platen_tracking_error_m=float(tracking),**metrics)
                audit.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                if i in (119,479,959):
                    np.savez_compressed(args.output/f'nodes_{i:04d}.npz',rest=rest,current=view.nodes())
                if i%4==0:
                    rgba=camera.get_rgba()
                    if rgba is not None and len(rgba): video.write(rgba[:,:,:3])
            np.savez_compressed(args.output/'final_nodes.npz',rest=rest,current=view.nodes())
        settled=rows[119]['nodal_shape_rms_m']
        held=max(r['nodal_shape_rms_m'] for r in rows if r['phase']=='hold')
        recovered=rows[-1]['nodal_shape_rms_m']
        report.update(status='completed',settled_shape_rms_m=settled,held_shape_rms_m=held,
                      recovered_shape_rms_m=recovered,physical_deformation_observed=held>settled+.0001,
                      unloaded_shape_preserved=settled<.0005,
                      recovered_after_release=recovered<held*.8,
                      platen_motion_verified=bool(all(r['platen_tracking_error_m'] < .005 for r in rows)
                          and min(r['actual_platen_gap_m'] for r in rows) < .08
                          and rows[-1]['actual_platen_gap_m'] > .109),
                      platen_actuation='force_limited_prismatic_joint_drives_20N_each',
                      max_shape_change_m=max(r['max_nodal_shape_change_m'] for r in rows),
                      sample_count=len(rows),grasp_success_claimed=False)
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            if video:
                try:video.close()
                except Exception as exc:
                    report['status']='failed'
                    report['video_error']=f'{type(exc).__name__}: {exc}'
                report['video_frames']=video.frame_count
            (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print(json.dumps(report),flush=True)
        finally:
            app.close()


if __name__=='__main__':main()
