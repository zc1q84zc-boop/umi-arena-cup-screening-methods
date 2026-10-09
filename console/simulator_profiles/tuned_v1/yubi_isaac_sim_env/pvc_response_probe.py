"""Isolated constant-actuation-force shell response; no policy or grasp claim.

Unlike the original prescribed-gap check, this diagnostic compares deformation
under the same bounded slider effort. Slider effort is NOT a contact force
sensor. Only quasi-static samples with low platen speed are compared.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def diagnostic_profile(youngs_gpa, iterations, thickness_mm, base=None):
    if base is None:
        from .pvc_shell import PROFILE
        base = PROFILE
    if youngs_gpa not in (1., 3., 6.) or iterations not in (32, 128, 255):
        raise ValueError('Unsupported isolated diagnostic')
    if thickness_mm not in (1., 1.5):
        raise ValueError('Unsupported diagnostic thickness')
    p = dict(base)
    p.update(id=f'response_e{youngs_gpa:g}_it{iterations}_t{thickness_mm:g}',
             youngs_modulus_Pa=youngs_gpa*1e9,
             surface_bend_stiffness_Pa=youngs_gpa*1e9/(12*(1-p['poissons_ratio']**2)),
             solver_position_iterations=iterations, thickness_m=thickness_mm/1000,
             mass_kg=base['mass_kg']*thickness_mm)
    return p


def material_readback(stage, profile):
    from pxr import UsdGeom, UsdShade
    from .pvc_shell import SHELL_PATH
    prim = stage.GetPrimAtPath(SHELL_PATH)
    material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    expected = {'omniphysics:youngsModulus': profile['youngs_modulus_Pa'],
                'omniphysics:poissonsRatio': profile['poissons_ratio'],
                'omniphysics:surfaceThickness': profile['thickness_m'],
                'omniphysics:surfaceBendStiffness': profile['surface_bend_stiffness_Pa'],
                'omniphysics:dynamicFriction': profile['dynamic_friction']}
    actual = {key: material.GetPrim().GetAttribute(key).Get() for key in expected}
    if str(material.GetPath()) != '/World/ContactExperiment/PVCElasticShell':
        raise RuntimeError('Wrong composed physics material binding')
    if not all(np.isclose(actual[k], value, rtol=2e-6) for k, value in expected.items()):
        raise RuntimeError('Authored physics material does not match requested values')
    count = prim.GetAttribute('physxDeformableBody:solverPositionIterationCount').Get()
    if count != profile['solver_position_iterations'] or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise RuntimeError('Wrong units or solver iteration count')
    return dict(material_path=str(material.GetPath()), binding=str(relationship.GetPath()),
                values=actual, solver_position_iterations=count,
                meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
                verified_composed_usd=True, internal_solver_parameter_readback_available=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--youngs-gpa', type=float, choices=(1., 3., 6.), required=True)
    parser.add_argument('--iterations', type=int, choices=(32, 128, 255), required=True)
    parser.add_argument('--hz', type=int, choices=(120, 240), required=True)
    parser.add_argument('--thickness-mm', type=float, choices=(1., 1.5), default=1.)
    args = parser.parse_args()
    profile = diagnostic_profile(args.youngs_gpa, args.iterations, args.thickness_mm)
    args.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'active_gpu': 0, 'physics_gpu': 0, 'multi_gpu': False})
    from pxr import UsdGeom, UsdPhysics, UsdShade, UsdLux, PhysxSchema, Gf
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    from .pvc_shell import author_shell, ShellCupView
    from .recording import VideoWriter
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); scene.CreateGravityMagnitudeAttr(9.81)
    physics = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physics.CreateEnableGPUDynamicsAttr(True); physics.CreateBroadphaseTypeAttr('GPU')
    physics.CreateSolverTypeAttr('TGS'); physics.CreateTimeStepsPerSecondAttr(args.hz)
    world = World(physics_dt=1/args.hz, rendering_dt=1/30, device='cuda:0')
    rest = author_shell(stage, profile)

    def cube(path, half, position):
        body = UsdGeom.Cube.Define(stage, path); body.CreateSizeAttr(2)
        body.AddTranslateOp().Set(Gf.Vec3d(*position)); body.AddScaleOp().Set(Gf.Vec3f(*half))
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        offset = PhysxSchema.PhysxCollisionAPI.Apply(body.GetPrim())
        offset.CreateContactOffsetAttr(.0007); offset.CreateRestOffsetAttr(0.)
        return body

    cube('/World/Floor', (.2, .2, .01), (0, 0, -.01))
    base = UsdGeom.Xform.Define(stage, '/World/Press/Base')
    base.AddTranslateOp().Set(Gf.Vec3d(0, 0, .052))
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr(1.)
    fixed = UsdPhysics.FixedJoint.Define(stage, '/World/Press/WorldJoint')
    fixed.CreateBody1Rel().SetTargets([base.GetPath()]); fixed.CreateLocalPos0Attr(Gf.Vec3f(0, 0, .052))
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
    platens = []
    for name, side in (('Left', -1), ('Right', 1)):
        body = cube('/World/Press/'+name, (.005, .02, .018), (side*.06, 0, .052))
        platens.append(body)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.1)
        joint = UsdPhysics.PrismaticJoint.Define(stage, '/World/Press/'+name+'Slide')
        joint.CreateBody0Rel().SetTargets([base.GetPath()]); joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateAxisAttr('X')
        joint.CreateLowerLimitAttr(-.065 if side < 0 else .025)
        joint.CreateUpperLimitAttr(-.025 if side < 0 else .065)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        # Direct-GPU mode does not implement runtime drive stiffness changes.
        # Author the force-controlled, velocity-damped drive before world.reset.
        drive.CreateStiffnessAttr(0.); drive.CreateDampingAttr(2.)
        drive.CreateMaxForceAttr(2.); drive.CreateTargetPositionAttr(side*.06)
    mat = UsdShade.Material.Define(stage, '/World/FingerMaterial')
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(.8); api.CreateDynamicFrictionAttr(.8)
    PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr('max')
    for body in platens:
        UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(mat, materialPurpose='physics')
    UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1200)
    usd_camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    usd_camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(.21, -.24, .16), Gf.Vec3d(0, 0, .035), Gf.Vec3d(0, 0, 1)).GetInverse())
    camera = Camera(prim_path='/World/Camera', resolution=(640, 480), frequency=30)
    camera.set_clipping_range(.001, 10.); camera.set_focal_length(.024); camera.set_horizontal_aperture(.036)
    report = dict(status='failed', purpose='same_actuation_force_stiffness_response_not_grasp',
                  measured_material=False, profile=profile, physics_hz=args.hz,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  shell_sha256=hashlib.sha256(Path(__file__).with_name('pvc_shell.py').read_bytes()).hexdigest(),
                  prescribed_cup_nodes=False, contact_force_sensor_available=False)
    video = None
    rows = []
    try:
        report['material_before_reset'] = material_readback(stage, profile)
        world.reset(); view = ShellCupView(rest, profile); camera.initialize()
        report['material_after_reset'] = material_readback(stage, profile)
        report['position_drive_stiffness_N_m'] = {
            name: stage.GetPrimAtPath('/World/Press/'+name+'Slide').GetAttribute('drive:linear:physics:stiffness').Get()
            for name in ('Left', 'Right')}
        if any(value != 0 for value in report['position_drive_stiffness_N_m'].values()):
            raise RuntimeError('Position drive must be disabled before simulation starts')
        import torch
        press = view.sim.create_articulation_view('/World/Press/WorldJoint')
        names = list(press.shared_metatype.dof_names)
        if press.count != 1 or set(names) != {'LeftSlide', 'RightSlide'}:
            raise RuntimeError('Wrong diagnostic press articulation')
        platen_views = [view.sim.create_rigid_body_view(str(body.GetPath())) for body in platens]
        indices = torch.tensor([0], dtype=torch.int32, device=view.sim.device)
        forces = press.get_dof_positions().clone()*0
        initial = forces.clone()
        for name, sign in (('LeftSlide', -1), ('RightSlide', 1)):
            initial[0, names.index(name)] = sign*.06
        # Explicit episode initialization ONLY: USD slider limits do not set
        # initial generalized coordinates. Zero otherwise clamps to +/-25mm.
        press.set_dof_positions(initial, indices)
        press.set_dof_velocities(forces.clone(), indices)
        view.set_world_poses(np.array([[0., 0., 0.]]), np.array([[1., 0., 0., 0.]]))
        view.set_velocities(np.zeros((1, 6)))
        for _ in range(8):
            world.step(render=True)
        video = VideoWriter(args.output/'video.mp4', 640, 480, 30)
        render_every = args.hz//30
        with (args.output/'deformation.jsonl').open('x') as audit:
            for i in range(8*args.hz):
                time_s = i/args.hz
                if time_s < 1: phase, effort = 'settle', 0.
                elif time_s < 4: phase, effort = 'load_025N', .25
                elif time_s < 6: phase, effort = 'load_1N', 1.
                else: phase, effort = 'release', -.25
                for name, sign in (('LeftSlide', 1), ('RightSlide', -1)):
                    forces[0, names.index(name)] = sign*effort
                press.set_dof_actuation_forces(forces, indices)
                world.step(render=(i % render_every == 0))
                actual = np.array([p.get_transforms()[0, :3].detach().cpu().numpy().copy() for p in platen_views])
                velocity = press.get_dof_velocities()[0].detach().cpu().numpy().copy()
                readback = press.get_dof_actuation_forces()[0].detach().cpu().numpy().copy()
                if not np.allclose(readback, forces[0].detach().cpu().numpy(), atol=1e-6):
                    raise RuntimeError('Actuation effort readback mismatch')
                metrics = view.metrics()
                if metrics['max_nodal_shape_change_m'] > .025:
                    raise RuntimeError('Diagnostic deformation exceeds 25mm limit')
                gap = float(actual[1, 0]-actual[0, 0]-.01)
                if i == 0 and abs(gap-.11) > .001:
                    raise RuntimeError('Initial slider position was not applied')
                if gap < .045 or metrics['min_node_world_z_m'] > .003:
                    raise RuntimeError('Invalid stiffness comparison: joint stop or cup lifted out of fixture')
                cup_position = view.get_world_poses()[0][0].tolist()
                row = dict(physics_step=i, physics_time_s=time_s, phase=phase,
                           commanded_inward_effort_per_platen_N=effort,
                           actuator_effort_readback_N=readback.tolist(),
                           actual_platen_gap_m=gap, cup_position_m=cup_position,
                           max_platen_speed_m_s=float(np.max(np.abs(velocity))), **metrics)
                audit.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
                if i % render_every == 0:
                    rgba = camera.get_rgba()
                    if rgba is not None and len(rgba): video.write(rgba[:, :, :3])
        summary = {}
        for phase, start in (('load_025N', 3.5), ('load_1N', 5.5)):
            selection = [r for r in rows if r['phase'] == phase and r['physics_time_s'] >= start]
            summary[phase] = dict(samples=len(selection),
                quasi_static=max(r['max_platen_speed_m_s'] for r in selection) < .001,
                median_gap_m=float(np.median([r['actual_platen_gap_m'] for r in selection])),
                median_shape_max_m=float(np.median([r['max_nodal_shape_change_m'] for r in selection])),
                median_rim_compression_fraction=float(np.median([r['rim_compression_fraction'] for r in selection])))
        report.update(status='completed', sample_count=len(rows), responses=summary,
                      grasp_success_claimed=False, max_shape_change_m=max(r['max_nodal_shape_change_m'] for r in rows))
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            if video:
                video.close(); report['video_frames'] = video.frame_count
            (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps(report), flush=True)
        finally:
            app.close()


if __name__ == '__main__':
    main()
