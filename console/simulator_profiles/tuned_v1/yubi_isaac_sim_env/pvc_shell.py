"""Opt-in elastic cup shell using Isaac 5.1 / PhysX 107.3 beta FEM.

No vertex animation, hidden rigid cup, attachments, or oracle forces. Parameters
are an UNMEASURED PVC-like elastic trial, not a material identification. Plastic
yield, permanent creases, fracture and viscoelasticity are not represented.
"""
from __future__ import annotations

import numpy as np

PROFILE = dict(id='pvc_elastic_shell_v1', measured=False,
    model='PhysX107.3_surface_XPBD_FEM_corotational_linear_elasticity',
    youngs_modulus_Pa=1.e9, poissons_ratio=.38,
    surface_bend_stiffness_Pa=1.e9/(12*(1-.38**2)),
    rest_bend_angles_default='restShapeDefault',
    thickness_m=.001, mass_kg=.0317, dynamic_friction=.4,
    bottom_outer_radius_m=.027, top_outer_radius_m=.04, height_m=.075,
    diagnostic_max_shape_change_m=.06,
    solver_position_iterations=32, circumferential_segments=48, height_segments=12,
    limitations='Unmeasured elastic PVC-like trial; no plastic yield, crease, fracture or viscoelasticity. Static friction unsupported by this PhysX deformable backend.')
SHELL_PATH = '/World/Objects/Cup/ElasticShell'


def cup_mesh(profile=PROFILE):
    """Single manifold triangle shell: open rim, joined bottom, no top cap."""
    n, h = profile['circumferential_segments'], profile['height_segments']
    t, height = profile['thickness_m'], profile['height_m']
    rb, rt = profile['bottom_outer_radius_m']-t/2, profile['top_outer_radius_m']-t/2
    if n < 12 or h < 2 or not (0 < t < rb < rt and height > t):
        raise ValueError('invalid thin-shell geometry')
    points, triangles = [(0., 0., t/2)], []
    # Radial bottom rings improve triangles compared with one 27 mm fan.
    for radius in np.linspace(rb/4, rb, 4):
        points.extend((radius*np.cos(2*np.pi*i/n), radius*np.sin(2*np.pi*i/n), t/2) for i in range(n))
    for i in range(n):
        triangles.append((0, 1+(i+1)%n, 1+i))  # bottom outward normal -Z
    for j in range(3):
        a, b = 1+j*n, 1+(j+1)*n
        for i in range(n):
            k=(i+1)%n
            triangles.extend(((a+i,a+k,b+k),(a+i,b+k,b+i)))
    # Ring 4 is shared by bottom and wall, not duplicated/disconnected.
    for j in range(1, h+1):
        f=j/h; radius=rb+(rt-rb)*f; z=t/2+(height-t)*f
        points.extend((radius*np.cos(2*np.pi*i/n), radius*np.sin(2*np.pi*i/n), z) for i in range(n))
        a, b = 1+(3+j-1)*n, 1+(3+j)*n
        for i in range(n):
            k=(i+1)%n
            triangles.extend(((a+i,b+k,a+k),(a+i,b+i,b+k)))
    # Above wall winding is reversed here for outward-facing wall normals.
    triangles[n+6*n:] = [(a,c,b) for a,b,c in triangles[n+6*n:]]
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def author_shell(stage, profile=PROFILE):
    """Session-only replacement; source rigid cup/scene files are untouched."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Vt
    from omni.physx.scripts import deformableUtils
    points, triangles = cup_mesh(profile)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        root=stage.GetPrimAtPath('/World/Objects/Cup')
        if not root:
            root=UsdGeom.Xform.Define(stage, '/World/Objects/Cup').GetPrim()
        root.RemoveAPI(UsdPhysics.RigidBodyAPI)
        for child in ('Collision', 'Visual'):
            prim=stage.GetPrimAtPath('/World/Objects/Cup/'+child)
            if prim: prim.SetActive(False)
        mesh=UsdGeom.Mesh.Define(stage, SHELL_PATH)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr([3]*len(triangles))
        mesh.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        if not deformableUtils.set_physics_surface_deformable_body(stage, mesh.GetPath()):
            raise RuntimeError('Installed PhysX does not support the surface deformable schema')
        prim=mesh.GetPrim()
        # The schema's flatDefault would try to flatten the manufactured cup,
        # rounding its base before a gripper even touched it. Preserve the
        # unloaded 3-D shell's dihedral angles, per the official schema guide.
        prim.GetAttribute('omniphysics:restBendAnglesDefault').Set(profile['rest_bend_angles_default'])
        prim.GetAttribute('omniphysics:mass').Set(profile['mass_kg'])
        if not prim.ApplyAPI('PhysxSurfaceDeformableBodyAPI'):
            raise RuntimeError('PhysxSurfaceDeformableBodyAPI unavailable')
        prim.GetAttribute('physxDeformableBody:solverPositionIterationCount').Set(profile['solver_position_iterations'])
        prim.GetAttribute('physxDeformableBody:selfCollision').Set(True)
        collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateRestOffsetAttr(t := profile['thickness_m']/2)
        collision.CreateContactOffsetAttr(t+.0002)
        material_path='/World/ContactExperiment/PVCElasticShell'
        if not deformableUtils.add_surface_deformable_material(stage, material_path,
                dynamic_friction=profile['dynamic_friction'], youngs_modulus=profile['youngs_modulus_Pa'],
                poissons_ratio=profile['poissons_ratio'], surface_thickness=profile['thickness_m'],
                surface_bend_stiffness=profile['surface_bend_stiffness_Pa']):
            raise RuntimeError('Surface material authoring failed')
        material=UsdShade.Material(stage.GetPrimAtPath(material_path))
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
        # Reuse the original cup's optical material, if present.
        optical=stage.GetPrimAtPath('/World/Objects/Cup/Looks/CupMaterial')
        if optical:
            UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(optical))
        else:
            mesh.CreateDisplayColorAttr([Gf.Vec3f(.12,.55,.7)])
        old_collision=stage.GetPrimAtPath('/World/Objects/Cup/Collision')
        if root.HasAPI(UsdPhysics.RigidBodyAPI) or (old_collision and old_collision.IsActive()):
            raise RuntimeError('Hidden rigid cup must not coexist with deformable shell')
    return points


def rigid_fit(rest, current):
    """Least-squares rigid pose for evaluation only; does not move the mesh."""
    rest, current=np.asarray(rest, dtype=float), np.asarray(current, dtype=float)
    if rest.shape != current.shape or not np.isfinite(current).all():
        raise ValueError('Invalid FEM nodal state')
    a,b=rest.mean(0),current.mean(0)
    u,_,vt=np.linalg.svd((rest-a).T@(current-b))
    rot=vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T
    translation=b-rot@a
    return translation,rot,current-(rest@rot.T+translation)


def deformation_metrics(rest, current, profile=PROFILE):
    translation,rot,residual=rigid_fit(rest,current)
    local=(np.asarray(current)-translation)@rot
    rim=local[-profile['circumferential_segments']:]
    eigen=np.linalg.eigvalsh(np.cov(rim[:,:2].T, bias=True))
    diameters=2*np.sqrt(np.maximum(eigen,0)*2)
    radius=profile['top_outer_radius_m']-profile['thickness_m']/2
    return dict(node_count=len(rest), nodal_shape_rms_m=float(np.sqrt(np.mean(residual**2))),
        max_nodal_shape_change_m=float(np.linalg.norm(residual,axis=1).max()),
        rim_minor_diameter_m=float(diameters[0]), rim_major_diameter_m=float(diameters[1]),
        rim_compression_fraction=float(1-diameters[0]/(2*radius)),
        min_node_world_z_m=float(np.min(np.asarray(current)[:,2])),
        pose_method='least_squares_nodal_rigid_fit_for_evaluation_only', measured_material=False)


class ShellCupView:
    """Small environment-compatible facade over real FEM nodal tensors."""
    def __init__(self, rest, profile=PROFILE):
        self.rest, self.profile=np.asarray(rest),profile
        self.initialize()

    def initialize(self):
        import omni.physics.tensors as tensors
        import torch
        self.torch=torch
        # PhysX GPU pipelines reject the NumPy tensor frontend. Only evaluation
        # snapshots are copied to the host; resets are sent on the PhysX device.
        self.sim=tensors.create_simulation_view('torch')
        self.sim.set_subspace_roots('/')
        self.view=self.sim.create_surface_deformable_body_view(SHELL_PATH)
        if self.view.count != 1 or self.view.max_simulation_nodes_per_body != len(self.rest):
            raise RuntimeError('Expected exactly one FEM cup with unchanged topology')

    def nodes(self):
        return self.view.get_simulation_nodal_positions()[0].detach().cpu().numpy().copy()

    def get_world_poses(self):
        from scipy.spatial.transform import Rotation
        p,r,_=rigid_fit(self.rest,self.nodes())
        q=Rotation.from_matrix(r).as_quat()[[3,0,1,2]]
        return p[None,:],q[None,:]

    def set_world_poses(self, positions, orientations):
        from scipy.spatial.transform import Rotation
        def array(value):
            return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
        p,q=array(positions)[0],array(orientations)[0]
        r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        self.view.set_simulation_nodal_positions(
            self.torch.as_tensor((self.rest@r.T+p)[None,:,:],dtype=self.torch.float32,device=self.sim.device),
            self.torch.tensor([0],dtype=self.torch.int32,device=self.sim.device))

    def set_velocities(self, value):
        if np.any(value.detach().cpu().numpy() if hasattr(value,'detach') else value):
            raise ValueError('Only zero-velocity episode reset supported')
        self.view.set_simulation_nodal_velocities(
            self.torch.zeros((1,len(self.rest),3),dtype=self.torch.float32,device=self.sim.device),
            self.torch.tensor([0],dtype=self.torch.int32,device=self.sim.device))

    def get_linear_velocities(self):
        return self.view.get_simulation_nodal_velocities().mean(dim=1).detach().cpu().numpy()

    def get_angular_velocities(self):
        p=self.nodes();p-=p.mean(0)
        v=self.view.get_simulation_nodal_velocities()[0].detach().cpu().numpy().copy();v-=v.mean(0)
        inertia=np.eye(3)*(p*p).sum()-p.T@p
        return np.linalg.lstsq(inertia,np.cross(p,v).sum(0),rcond=None)[0][None,:]

    def get_masses(self):
        return np.array([self.profile['mass_kg']])

    def metrics(self):
        return deformation_metrics(self.rest,self.nodes(),self.profile)
