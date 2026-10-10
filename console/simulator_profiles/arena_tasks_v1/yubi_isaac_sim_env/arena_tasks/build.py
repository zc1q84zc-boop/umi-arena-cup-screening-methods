"""Author task-only USD layers; all robot/table/camera assets are composed."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Vt
from .catalog import ROOT, TASK_IDS, load_task


def quat_yaw(angle):
    return Gf.Quatd(math.cos(angle/2), Gf.Vec3d(0, 0, math.sin(angle/2)))


class Builder:
    def __init__(self, task, output):
        self.task = task
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            raise FileExistsError(self.output)
        self.stage = Usd.Stage.CreateNew(str(self.output))
        base = ROOT.parent/'scenes/dual_franka_yubi_cup40k_cupfriction_trial.usda'
        self.stage.GetRootLayer().subLayerPaths = [os.path.relpath(base, self.output.parent)]
        # Keep the detailed robots and physics. The old cup/plate are absent.
        for name in ('Cup', 'Tray'):
            self.stage.OverridePrim('/World/Objects/'+name).SetActive(False)
        self.meta = {}
        self.envelopes = {}
        self.cad_mass_properties = {}
        self.contact = self.material('Contact', [.3, .3, .3], physics=True)
        PhysxSchema.PhysxSceneAPI.Apply(self.stage.GetPrimAtPath('/World/PhysicsScene')).CreateEnableCCDAttr(True)

    def material(self, name, color, physics=False):
        mat = UsdShade.Material.Define(self.stage, '/World/ArenaLooks/'+name)
        if physics:
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(.5); api.CreateDynamicFrictionAttr(.4)
            api.CreateRestitutionAttr(0.)
        else:
            sh = UsdShade.Shader.Define(self.stage, str(mat.GetPath())+'/Surface')
            sh.CreateIdAttr('UsdPreviewSurface')
            sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.65)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return mat

    def shape(self, geom, xyz, material, collision=True):
        geom.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        bind = UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()); bind.Bind(material)
        if collision:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            bind.Bind(self.contact, UsdShade.Tokens.weakerThanDescendants, 'physics')
            api = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
            # Sub-millimetre contact expansion is needed for insertion geometry.
            api.CreateContactOffsetAttr(.00025); api.CreateRestOffsetAttr(0.)
        return geom

    def cube(self, path, size, xyz, mat, collision=True):
        geom = UsdGeom.Cube.Define(self.stage, path); geom.CreateSizeAttr(1.)
        self.shape(geom, xyz, mat, collision)
        geom.AddScaleOp().Set(Gf.Vec3f(*size))
        return geom

    def cylinder(self, path, radius, height, xyz, mat, axis='Z', collision=True):
        g = UsdGeom.Cylinder.Define(self.stage, path)
        g.CreateRadiusAttr(radius); g.CreateHeightAttr(height); g.CreateAxisAttr(axis)
        return self.shape(g, xyz, mat, collision)

    def root(self, obj):
        path = '/World/ArenaObjects/'+obj['id']
        xf = UsdGeom.Xform.Define(self.stage, path)
        xf.AddTranslateOp().Set(Gf.Vec3d(*obj['position_m']))
        q = obj['quaternion_wxyz']; xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
        api = UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        api.CreateKinematicEnabledAttr(not obj['dynamic'])
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(xf.GetPrim())
        body.CreateEnableCCDAttr(True)
        numerics=self.task['body_numerics']
        body.CreateSolverPositionIterationCountAttr(numerics['position_iterations'])
        body.CreateSolverVelocityIterationCountAttr(numerics['velocity_iterations'])
        body.CreateSleepThresholdAttr(numerics['sleep_threshold_m2_s2'])
        body.CreateStabilizationThresholdAttr(numerics['stabilization_threshold_m2_s2'])
        mat = self.material(obj['id'], obj['color'])
        return path, mat

    def mass(self, path, obj, bounds):
        low, high = np.array(bounds); size = high-low; center = (high+low)/2
        api = UsdPhysics.MassAPI.Apply(self.stage.GetPrimAtPath(path))
        x,y,z=size; m=obj['mass_kg']/12
        diagonal=np.array([m*(y*y+z*z),m*(x*x+z*z),m*(x*x+y*y)])
        axes=np.array([0.,0.,0.,1.]); method='bounding box approximation; estimated mass'
        if path in self.cad_mass_properties:
            from scipy.spatial.transform import Rotation
            properties=self.cad_mass_properties[path]
            center=properties['center']
            diagonal,basis=np.linalg.eigh(properties['inertia_per_kg']*obj['mass_kg'])
            if np.linalg.det(basis)<0: basis[:,2]*=-1
            axes=Rotation.from_matrix(basis).as_quat()
            method='CAD solid uniform density; estimated total mass'
        api.CreateMassAttr(obj['mass_kg']); api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
        api.CreatePrincipalAxesAttr(Gf.Quatf(float(axes[3]),Gf.Vec3f(*axes[:3])))
        self.meta[obj['id']] = {**obj, 'prim_path':path, 'local_bounds_m': [low.tolist(),high.tolist()]}
        self.meta[obj['id']]['mass_properties']=dict(method=method,center_of_mass_m=center.tolist(),diagonal_inertia_kg_m2=diagonal.tolist(),principal_axes_xyzw=axes.tolist())
        if path in self.envelopes:
            self.meta[obj['id']]['local_envelope_points_m'] = self.envelopes[path]

    def box(self, path, size, wall, mat, lid=False):
        x,y,z=size
        bz=z-wall/2 if lid else wall/2
        self.cube(path+'/Base',[x,y,wall],[0,0,bz],mat)
        for s in (-1,1):
            self.cube(path+f'/WallX{str(s).replace("-","m")}',[wall,y,z],[s*(x-wall)/2,0,z/2],mat)
            self.cube(path+f'/WallY{str(s).replace("-","m")}',[x,wall,z],[0,s*(y-wall)/2,z/2],mat)

    def digit(self, path, digit, position, mat):
        # Seven-segment geometry makes cell numbers visible without external fonts.
        layouts={'1':'bc','2':'abged','3':'abgcd','4':'fgbc','5':'afgcd','6':'afgecd','7':'abc','8':'abcdefg','9':'abfgcd'}
        segments={'a':(0,.006,.010,.002),'g':(0,0,.010,.002),'d':(0,-.006,.010,.002),
                  'b':(.005,.003,.002,.006),'c':(.005,-.003,.002,.006),
                  'f':(-.005,.003,.002,.006),'e':(-.005,-.003,.002,.006)}
        for s in layouts[str(digit)]:
            dx,dy,sx,sy=segments[s]
            self.cube(path+'/'+s,[sy,sx,.00012],[position[0]+dy,position[1]-dx,position[2]],mat,False)

    def cad(self, path, key, mat):
        import trimesh
        source = ROOT/'assets'/f'{key}.stl'
        manifest=json.loads((ROOT/'assets/source_manifest.json').read_text())
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest[key]['sha256']:
            raise ValueError(f'CAD hash mismatch: {key}')
        tri=trimesh.load_mesh(source, process=False)
        points=np.asarray(tri.vertices,dtype=float)*.001
        order=np.argsort(np.ptp(points,axis=0))[::-1]
        basis=np.eye(3)[:,order]
        if np.linalg.det(basis)<0: basis[:,2]*=-1
        points=points@basis
        lo=points.min(axis=0);hi=points.max(axis=0)
        points-=np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]])
        lo=points.min(axis=0); hi=points.max(axis=0)
        solid=trimesh.Trimesh(vertices=points,faces=tri.faces,process=True)
        if solid.is_watertight and solid.volume>0:
            inertia=solid.moment_inertia/solid.mass
            if np.isfinite(inertia).all() and np.linalg.eigvalsh(inertia).min()>0:
                self.cad_mass_properties[path]=dict(center=np.asarray(solid.center_mass),inertia_per_kg=inertia)
        from scipy.spatial import ConvexHull
        self.envelopes[path]=points[ConvexHull(points).vertices].tolist()
        geom=UsdGeom.Mesh.Define(self.stage,path+'/CAD')
        geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        faces=np.asarray(tri.faces,dtype=np.int32)
        geom.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,dtype=np.int32)))
        geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten()))
        geom.CreateSubdivisionSchemeAttr('none');geom.CreateDoubleSidedAttr(True)
        self.shape(geom,[0,0,0],mat)
        UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr('convexDecomposition')
        return [lo.tolist(), hi.tolist()]

    def object(self, obj):
        path, mat = self.root(obj); x,y,z=obj['size_m']; k=obj['kind']
        bounds=[[-x/2,-y/2,0],[x/2,y/2,z]]
        if k in ('open_box','lid','sorter'):
            self.box(path,[x,y,z],obj['wall_m'],mat,k=='lid')
            if k=='sorter':
                t=obj['wall_m'];nx,ny=obj['grid'];cells={}
                for i in range(1,nx):self.cube(path+f'/PartitionX{i}',[t,y,z],[-x/2+x*i/nx,0,z/2],mat)
                for j in range(1,ny):self.cube(path+f'/PartitionY{j}',[x,t,z],[0,-y/2+y*j/ny,z/2],mat)
                label_mat=self.material('CellLabels',[.94,.94,.9])
                for n in range(1,10):
                    r,c=divmod(n-1,ny); cx=x/2-(r+.5)*x/nx;cy=y/2-(c+.5)*y/ny
                    cells[str(n)]=dict(center_local_m=[cx,cy,t],size_m=[x/nx-2*t,y/ny-2*t,z-t])
                    self.digit(path+f'/Label{n}',n,[cx,cy,t+.00008],label_mat)
                obj={**obj,'cells':cells}
            elif k=='lid':
                white=self.material('LidPrint',[.9,.92,.88])
                self.cube(path+'/PrintStrip',[x*.24,y*.94,.0001],[x*.28,0,z+.0001],white,False)
            obj={**obj,'inner_size_m':[x-2*obj['wall_m'],y-2*obj['wall_m'],z-obj['wall_m']]}
        elif k=='pen':
            self.cylinder(path+'/Body',y/2,x,[0,0,y/2],mat,axis='X')
            dark=self.material(obj['id']+'Trim',[.012,.012,.016])
            self.cylinder(path+'/Cap',y*.52,x*.12,[x*.39,0,y/2],dark,axis='X')
            bounds=[[-x/2,-y*.52,-y*.02],[x/2,y*.52,y*1.02]]
        elif k=='cad':
            bounds=self.cad(path,obj['cad_asset'],mat)
        elif k=='gear':
            r=.010; h=.009
            self.cylinder(path+'/Hub',r,h,[0,0,h/2],mat)
            for i in range(16):
                a=2*math.pi*i/16
                g=self.cube(path+f'/Tooth{i}',[.0035,.0025,h],[r*math.cos(a),r*math.sin(a),h/2],mat)
                g.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat_yaw(a))
            bounds=[[-.012,-.012,0],[.012,.012,h]]
        elif k=='phone':
            self.cube(path+'/Body',[x,y,z],[0,0,z/2],mat)
            glass=self.material('PhoneGlass',[.01,.014,.02])
            self.cube(path+'/Screen',[x-.004,y-.004,.0003],[0,0,z+.00015],glass,False)
            lens=self.material('PhoneLens',[.12,.13,.15])
            self.cylinder(path+'/Lens',.004,.0001,[-x*.42,-y*.33,z+.0004],lens,collision=False)
        elif k=='divider':
            self.box(path,[x,y,z],obj['wall_m'],mat)
        elif k=='usb_plug':
            metal=self.material('USBMetal',[.60,.62,.65])
            self.cube(path+'/Housing',[.015,.012,.008],[-.0035,0,.004],mat)
            self.cube(path+'/Tongue',[.007,.0084,.0026],[.0075,0,.004],metal)
        elif k=='socket':
            cfg=self.task['socket'];oy=cfg['opening_y_m'];oz=cfg['opening_z_m'];t=.001;cz=.006
            # Four separate rails and a rear stop; the mouth remains physically open.
            length=cfg['depth_m'];cx=-x/2+length/2
            for s in (-1,1):
                self.cube(path+f'/Side{str(s).replace("-","m")}',[length,t,oz+2*t],[cx,s*(oy+t)/2,cz],mat)
                self.cube(path+f'/Top{str(s).replace("-","m")}',[length,oy,t],[cx,0,cz+s*(oz+t)/2],mat)
            self.cube(path+'/RearStop',[t,oy+2*t,oz+2*t],[-x/2+length+t/2,0,cz],mat)
            board=self.material('PCB',[.015,.12,.20])
            self.cube(path+'/PCB',[.030,.024,.001],[-.002,0,.002],board)
            obj={**obj,'mouth_local_m':[-x/2,0,cz]}
        elif k=='fixture':
            bounds=self.cad(path,'grip',mat)
        else:raise ValueError(k)
        self.mass(path,obj,bounds)

    def cable(self):
        cfg=self.task['cable'];n=cfg['segments'];length=cfg['length_m']/n;r=cfg['radius_m']
        plug=self.meta['usb_plug'];start=np.asarray(plug['position_m'])+[-.011,0,.004]
        cable_mat=self.material('CableBlue',[.04,.18,.4]);previous=plug['prim_path']
        previous_end=Gf.Vec3f(-.011,0,.004)
        for i in range(n):
            angle=math.pi + 2*math.pi*.87*i/n
            delta=np.array([length*math.cos(angle),length*math.sin(angle),0.])
            center=start+delta/2;name=f'cable_{i:02d}'
            obj=dict(id=name,kind='cable_segment',position_m=center.tolist(),quaternion_wxyz=[math.cos(angle/2),0,0,math.sin(angle/2)],size_m=[length,2*r,2*r],mass_kg=cfg['mass_kg']/n,color=[.04,.18,.4],dynamic=True)
            path,_=self.root(obj)
            geom=UsdGeom.Capsule.Define(self.stage,path+'/Capsule')
            geom.CreateRadiusAttr(r);geom.CreateHeightAttr(length-2*r);geom.CreateAxisAttr('X')
            self.shape(geom,[0,0,0],cable_mat)
            self.mass(path,obj,[[-length/2,-r,-r],[length/2,r,r]])
            joint=UsdPhysics.SphericalJoint.Define(self.stage,f'/World/ArenaJoints/Cable{i:02d}')
            joint.CreateBody0Rel().SetTargets([Sdf.Path(previous)]);joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
            joint.CreateLocalPos0Attr(previous_end);joint.CreateLocalPos1Attr(Gf.Vec3f(-length/2,0,0))
            joint.CreateAxisAttr('X');joint.CreateConeAngle0LimitAttr(cfg['bend_limit_deg']);joint.CreateConeAngle1LimitAttr(cfg['swing_limit_deg'])
            # Match each initial link orientation in the joint frame. Adjacent segments do not collide.
            if i==0:relangle=angle
            else:relangle=2*math.pi*.87/n
            joint.CreateLocalRot0Attr(Gf.Quatf(math.cos(relangle/2),Gf.Vec3f(0,0,math.sin(relangle/2))))
            joint.CreateLocalRot1Attr(Gf.Quatf(1,Gf.Vec3f(0,0,0)));joint.CreateCollisionEnabledAttr(False)
            previous=path;previous_end=Gf.Vec3f(length/2,0,0);start+=delta

    def finish(self):
        for obj in self.task['objects']:self.object(obj)
        if 'cable' in self.task:self.cable()
        self.stage.GetRootLayer().customLayerData=dict(task_id=self.task['task_id'],geometry_status=self.task['geometry_status'],physics_status=self.task['physics_status'])
        self.stage.GetRootLayer().Save()
        result={**self.task,'objects':list(self.meta.values()),'scene_path':self.output.name,'base_scene':'dual_franka_yubi_cup40k_cupfriction_trial','robot_assets_rebuilt':False}
        self.output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        return result


def build(task_id, output=None):
    output=Path(output) if output else ROOT/'scenes'/f'{task_id}.usda'
    return Builder(load_task(task_id),output).finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task',choices=[*TASK_IDS,'all'],default='all')
    args=parser.parse_args()
    for key in TASK_IDS if args.task=='all' else [args.task]:
        value=build(key);print(key,len(value['objects']),len(value['primitives']))

if __name__=='__main__':main()
