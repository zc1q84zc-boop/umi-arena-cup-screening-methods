"""Reversible render-only lab appearance. No collision or control overrides."""
from pathlib import Path

PROFILE_ID = 'wrist_visual_aligned_v3_housing_visible'

def apply_object_appearance(stage):
    """Fixed color response approximations, reapplied after reset colors."""
    from pxr import Gf, Sdf, UsdShade
    previous=stage.GetEditTarget();stage.SetEditTarget(stage.GetSessionLayer())
    plate=UsdShade.Shader(stage.GetPrimAtPath('/World/Objects/Tray/Looks/TrayMaterial/PreviewSurface'))
    if plate:plate.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.24,.35,.26))
    cup=UsdShade.Shader(stage.GetPrimAtPath('/World/Objects/Cup/Looks/CupMaterial/PreviewSurface'))
    # A fixed soft fill approximates the translucent blue plastic in the video.
    # It is a render appearance prior, not a measured optical material model.
    if cup:cup.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.08,.14,.18))
    stage.SetEditTarget(previous)

def apply_visual_alignment(stage, *, intensity=2200.):
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, UsdLux
    import numpy as np
    root='/World/VisualAlignment'
    previous=stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())

    def material(name,color,roughness=.65,metallic=0.,emission=None):
        mat=UsdShade.Material.Define(stage,f'{root}/Looks/{name}')
        sh=UsdShade.Shader.Define(stage,mat.GetPath().AppendChild('Surface'))
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
        sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metallic)
        if emission is not None: sh.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
        return mat
    def bind(prim,mat):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    def box(name,p,half,mat):
        b=UsdGeom.Cube.Define(stage,f'{root}/Lab/{Tf.MakeValidIdentifier(name)}');b.CreateSizeAttr(2.)
        b.AddTranslateOp().Set(Gf.Vec3d(*p));b.AddScaleOp().Set(Gf.Vec3f(*half));bind(b.GetPrim(),mat)
        return b
    white=material('WarmWhite',(.75,.77,.73),.9)
    grey=material('Floor',(.32,.34,.32),.9)
    aluminum=material('Aluminum',(.45,.48,.5),.35,.65)
    black=material('GloveBlack',(.012,.009,.012),.88)
    red=material('GloveRed',(.30,.012,.025),.7)
    screen=material('Screen',(.025,.04,.045),.45,emission=(.025,.045,.06))
    # Keep the genuine motorized jaw silhouette and every physics binding.
    red_faces={}
    for side in ['LeftMount','RightMount']:
        robot=f'/World/Robots/{side}/Panda'
        # Keep the actual mechanical connection visible in every camera.
        # The previous glove appearance approximation left the jaws floating
        # in the head view. This session override changes rendering only.
        for housing in ['fixed_visual','franka_adapter']:
            prim=stage.GetPrimAtPath(f'{robot}/yubi_base/{housing}')
            if prim:UsdGeom.Imageable(prim).CreateVisibilityAttr().Set('inherited')
        for name in ['Body','Adapter','Jaws']:
            shader=stage.GetPrimAtPath(f'{robot}/YubiLooks/{name}/PreviewSurface')
            if shader:
                shader.GetAttribute('inputs:diffuseColor').Set(Gf.Vec3f(.012,.009,.012))
                UsdShade.Shader(shader).CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.88)
        for finger in ['yubi_leftfinger','yubi_rightfinger']:
            mesh=UsdGeom.Mesh(stage.GetPrimAtPath(f'{robot}/{finger}/geometry'))
            if not mesh: continue
            bind(mesh.GetPrim(),black)
            points=np.asarray(mesh.GetPointsAttr().Get());indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());offset=0;chosen=[]
            for fi,count in enumerate(counts):
                centre=points[indices[offset:offset+count]].mean(axis=0);offset+=count
                # Distal shell trim; a visual face subset, not new jaw geometry.
                if centre[2]>.072 and centre[1]>.001:chosen.append(fi)
            subset=UsdGeom.Subset.Define(stage,mesh.GetPath().AppendChild('RedDistalTrim'))
            subset.CreateElementTypeAttr('face');subset.CreateFamilyNameAttr('materialBind')
            subset.CreateIndicesAttr(chosen);bind(subset.GetPrim(),red)
            red_faces[str(mesh.GetPath())]=len(chosen)
    # Procedural cloth is generated from noise, not a pasted training frame.
    table=stage.GetPrimAtPath('/World/Environment/Table/Looks/TableMaterial/PreviewSurface')
    if table:
        table.GetAttribute('inputs:diffuseColor').Set(Gf.Vec3f(.86,.87,.82))
        UsdShade.Shader(table).CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.95)
    cloth=UsdShade.Material.Define(stage,f'{root}/Looks/Cloth')
    sh=UsdShade.Shader.Define(stage,cloth.GetPath().AppendChild('Surface'));sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.95)
    uv=UsdShade.Shader.Define(stage,cloth.GetPath().AppendChild('UV'));uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    tex=UsdShade.Shader.Define(stage,cloth.GetPath().AppendChild('Texture'));tex.CreateIdAttr('UsdUVTexture')
    texture=Path(__file__).resolve().parent/'assets/visual_alignment/cloth.png'
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
    tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
    cloth.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
    m=UsdGeom.Mesh.Define(stage,f'{root}/ClothSurface')
    m.CreatePointsAttr([(-.409,-.649,.75015),(.409,-.649,.75015),(.409,.649,.75015),(-.409,.649,.75015)])
    m.CreateFaceVertexCountsAttr([4]);m.CreateFaceVertexIndicesAttr([0,1,2,3]);m.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(m).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set([(0,0),(1,0),(1,1),(0,1)])
    bind(m.GetPrim(),cloth)
    # Approximate lab context, fixed in 3D; never tracks the cup or the camera.
    box('Floor',(0,0,-.025),(3,3,.025),grey)
    box('FarWall',(2.1,0,1.45),(.025,2.8,1.45),white)
    for y in [-2.,2.]:box('Wall'+str(y),(0,y,1.45),(2.1,.025,1.45),white)
    for y in [-.664,.664]:
        box('Rail'+str(y),(0,y,.765),(.425,.012,.012),aluminum)
        for x in [-.425,.425]:box('Post'+str(x)+str(y),(x,y,1.13),(.012,.012,.37),aluminum)
    for x in [-.425,.425]:box('CrossRail'+str(x),(x,0,.765),(.012,.66,.012),aluminum)
    for i,(x,y) in enumerate([(1.25,-1.1),(1.25,0),(1.25,1.1),(-1.2,-1.2),(-1.2,1.2),(0,-1.4),(0,1.4)]):
        box(f'Desk{i}',(x,y,.72),(.30,.38,.025),white)
        for dx in [-.24,.24]:
            for dy in [-.30,.30]:box(f'Leg{i}_{dx}_{dy}',(x+dx,y+dy,.35),(.018,.018,.35),aluminum)
        box(f'Monitor{i}',(x+.1,y,.96),(.025,.18,.12),black)
        box(f'Display{i}',(x+.073,y,.96),(.002,.165,.105),screen)
        box(f'Stand{i}',(x+.1,y,.79),(.018,.018,.07),black)
    for y in [-1.97,1.97]:
        for x in [-1.5,-.75,0,.75,1.5]:
            box(f'Partition_{x}_{y}',(x,y,1.35),(.012,.012,1.35),aluminum)
        for z in [.85,1.75,2.6]:
            box(f'PartitionRail_{z}_{y}',(0,y,z),(2.,.012,.012),aluminum)
    dome=stage.GetPrimAtPath('/World/Lights/InspectionLight')
    if dome:dome.GetAttribute('inputs:intensity').Set(float(intensity))
    for i,(x,y) in enumerate([(-.7,-.8),(-.7,.8),(.7,-.8),(.7,.8)]):
        lamp=UsdLux.RectLight.Define(stage,f'{root}/Lights/Ceiling{i}')
        lamp.CreateWidthAttr(.7);lamp.CreateHeightAttr(.13);lamp.CreateIntensityAttr(450.)
        lamp.AddTranslateOp().Set(Gf.Vec3d(x,y,2.45))
        lamp.CreateColorAttr(Gf.Vec3f(1.,.98,.94))
    # Brighter material response inside the otherwise very dark opaque cup.
    for obj,mat_name in [('Cup','CupMaterial'),('Tray','TrayMaterial')]:
        shader=UsdShade.Shader(stage.GetPrimAtPath(f'/World/Objects/{obj}/Looks/{mat_name}/PreviewSurface'))
        if shader:shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.32)
    apply_object_appearance(stage)
    stage.SetEditTarget(previous)
    return dict(id=PROFILE_ID,dome_intensity=float(intensity),red_trim_faces=red_faces,
                render_only=True,physics_changed=False,background='fixed procedural 3D lab',
                motor_housing_visuals_hidden=False,
                plate_diffuse_linear_rgb=[.24,.35,.26],cup_emissive_fill_linear_rgb=[.08,.14,.18],
                measured_hand_eye=False)
