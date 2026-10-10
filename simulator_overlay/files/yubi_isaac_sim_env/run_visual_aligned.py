"""Run the current corrected control pipeline with fixed visual alignment.

This entry point decorates the current runner rather than copying its control
loop. The original ``yubi_isaac_sim_env.run`` remains available for A/B checks.
"""
import hashlib
import json
import os
from pathlib import Path

from . import run as runner
from . import wrist_rig
from .wrist_rig_visual import camera_pose
from .visual_alignment import apply_visual_alignment, apply_object_appearance, PROFILE_ID

PKG=Path(__file__).resolve().parent
PROFILE=PKG/'wrist_camera_visual_aligned_v3.json'

def configure():
    profile=json.loads(PROFILE.read_text())
    intensity=float(profile.get('render_appearance',{}).get('dome_intensity',2200.))
    runner.WRIST_CAMERA_MODEL_PATH=PROFILE
    original_spec=runner._camera_spec
    original_make=runner._make_camera
    original_create=runner.create_sim
    original_write=runner._write_json
    visual={'id':PROFILE_ID,'reference_light_intensity':intensity,
            'finger_visual_material':'black_with_red_distal_trim',
            'background':'fixed_procedural_3D_lab','table':'procedural_white_cloth',
            'fixed_camera_profile':str(PROFILE),'measured_hand_eye_calibration':False,
            'motor_housing_visuals_hidden':False,
            'plate_diffuse_linear_rgb':[.24,.35,.26],
            'cup_emissive_fill_linear_rgb':[.08,.14,.18],
            'render_only':True,'physics_changed':False}
    paths=[Path(__file__),PKG/'shared_camera_render.py',PKG/'visual_alignment.py',PKG/'wrist_rig_visual.py',
           PROFILE,PKG/'assets/visual_alignment/cloth.png']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    shared = None

    def spec(name,head_calibration=None):
        value=original_spec(name,head_calibration)
        if name in ['left_wrist','right_wrist']:
            side=value['robot_side']
            value['rigid_mount']=profile['per_arm_mount'][side]
            value['intrinsics_px']=profile['per_arm_intrinsics_px'][side]
            value['visual_alignment_profile']=PROFILE_ID
            value['effective_fov_deg']=profile['per_arm_effective_fov_deg'][side]
            value['renderer_projection_status']=profile['renderer_projection_status']
        return value

    def create(*args,**kwargs):
        app,env=original_create(*args,**kwargs)
        apply_visual_alignment(env.stage,intensity=intensity)
        original_colors=env._set_material_colors
        def colors(scenario):
            original_colors(scenario)
            apply_object_appearance(env.stage)
        env._set_material_colors=colors
        return app,env

    def make(*args,**kwargs):
        camera=original_make(*args,**kwargs)
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        light=stage.GetPrimAtPath('/World/Lights/InspectionLight')
        light.GetAttribute('inputs:intensity').Set(intensity)
        return camera

    def write(path,value):
        if Path(path).name in ('manifest.json','report.json'):
            value['visual_profile']=visual
            value.setdefault('input_sha256',{}).update(hashes)
            if shared is not None:
                value['shared_camera_render'] = dict(shared.stats)
        original_write(path,value)

    wrist_rig.camera_pose=camera_pose
    runner._camera_spec=spec
    runner._make_camera=make
    runner.create_sim=create
    runner._write_json=write
    mode = os.environ.get('UMI_SHARED_CAMERA_RENDER', '1')
    if mode not in ('0', '1'):
        raise ValueError('UMI_SHARED_CAMERA_RENDER must be 0 or 1')
    if mode == '1':
        from .shared_camera_render import install_shared_camera_render
        shared = install_shared_camera_render(runner)
    return shared

def main():
    shared = configure()
    try:
        return runner.main()
    finally:
        if shared is not None:
            shared.close()

if __name__=='__main__':raise SystemExit(main())
