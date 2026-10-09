"""GPU render audit of fixed optics and render-only appearance, no policy."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from yubi_isaac_sim_env import create_sim
from yubi_isaac_sim_env.run import _camera_spec, _make_camera, _rendered_rgb
from yubi_isaac_sim_env.wrist_rig import follow_wrist_cameras
from yubi_isaac_sim_env.visual_alignment import apply_visual_alignment, apply_object_appearance

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'renders'
PKG=ROOT/'yubi_isaac_sim_env'

def main():
    OUT.mkdir(exist_ok=False)
    model=json.loads((ROOT/'wrist_camera_visual_aligned_v3.json').read_text())
    setups=json.loads((ROOT/'pose_setups.json').read_text())
    app,env=create_sim(scene='dual_franka_yubi_cup40k_cupfriction_trial',gui=False,
             setup='online_aligned_v2',seed=42,
             head_camera_calibration=PKG/'head_camera_online_aligned_v2.json')
    audit=[]
    try:
        specs={n:_camera_spec(n,PKG/'head_camera_online_aligned_v2.json') for n in ['head','left_wrist','right_wrist']}
        cameras={n:_make_camera(s,False,True) for n,s in specs.items()}
        for phase in ['legacy','optics','aligned']:
            if phase!='legacy':
                for side in ['left','right']:
                    n=side+'_wrist';specs[n]['rigid_mount']=model['per_arm_mount'][side]
                    specs[n]['intrinsics_px']=model['per_arm_intrinsics_px'][side]
                    i=specs[n]['intrinsics_px'];cameras[n].set_opencv_fisheye_properties(
                        cx=i['cx'],cy=i['cy'],fx=i['fx'],fy=i['fy'],fisheye=model['distortion_k1_k4'])
            visual=None
            if phase=='aligned':visual=apply_visual_alignment(env.stage,intensity=model['render_appearance']['dome_intensity'])
            for label,setup in setups.items():
                observation=env.reset(setup=setup,seed=42)
                if phase=='aligned':apply_object_appearance(env.stage)
                follow_wrist_cameras(cameras,specs,observation)
                for n,camera in cameras.items():
                    rgb=_rendered_rgb(camera,env.world,specs[n]);Image.fromarray(rgb).save(OUT/f'{phase}_{label}_{n}.png')
                audit.append(dict(phase=phase,source_frame=int(label),robots=observation['robots'],
                                  objects=observation['objects'],specs=json.loads(json.dumps(specs)),visual=visual))
        (OUT/'render_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        env.stage.GetSessionLayer().Export(str(OUT/'visual_session.usda'))
        print('VISUAL_ALIGNMENT_RENDER_OK',flush=True)
    finally:app.close()

if __name__=='__main__':main()
