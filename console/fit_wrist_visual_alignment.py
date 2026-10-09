"""Offline fixed wrist optics fit; color silhouettes are approximate landmarks.

Uses only recorded pre-contact frames and the fixed scene registration. No
runtime image warping, object tracking, policy actions, or physical calibration.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'sim_validation/wrist_visual_alignment_20261008'
PKG = ROOT / 'simulator_profiles/tuned_v1/yubi_isaac_sim_env'
FRAMES = list(range(0, 43, 3))
TRAIN = np.arange(len(FRAMES)) % 2 == 0
A = np.array([[0.,1,0],[-1,0,0],[0,0,1]])
C = np.array([[0.,0,1],[1,0,0],[0,1,0]])
T = np.array([-.209307083410,.202121290786,.749006156995])

def detect(rgb, kind):
    r,g,b = np.moveaxis(rgb.astype(float), -1, 0)
    if kind == 'cup':
        mask = (b-r > 18) & (b-g > 1) & (g-r > 8) & (g > 65)
    else:
        mask = (g-r > 7) & (g-b > 3) & (g > 80) & (r > 60)
    n, lab, stats, centers = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    valid = [i for i in range(1,n) if stats[i,4] > 150]
    if not valid: raise ValueError(f'No {kind} region')
    k = max(valid, key=lambda i: stats[i,4])
    y,x = np.where(lab==k)
    return dict(uv=centers[k].tolist(), area=int(stats[k,4]),
                bbox=[int(x.min()),int(y.min()),int(x.max()),int(y.max())],
                rgb_median=np.median(rgb[lab==k],axis=0).tolist())

def build():
    OUT.mkdir(exist_ok=True, parents=True)
    source = json.loads((ROOT/'data/episode-259632.json').read_text())
    setup = json.loads((PKG/'setups/online_aligned_v2.json').read_text())
    model = json.loads((PKG/'wrist_camera_model.json').read_text())
    # Silhouette color centroid approximates the visible upper cup, not its COM.
    objects = np.array([[*setup['cup']['xy_m'],.75],[*setup['plate']['xy_m'],.75]],float)
    objects[:,2] += [.050,.010]
    prior_t=np.array(model['nominal_mount_in_yubi_frame']['translation_m'])
    prior_r=Rotation.from_euler('x',190,degrees=True)*Rotation.from_euler('z',180,degrees=True)
    fits={}; per_mount={}; per_intrinsics={}
    for ai,side in enumerate(['left','right']):
        cap=cv2.VideoCapture(str(ROOT/f'videos/episode-259632-{side}.mp4'))
        detections=[];bases=[]
        for frame in FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
            rgb=bgr[:,:,::-1]; detections.append([detect(rgb,k) for k in ['cup','plate']])
            cv2.imwrite(str(OUT/f'train_{side}_{frame:03}.png'),bgr)
            hand=np.array(source['poses'][frame][ai]); r=Rotation.from_quat(hand[3:]).as_matrix()
            br=A@r@C
            tool=A@(hand[:3]+r@np.array([.09343,0,0]))+T
            bp=tool-br@np.array([0,.018,.145]);bases.append((bp,br))
        cap.release()
        target=np.array([[d['uv'] for d in ds] for ds in detections])
        # Horizontal plate extent limits an otherwise poorly constrained focal fit.
        widths=np.array([ds[1]['bbox'][2]-ds[1]['bbox'][0] for ds in detections])
        circle=objects[1]+np.column_stack([.095*np.cos(np.linspace(0,2*np.pi,64)),
                        .095*np.sin(np.linspace(0,2*np.pi,64)),np.zeros(64)])
        def project(points,bp,br,x):
            lr=prior_r.as_matrix()@Rotation.from_rotvec(x[:3]).as_matrix()
            eye=bp+br@x[3:6]; v=(points-eye)@(br@lr)
            v=v*np.array([1,-1,-1]);rho=np.linalg.norm(v[:,:2],axis=1)
            theta=np.arctan2(rho,v[:,2]);xy=v[:,:2]*(theta/np.maximum(rho,1e-9))[:,None]
            # RTX 5.1 probe renders use one effective equidistant focal scale:
            # unequal fx/fy attributes did not match the analytic vertical scale.
            # Fit a square-pixel camera so the CPU and renderer agree.
            return xy*x[6]+[320,240]
        def predict(x):
            return np.array([project(objects,bp,br,x) for bp,br in bases])
        def residual(x):
            widths_pred=[]
            for bp,br in bases:
                pix=project(circle,bp,br,x);widths_pred.append(np.ptp(pix[:,0]))
            return np.r_[(predict(x)[TRAIN]-target[TRAIN]).ravel(),
                          (np.array(widths_pred)[TRAIN]-widths[TRAIN])*.35,
                          (x[3:6]-prior_t)*300, x[:3]*1.5,
                          (x[6]-x[7])*.07]
        initial=np.r_[np.zeros(3),prior_t,[280,280]]
        bounds=(np.r_[[-.8]*3,prior_t-.025,[210,210]],
                np.r_[[.8]*3,prior_t+.025,[380,380]])
        result=least_squares(residual,initial,bounds=bounds,loss='soft_l1',f_scale=8,max_nfev=700)
        x=result.x;pred=predict(x);error=np.linalg.norm(pred-target,axis=-1)
        q=(prior_r*Rotation.from_rotvec(x[:3])).as_quat()[[3,0,1,2]]
        per_mount[side]=dict(translation_m=x[3:6].tolist(),quaternion_wxyz=q.tolist(),
                             status='Fixed offline silhouette fit, not measured hand-eye')
        per_intrinsics[side]=dict(fx=float(x[6]),fy=float(x[6]),cx=320,cy=240)
        fits[side]=dict(detections=detections,source_frames=FRAMES,
            fit_frames=np.array(FRAMES)[TRAIN].tolist(),holdout_frames=np.array(FRAMES)[~TRAIN].tolist(),
            predictions_px=pred.tolist(),errors_px=error.tolist(),
            fit_mean_error_px=float(error[TRAIN].mean()),holdout_mean_error_px=float(error[~TRAIN].mean()),
            legacy_mean_error_px=float(np.linalg.norm(predict(np.r_[np.zeros(3),prior_t,
                 [model['intrinsics_px']['fx'],model['intrinsics_px']['fy']]])-target,axis=-1).mean()))
    model.update(schema_version=2,per_arm_mount=per_mount,per_arm_intrinsics_px=per_intrinsics,
                 per_arm_effective_fov_deg={s:{'horizontal':float(np.degrees(640/i['fx'])),
                                               'vertical':float(np.degrees(480/i['fy']))}
                                           for s,i in per_intrinsics.items()},
                 intrinsics_status='Offline episode 259632 pre-contact silhouette fit with held-out frames; not physical camera calibration',
                 alignment_reference_episode=259632,alignment_reference_frames=FRAMES,
                 render_appearance={'dome_intensity':2200.},
                 renderer_projection_status='Square-pixel effective equidistant focal fit after actual RTX 5.1 image audit; unequal focal attributes did not reproduce analytic vertical scaling')
    (OUT/'wrist_camera_visual_aligned_v3.json').write_text(json.dumps(model,indent=2)+'\n')
    audit=dict(reference_episode=259632,objects_centroid_world_m=objects.tolist(),camera_fits=fits,
               measured_hand_eye=False,method='Fixed per-arm rigid extrinsic and equidistant intrinsics, pre-contact frames only; independent alternating holdout frames',
               limitations='Color centroids and fixed object height priors are approximate; held-out RTX render still required')
    (OUT/'offline_fit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({s:{k:v for k,v in f.items() if k.endswith('error_px')} for s,f in fits.items()},indent=2))
    print(json.dumps(per_mount,indent=2));print(json.dumps(per_intrinsics,indent=2))

if __name__=='__main__':build()
