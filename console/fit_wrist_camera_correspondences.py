"""Approximate multi-frame camera fit. Color centroids are not measured landmarks."""
import json, subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from online_calibration import MirroredReplayPrior, matrix

ROOT=Path(__file__).resolve().parent
FRAMES=[0,5,10,15,20,25]

def video_frames(name):
    cmd=['ffmpeg','-v','error','-i',str(ROOT/f'videos/episode-136238-{name}.mp4'),
         '-vf','select='+ '+'.join(f'eq(n\\,{n})' for n in FRAMES),'-vsync','0',
         '-f','rawvideo','-pix_fmt','rgb24','-']
    raw=subprocess.check_output(cmd)
    return np.frombuffer(raw,np.uint8).reshape(-1,480,640,3)

def detect(frame,kind):
    r,g,b=np.moveaxis(frame.astype(float),-1,0)
    if kind=='cup': mask=(b-r>20)&(b-g>2)&(g-r>10)&(g>65)
    else: mask=(g-r>7)&(g-b>3)&(g>80)&(r>60)
    # No prior simulator-colored jaw centroids are used as cup landmarks.
    labels,n=label(mask)
    sizes=np.bincount(labels.ravel()); sizes[0]=0
    if not n or sizes.max()<150: raise ValueError('no reliable color region')
    k=sizes.argmax(); ys,xs=np.where(labels==k)
    return dict(uv=[float(xs.mean()),float(ys.mean())],area=int(sizes[k]),
                bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())])

def build():
    data=json.loads((ROOT/'dataset_replay/official_cup_5/episode-136238.json').read_text())
    head=json.loads((ROOT.parent/'dual-franka-yubi-isaac-sim/yubi_isaac_sim_env/head_camera_calibration.json').read_text())
    hc=matrix(head['eye_world_m'],head['quaternion_wxyz'])
    detections={v:{str(i):{k:detect(f,k) for k in ['cup','plate']}
                   for i,f in zip(FRAMES,video_frames(v))} for v in ['center','left','right']}
    objects={}
    for k,z in [('cup',.795),('plate',.755)]:
        # Canonical anatomical +Y is image left in a physically rear-facing
        # camera. Do not retain the legacy horizontal reflection here.
        uv=np.median([detections['center'][str(i)][k]['uv'] for i in FRAMES],axis=0)
        ray=hc[:3,:3]@np.array([(uv[0]-320)/564,-(uv[1]-240)/564,-1])
        objects[k]=hc[:3,3]+ray*((z-hc[2,3])/ray[2])
    cal=MirroredReplayPrior(); results={}
    nominal=np.array([0,.05903,.05837])
    for arm_idx,side in enumerate(['left','right']):
        bases=[]
        for i in FRAMES:
            h=data['frames'][i]['observation']['poses_xyzw'][arm_idx]
            p,q=cal.hand_to_world_tool(side,h[:3],[h[6],*h[3:6]])
            tool_config=json.loads((ROOT.parent/'dual-franka-yubi-isaac-sim/yubi_isaac_sim_env/config.json').read_text())
            offset=np.array(tool_config['yubi']['tool_frame_xyz_m']) if 'yubi' in tool_config else None
            if offset is None:
                offset=np.array(next(v['tool_frame_xyz_m'] for v in tool_config.values() if isinstance(v,dict) and 'tool_frame_xyz_m' in v))
            t=matrix(p,q); t[:3,3]-=t[:3,:3]@offset; bases.append(t)
        target=np.array([[detections[side][str(i)][k]['uv'] for k in ['cup','plate']] for i in FRAMES])
        def predict(x):
            local=Rotation.from_rotvec(x[:3]).as_matrix(); out=[]
            for base in bases:
                eye=base[:3,3]+base[:3,:3]@x[3:]
                r=base[:3,:3]@local
                points=(np.array(list(objects.values()))-eye)@r
                v=np.column_stack([points[:,0],-points[:,1],-points[:,2]])
                rho=np.linalg.norm(v[:,:2],axis=1); theta=np.arctan2(rho,v[:,2])
                out.append(v[:,:2]*(260*theta/np.maximum(rho,1e-9))[:,None]+[320,240])
            return np.array(out)
        def residual(x):
            return np.r_[(predict(x)[::2]-target[::2]).ravel(),(x[3:]-nominal)*500]
        rng=np.random.default_rng(13); best=None
        for _ in range(24):
            x=np.r_[Rotation.random(random_state=rng).as_rotvec(),nominal]
            fit=least_squares(residual,x,bounds=(np.r_[[-6]*3,nominal-.025],np.r_[[6]*3,nominal+.025]),max_nfev=180)
            if best is None or np.linalg.norm(fit.fun)<np.linalg.norm(best.fun): best=fit
        pred=predict(best.x); pixel=np.linalg.norm(pred-target,axis=-1)
        results[side]=dict(translation_m=best.x[3:].tolist(),quaternion_xyzw=Rotation.from_rotvec(best.x[:3]).as_quat().tolist(),
                          fit_frames=FRAMES[::2],check_frames=FRAMES[1::2],
                          errors_px=pixel.tolist(),predictions_px=pred.tolist(),observations_px=target.tolist())
    return dict(episode=136238,frames=FRAMES,objects_world_center_m={k:v.tolist() for k,v in objects.items()},
                detections=detections,camera_fits=results,measured=False,
                caveat='Approximate silhouette-centroid fit with head-camera/height priors; not a measured camera calibration. Held-out frame validation and actual RTX rerender required.')

if __name__=='__main__': print(json.dumps(build(),indent=2))
