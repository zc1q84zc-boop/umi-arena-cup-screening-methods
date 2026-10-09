"""Read-only validation of private direction-probe and official replay data."""
import argparse
import json
from pathlib import Path
import numpy as np
from online_calibration import matrix


def audit(probe, source):
    rows=[json.loads(s) for s in (probe/'direction_probe.jsonl').read_text().splitlines()]
    assert len(rows)==140
    out={'probe':str(probe),'arm_directions':{},'gripper':{},'camera':{}}
    for side in ('left','right'):
        movements={}
        for start,end,axis,sign in [(20,39,0,1),(60,79,2,-1)]:
            a=np.array(rows[start]['robots'][side]['tool_pose']['position_m'])
            b=np.array(rows[end]['robots'][side]['tool_pose']['position_m'])
            delta=b-a
            assert .015<sign*delta[axis]<.025
            assert max(abs(np.delete(delta,axis)))<.003
            movements[rows[start]['phase']]=delta.tolist()
        out['arm_directions'][side]=movements
        close=rows[119]['robots'][side]['gripper_joint_position_rad']
        opened=rows[139]['robots'][side]['gripper_joint_position_rad']
        assert abs(close)<.03 and opened>.3
        out['gripper'][side]={'closed_observed_rad':close,'reopening_after_2s_rad':opened,
            'source_increases_opening':True,'full_aperture_calibrated':False}
        local=[]
        for r in rows:
            b=r['robots'][side]['link_poses']['base']
            m=r['image_metadata'][side+'_wrist']
            assert m['resolution_px']==[640,480] and not m['horizontal_flip_for_dataset']
            c=m['nominal_pose_world_from_gpu_base']
            local.append(np.linalg.inv(matrix(b['position_m'],b['quaternion_wxyz']))@matrix(c['position_m'],c['quaternion_wxyz']))
        out['camera'][side]={'nominal_mount_translation_m':local[0][:3,3].tolist(),
            'nominal_mount_matrix_drift':float(np.ptp(local,axis=0).max()),
            'status':'GPU-base/CAD audit only, not a measured renderer or physical-camera extrinsic',
            'intrinsics_measured':False,'training_image_match_verified':False}
    body_errors=[];world_errors=[];count=0
    for path in source.glob('episode-*.json'):
        d=json.loads(path.read_text());count+=1
        for before,after in zip(d['frames'],d['frames'][1:]):
            for s in (0,1):
                a=np.array(before['observation']['poses_xyzw'][s]);b=np.array(after['observation']['poses_xyzw'][s])
                action=np.array(after['action']['relative_poses_xyzw'][s])
                rotation=matrix(a[:3],a[[6,3,4,5]])[:3,:3]
                body_errors.append(np.linalg.norm(rotation.T@(b[:3]-a[:3])-action[:3]))
                world_errors.append(np.linalg.norm(b[:3]-a[:3]-action[:3]))
    assert max(body_errors)<1e-6
    out['source_contract']={'episodes_checked':count,'body_delta_max_reconstruction_error_m':max(body_errors),
        'incorrect_world_delta_median_error_m':float(np.median(world_errors)),
        'meaning':'delta in previous hand-local basis; raw dx is not universally robot forward',
        'official_cup61164':{'open_frame0_rad':.74091272,'holding_frame60_rad':.46172822},
        'gripper_scale':'0..0.78 source radians -> 0..0.6 simulator radians; sign verified, metric aperture scale provisional'}
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('probe',type=Path);parser.add_argument('source',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=json.dumps(audit(args.probe,args.source),indent=2)
    if args.output: args.output.write_text(report+'\n')
    print(report)
