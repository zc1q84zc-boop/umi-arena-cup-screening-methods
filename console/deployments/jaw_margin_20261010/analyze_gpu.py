"""Analyze frozen-target physics trials, including real deformed-cup nodes."""
import argparse
import json
from pathlib import Path
import numpy as np


def rotation(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def jaw_mesh(path,pivot):
    b=path.read_bytes();dtype=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attribute','<u2')])
    v=np.unique(np.frombuffer(b,dtype=dtype,count=int.from_bytes(b[80:84],'little'),offset=84)['vertices'].reshape(-1,3),axis=0)
    return v[v[:,2]>.115].astype(float)-pivot


def analyze(root,package):
    result=json.loads((root/'result.json').read_text())
    assert result['status'] in ('completed','stopped')
    allrows=[json.loads(s) for s in (root/'states.jsonl').read_text().splitlines()]
    online=result['classification']=='live_pi05_left_phase_reset_diagnostic'
    phase_start=0 if online else 870
    endpoint=[r for r in allrows if r['servo_step']%3==2]
    data=[];streak=max_streak=0;first_tip=None
    for r in allrows:
        i=r['model_step'];robot=r['left'];q=np.asarray(robot['joint_positions']);cmd=np.asarray(r['governed_q'])
        a=r['action']['left'];goal=np.asarray(a['arm_joint_targets_rad']+[-.1+.8*a['gripper_open_fraction']])
        cup=r['cup'];tilt=np.degrees(np.arccos(np.clip(rotation(cup['quaternion_wxyz'])[2,2],-1,1)))
        clearance=(cup['deformation']['min_node_world_z_m']-.7505)*1000
        if i>=phase_start:
            streak=streak+1 if tilt<=15 and clearance>10 else 0;max_streak=max(streak,max_streak)
            if first_tip is None and tilt>30:first_tip=r['physics_time_s']
        data.append(dict(model_step=i,physics_time_s=r['physics_time_s'],
            reference_time_s=r.get('reference_time_s',r['physics_time_s']),
            arm_governor_mae_rad=float(abs(goal[:7]-cmd[:7]).mean()),
            arm_plant_mae_rad=float(abs(cmd[:7]-q[:7]).mean()),
            tool_error_mm=float(np.linalg.norm(np.asarray(r['goal']['left']['position_m'])-robot['tool_pose']['position_m'])*1000),
            cup_tilt_deg=float(tilt),cup_clearance_mm=clearance,
            jaw_actual_rad=float(q[7]),jaw_command_rad=float(cmd[7]),jaw_goal_rad=float(goal[7]),
            jaw_opposition_error_rad=float(abs(q[7]+q[8])),
            shape_change_mm=cup['deformation']['max_nodal_shape_change_m']*1000,
            rim_compression_percent=cup['deformation']['rim_compression_fraction']*100))
    windows={}
    windows_to_analyze=[('left_all',870,1599),('pre_lift',1400,1501),('original_lift_window',1502,1507)]
    if online:
        lift_start=next((r['model_step'] for r in data if r['cup_clearance_mm']>10 and r['cup_tilt_deg']<=15),None)
        windows_to_analyze=[('left_all',0,data[-1]['model_step'])]
        if lift_start is not None:windows_to_analyze.append(('first_lift_window',lift_start,lift_start+6))
    for name,lo,hi in windows_to_analyze:
        chosen=[r for r in data if lo<=r['model_step']<=hi]
        windows[name]={k:float(np.mean([r[k] for r in chosen])) for k in ('arm_governor_mae_rad','arm_plant_mae_rad','tool_error_mm')}
        windows[name]['tool_error_p95_mm']=float(np.percentile([r['tool_error_mm'] for r in chosen],95))
        physical=[r for r in allrows if lo<=r['model_step']<=hi]
        velocity=np.asarray([r['left']['joint_velocities'][:7] for r in physical])
        windows[name]['actual_arm_abs_velocity_p95_rad_s']=float(np.percentile(abs(velocity),95))
        windows[name]['actual_arm_abs_acceleration_p95_rad_s2']=float(np.percentile(abs(np.diff(velocity,axis=0)*30),95))
    geo=[]
    pivots={'left_finger':np.array([.015,.025,.045]),'right_finger':np.array([-.015,.025,.045])}
    meshes={name:jaw_mesh(package/f'assets/yubi/meshes/jaw_{name.split("_")[0]}.stl',p) for name,p in pivots.items()}
    npz=np.load(root/'cup_nodes.npz');nodes={int(i):n for i,n in zip(npz['model_steps'],npz['nodes'])}
    for r in endpoint:
        i=r['model_step']
        if i not in nodes:continue
        if not online and not 1400<=i<=1510:continue
        if online and (np.linalg.norm(np.asarray(r['left']['tool_pose']['position_m'])-r['cup']['position_m'])>.15
                       or r['left']['gripper_open_fraction']>.75):continue
        n=nodes[i];out=dict(model_step=i,physics_time_s=r['physics_time_s'],jaws={})
        for name,vertices in meshes.items():
            pose=r['left']['link_poses'][name]
            world=vertices@rotation(pose['quaternion_wxyz']).T+np.asarray(pose['position_m'])
            distances=np.linalg.norm(world[:,None,:]-n[None,:,:],axis=2)
            v,j=np.unravel_index(np.argmin(distances),distances.shape)
            ring=(int(j)-1)//48
            # Four bottom rings plus 12 wall rings; top rim is ring 15.
            rest_height_mm=.5 if ring<=3 else .5+(ring-3)/12*74
            out['jaws'][name]=dict(nearest_node_gap_mm=float(distances[v,j]*1000),cup_node_index=int(j),
                                   cup_node_rest_height_mm=rest_height_mm,near_rim_node=ring>=14,
                                   world_cup_node_m=n[j].tolist(),world_jaw_vertex_m=world[v].tolist())
        geo.append(out)
    summary=dict(gain=result['response_gain'],jaw_response_gain=result.get('jaw_response_gain',result['response_gain']),classification=result['classification'],windows=windows,
        longest_upright_clearance_above10mm_s=max_streak/30,first_left_tilt_over30_time_s=first_tip,
        peak_left_clearance_mm=max(r['cup_clearance_mm'] for r in data if r['model_step']>=phase_start),
        peak_shape_change_mm=max(r['shape_change_mm'] for r in data if r['model_step']>=phase_start),
        peak_rim_compression_percent=max(r['rim_compression_percent'] for r in data if r['model_step']>=phase_start),
        closest_deformed_cup_node_geometry=geo,
        contact_measurement='distal CAD vertices to actual 769 FEM nodes; surface proximity only, not contact forces; node/mesh discretization affects distances',
        parameter_scope=('Live model inference from current images; matched left-phase reset; left closure margin experiment' if online
                         else 'left closure margin only; same fixed Cartesian model pose targets; response4/4; .8rad/s and1.5rad/s²; 240Hz and128 iterations'),
        left_jaw_margin_rad=result.get('left_jaw_margin_rad',0),
        no_full_task_success_claim=True)
    (root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'plot_samples.json').write_text(json.dumps(data))
    print(json.dumps({k:v for k,v in summary.items() if k!='closest_deformed_cup_node_geometry'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--package',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.package)
