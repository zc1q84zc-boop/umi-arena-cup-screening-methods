"""Live bimanual PI0.5 left phase from a reproducible plate-placement reset."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from continuous_targets_candidate import ContinuousTargets
from analyze_gpu import rotation

HERE=Path(__file__).resolve().parent
TUNED=Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1')
sys.path.insert(0,str(TUNED))
from yubi_isaac_sim_env import continuous_targets,run,run_visual_aligned
from yubi_isaac_sim_env.policy_adapter import TrajectoryChunkExecutor,trajectory_executor_profile


def make_reset():
    baseline=Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/runs/smoothing_response_20261010_gain4')
    with (baseline/'states.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['servo_step']==3*869+2:break
        else:raise ValueError('Baseline left phase initial state unavailable')
    source=json.loads((HERE/'frozen_targets.json').read_text())
    # Right joint states come from the original recording, rather than
    # reconstructing them from commanded positions.
    joints={side:{} for side in ('left','right')}
    import csv
    log=Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/runs/console_612620c03d8a/joints.csv')
    with log.open() as stream:
        for item in csv.DictReader(stream):
            sample=int(item['sample_index'])
            if sample>2610:break
            if sample==2610:joints[item['side']][item['joint_name']]=float(item['position'])
    setup=dict(name='smoothing_left_phase_612620c03d8a_reset',id='smoothing_left_phase_reset',
               cup=dict(position_m=row['cup']['position_m'],quaternion_wxyz=row['cup']['quaternion_wxyz'],color='#78c8e4'),
               plate=dict(position_m=row['plate']['position_m'],quaternion_wxyz=row['plate']['quaternion_wxyz'],color='#8eb8a1'),
               initial_arm_joint_rad={side:[joints[side][f'panda_joint{i}'] for i in range(1,8)] for side in joints},
               initial_gripper_open_fraction={side:(joints[side]['yubi_finger_joint']+.1)/.8 for side in joints},
               registration_profile='aligned_v2_left_phase_reset',
               diagnostic_return_origin_m=source['initial_observation']['objects']['cup']['position_m'])
    (HERE/'left_phase_reset.json').write_text(json.dumps(setup,indent=2)+'\n')
    return setup


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gain',type=float,default=8.)
    args=p.parse_args();assert args.gain in (4,8)
    setup=make_reset();package=TUNED/'yubi_isaac_sim_env'
    lock=Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/console.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    args.output.mkdir(mode=0o700,exist_ok=False);os.environ['SIM_ADAPTER_AUDIT_DIR']=str(args.output)
    continuous_targets.ContinuousTargets=lambda *a,**kw:ContinuousTargets(*a,**kw,response_gain=args.gain)
    shared=run_visual_aligned.configure();config=json.loads((package/'config.json').read_text())
    config.update(physics_hz=240,policy_hz=30,max_policy_steps=100000)
    policy=run._load_callable(TUNED/'adapters/pi05_intersection_isaac_online_adapter.py','predict')
    result=dict(status='running',response_gain=args.gain,classification='live_pi05_left_phase_reset_diagnostic',
                source_run='612620c03d8a',phase_start_model_step=870,model='pi05-cup-intersection-30000',
                cup_on_plate_by_reset=True,right_arm_commands_held=False,
                both_arm_model_targets_unmodified=True,
                shell_initialized_at_fitted_pose=True,velocities_reset_to_zero=True,
                left_model_targets_unmodified=True,extra_closure=0.,physics_hz=240,solver_iterations=128,
                force_telemetry_available=False,full_task_success=False,
                observation_window_s=100,setup=setup)
    app=None;videos={};nodes=[];node_steps=[];wall=time.monotonic();lift_streak=tip_streak=return_streak=0;lifted=False
    try:
        app,env=run.create_sim(scene='dual_franka_yubi_official_fingertip_friction_trial',gui=False,
            setup=setup,seed=42,task_config=config,head_camera_calibration=package/'head_camera_online_aligned_v2.json')
        state=env.observe();env.configure_online_continuous();result['initial_observation']=state
        assert env.cup_physics_profile['solver_position_iterations']==128 and env.cup_physics_profile['physics_hz']==240
        result['cup_physics_profile']=env.cup_physics_profile
        executor=TrajectoryChunkExecutor(policy_hz=30,**trajectory_executor_profile('franka-lookahead'))
        specs={n:run._camera_spec(n,package/'head_camera_online_aligned_v2.json') for n in ('head','left_wrist','right_wrist')}
        cameras={n:run._make_camera(s,gui=False,recording_video=True) for n,s in specs.items()}
        os.environ['PATH']='/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/.venv/bin:'+os.environ['PATH']
        import shutil
        ffmpeg=shutil.which('ffmpeg');result['ffmpeg']=ffmpeg
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-f','lavfi','-i','color=s=64x64',
                        '-frames:v','1','-c:v','libx264','-preset','veryfast','-f','null','-'],check=True)
        videos={n:run.VideoWriter(args.output/f'{n}.mp4',640,480,30) for n in specs}
        initial_cup_z=state['objects']['cup']['position_m'][2];origin=np.asarray(setup['diagnostic_return_origin_m'])
        reason='observation_window_complete';latest_step=-1
        with (args.output/'states.jsonl').open('x') as log:
            for i in range(1000):
                latest_step=i
                if (args.output/'stop').exists():reason='user_stop';break
                observed={**run._policy_observation(state,cameras,specs,env.world),'task_stage':'return_to_origin'}
                if i % 50 == 0:
                    from PIL import Image
                    for view, frame in observed['images'].items():
                        Image.fromarray(frame).save(args.output/f'{view}_latest.jpg')
                    (args.output/'snapshot.json').write_text(json.dumps(dict(
                        model_step=i,physics_time_s=state['physics_time_s'],
                        source='Exact rendered arrays supplied to this model request')))
                chunk=policy(observed,i,0);goal=copy.deepcopy(chunk['waypoints'][0])
                def predict(obs,step,episode):return chunk
                for sub in range(3):
                    action=executor.act(observed if sub==0 else state,3*i+sub,0,predict)
                    state,_,_,_,_=env.step(action);cup=state['objects']['cup'];left=state['robots']['left']
                    cmd=env.current_targets[env.robot_index['left'],env.command_dof_indices].detach().cpu().numpy().tolist()
                    tilt=float(np.degrees(np.arccos(np.clip(rotation(cup['quaternion_wxyz'])[2,2],-1,1))))
                    lift=cup['position_m'][2]-initial_cup_z
                    valid=lift>=.05 and tilt<=15 and left['gripper_open_fraction']<.65
                    lift_streak=lift_streak+1 if valid else 0;lifted|=lift_streak>=10
                    tip_streak=tip_streak+1 if tilt>45 else 0
                    returned=(lifted and tilt<=15 and left['gripper_open_fraction']>=.65
                              and np.linalg.norm(np.asarray(cup['position_m'])[:2]-origin[:2])<=.035
                              and abs(cup['position_m'][2]-origin[2])<=.008
                              and np.linalg.norm(cup['linear_velocity_m_s'])<=.05)
                    return_streak=return_streak+1 if returned else 0
                    row=dict(model_step=i,servo_step=3*i+sub,physics_time_s=state['physics_time_s'],goal=goal,
                             action=action,left=left,right=state['robots']['right'],cup=cup,plate=state['objects']['plate'],governed_q=cmd,
                             interarm_guard=executor.last_interarm_guard,stable_lift_confirmed=lifted,cup_tilt_deg=tilt)
                    log.write(json.dumps(row)+'\n')
                    run._sample(videos['head'],None,cameras['head'],env.world,specs['head'],0,3*i+sub,'step',i,state,
                                {n:videos[n] for n in ('left_wrist','right_wrist')},cameras,specs)
                nodes.append(env.objects['cup'].nodes().copy());node_steps.append(i)
                if i%50==0:
                    progress=dict(step=i,physics_time_s=state['physics_time_s'],wall_s=time.monotonic()-wall,
                                  stable_lift_confirmed=lifted,cup_tilt_deg=tilt)
                    (args.output/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
                if return_streak>=15:reason='left_return_success';break
                if tip_streak>=60 and not lifted:reason='sustained_cup_tipover';break
            result.update(status='stopped' if reason=='user_stop' else 'completed',stop_reason=reason,
                          stable_lift_confirmed=lifted,left_return_success=return_streak>=15,
                          model_requests=latest_step+1,last_state=state,wall_s=time.monotonic()-wall,
                          shared_camera_render=dict(shared.stats))
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:
        if nodes:np.savez_compressed(args.output/'cup_nodes.npz',model_steps=np.asarray(node_steps),nodes=np.array(nodes))
        errors=[]
        for v in videos.values():
            try:v.close()
            except Exception as e:errors.append(repr(e))
        if errors:result.update(status='failed',encoder_errors=errors)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        shared.close()
        if app is not None:app.close()
        lock.close()


if __name__=='__main__':main()
