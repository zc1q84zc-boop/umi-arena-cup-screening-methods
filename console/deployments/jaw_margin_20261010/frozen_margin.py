"""Fixed model Cartesian targets, same physics; change only governor gain."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

import numpy as np
from jaw_margin import adjust

HERE = Path(__file__).resolve().parent
TUNED = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1')
sys.path.insert(0, str(TUNED))
from yubi_isaac_sim_env import continuous_targets, run, run_visual_aligned
from yubi_isaac_sim_env.policy_adapter import TrajectoryChunkExecutor, trajectory_executor_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gain', type=float, required=True)
    parser.add_argument('--jaw-gain', type=float)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase-reset', action='store_true')
    parser.add_argument('--margin', type=float, required=True)
    args = parser.parse_args()
    assert args.gain == 4 and args.jaw_gain == 4 and args.phase_reset
    assert args.margin in (0, .005, .01, .02)
    assert args.jaw_gain is None or args.jaw_gain in (4, 8)
    source = json.loads((HERE/'frozen_targets.json').read_text())
    setup='online_aligned_v2'
    if args.phase_reset:
        setup=json.loads((HERE/'left_phase_reset.json').read_text())
    package = TUNED/'yubi_isaac_sim_env'
    for name, expected in source['expected_sha256'].items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest() == expected, name
    lock = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/console.lock').open('a')
    print('Waiting for console simulation lock',flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    for name, expected in source['expected_sha256'].items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest() == expected, name
    assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    args.output.mkdir(mode=0o700, exist_ok=False)
    os.environ['SIM_ADAPTER_AUDIT_DIR'] = str(args.output)
    shared = run_visual_aligned.configure()
    config = json.loads((package/'config.json').read_text())
    config.update(physics_hz=240, policy_hz=30, max_policy_steps=100000)
    result = dict(status='running',response_gain=args.gain,source_run=source['source_run'],
                  jaw_response_gain=args.gain if args.jaw_gain is None else args.jaw_gain,
                  source_sha256=source['source_sha256'],classification='frozen_model_target_physics_comparison',
                  velocity_rad_s=.8,acceleration_rad_s2=1.5,physics_hz=240,solver_iterations=128,
                  model_inference_requests=0,video_start_simulation_time_s=140,
                  altered_model_targets=True,left_jaw_margin_rad=args.margin,arm_targets_unchanged=True,right_targets_unchanged=True,
                  object_feedback_in_actions=False,closure_scope='left aperture <=.60 full; taper to zero at .65; release unchanged',
                  expected_sha256=source['expected_sha256'])
    if args.phase_reset:
        result.update(classification='frozen_model_target_left_phase_reset_comparison',
                      cup_on_plate_by_reset=True,phase_start_model_step=870,
                      velocities_reset_to_zero=True,shell_initialized_at_fitted_pose=True,
                      reference_time_offset_s=87.)
    app = env = None
    videos = {}; rows=[]; node_steps=[]; nodes=[]; wall=time.monotonic()
    try:
        app, env = run.create_sim(scene='dual_franka_yubi_official_fingertip_friction_trial',gui=False,
            setup=setup,seed=42,task_config=config,
            head_camera_calibration=package/'head_camera_online_aligned_v2.json')
        state = env.observe()
        initial = source['initial_observation']
        if not args.phase_reset:
            for side in ('left','right'):
                np.testing.assert_allclose(state['robots'][side]['joint_positions'],initial['robots'][side]['joint_positions'],atol=2e-6)
            for obj in ('cup','plate'):
                np.testing.assert_allclose(state['objects'][obj]['position_m'],initial['objects'][obj]['position_m'],atol=2e-6)
        result['initial_observation'] = state
        assert env.cup_physics_profile['physics_hz']==240 and env.cup_physics_profile['solver_position_iterations']==128
        result['cup_physics_profile'] = env.cup_physics_profile
        env.configure_online_continuous()
        executor = TrajectoryChunkExecutor(policy_hz=30,**trajectory_executor_profile('franka-lookahead'))
        specs = {name:run._camera_spec(name,package/'head_camera_online_aligned_v2.json')
                 for name in ('head','left_wrist','right_wrist')}
        cameras = {name:run._make_camera(spec,gui=False,recording_video=True) for name,spec in specs.items()}
        # Validate RTX acquisition and the actual encoder before long physics
        # execution; the console's bundled decoder lacks libx264 presets.
        os.environ['PATH']=str(HERE/'bin')+':'+os.environ['PATH']
        ffmpeg=shutil.which('ffmpeg');result['ffmpeg']=ffmpeg
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-f','lavfi','-i','color=s=64x64',
                        '-frames:v','1','-c:v','libx264','-preset','veryfast','-f','null','-'],check=True)
        run._policy_observation(state,cameras,specs,env.world)
        for name in cameras:
            videos[name] = run.VideoWriter(args.output/f'{name}.mp4',640,480,30)
        with (args.output/'states.jsonl').open('x') as log:
            for i, src in enumerate(source['rows'][870:] if args.phase_reset else source['rows']):
                if (args.output/'stop').exists():
                    result['status']='stopped';break
                goal = src['waypoint']
                def predict(obs, step, episode):
                    return dict(action_dt_s=1/30, execute_steps=3,waypoints=[copy.deepcopy(goal) for _ in range(3)])
                for sub in range(3):
                    action = executor.act(state,3*i+sub,0,predict)
                    action, margin_info = adjust(action, args.margin, env.q_closed_rad, env.q_open_rad)
                    state,_,_,_,_ = env.step(action)
                    robot = state['robots']['left']
                    commanded = env.current_targets[env.robot_index['left'],env.command_dof_indices].detach().cpu().numpy().tolist()
                    reference_time=state['physics_time_s']+(87. if args.phase_reset else 0.)
                    row = dict(model_step=src['step'],servo_step=3*i+sub,physics_time_s=state['physics_time_s'],reference_time_s=reference_time,
                               goal=goal,action=action,left=robot,cup=state['objects']['cup'],
                               plate=state['objects']['plate'],governed_q=commanded,
                               interarm_guard=executor.last_interarm_guard,jaw_margin=margin_info)
                    log.write(json.dumps(row)+'\n');rows.append(row)
                    if reference_time >= 140:
                        run._sample(videos['head'],None,cameras['head'],env.world,specs['head'],0,
                                    3*i+sub,'step',i,state,
                                    {n:videos[n] for n in ('left_wrist','right_wrist')},cameras,specs)
                if src['step'] >= 870:
                    nodes.append(env.objects['cup'].nodes().copy());node_steps.append(src['step'])
                if i % 100 == 0:
                    print(json.dumps(dict(gain=args.gain,step=i,sim_time_s=state['physics_time_s'],wall_s=time.monotonic()-wall)),flush=True)
                    (args.output/'progress.json').write_text(json.dumps(dict(step=i,physics_time_s=state['physics_time_s'],wall_s=time.monotonic()-wall)))
            else:
                result['status']='completed'
        result.update(servo_steps=len(rows),last_state=state,wall_s=time.monotonic()-wall,
                      shared_camera_render=dict(shared.stats))
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        if nodes:
            np.savez_compressed(args.output/'cup_nodes.npz',model_steps=np.array(node_steps),nodes=np.array(nodes))
        encoder_errors=[]
        for writer in videos.values():
            try:writer.close()
            except Exception as exc:encoder_errors.append(repr(exc))
        if encoder_errors:result.update(status='failed',encoder_errors=encoder_errors)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        shared.close()
        if app is not None: app.close()
        lock.close()
    print(json.dumps(dict(status=result['status'],gain=args.gain,servo_steps=result.get('servo_steps'),wall_s=result.get('wall_s'))),flush=True)
    if result['status']=='failed':raise RuntimeError('Comparison failed; inspect result.json')


if __name__=='__main__':main()
