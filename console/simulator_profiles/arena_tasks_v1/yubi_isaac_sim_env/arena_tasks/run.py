"""Run tasks 2–5 with existing RTX cameras, GPU PhysX and trajectory control."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from .catalog import ROOT, TASK_IDS, task_catalog


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=TASK_IDS)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--no-randomization',action='store_true')
    parser.add_argument('--record-run',type=Path)
    parser.add_argument('--stop-file',type=Path)
    parser.add_argument('--camera',choices=('head','overview'),default='head')
    parser.add_argument('--policy-script',type=Path,help='policy.py exposing Policy(checkpoint_dir).infer(exact official obs)')
    parser.add_argument('--checkpoint-dir',type=Path)
    parser.add_argument('--adopt-rows',type=int,default=16)
    args=parser.parse_args(argv)
    if args.list:print(json.dumps(task_catalog(),ensure_ascii=False,indent=2));return 0
    if not args.task or not args.record_run:parser.error('--task and --record-run are required')
    if args.steps<=0 or not 1<=args.adopt_rows<=16 or min(args.seed,args.index)<0:parser.error('Invalid steps, seed, index or adopted row count')
    if bool(args.policy_script)!=bool(args.checkpoint_dir):parser.error('Supply both --policy-script and --checkpoint-dir')
    if not (ROOT/'scenes'/f'{args.task}.usda').is_file():parser.error('Build scenes first with -m yubi_isaac_sim_env.arena_tasks.build')
    output=args.record_run.resolve();output.mkdir(parents=True,exist_ok=False)
    os.environ['SIM_ADAPTER_AUDIT_DIR']=str(output)
    from isaacsim import SimulationApp
    selected=int(os.environ.get('ISAAC_ACTIVE_GPU','0'))
    app=SimulationApp(dict(headless=args.headless,active_gpu=selected,physics_gpu=0,multi_gpu=False,renderer='RaytracedLighting'))
    report=dict(status='starting',task_id=args.task,policy='official_adapter' if args.policy_script else 'hold',
                simulator_profile='arena_tasks_v1',policy_input_keys=sorted(__import__(__package__+'.interface',fromlist=['KEYS']).KEYS),
                primitive_count=len(__import__(__package__+'.catalog',fromlist=['load_task']).load_task(args.task)['primitives']),
                scene_geometry='provisional demo reconstruction',task_level_policy_validation=False,seed=args.seed,index=args.index,
                recorded_views=[args.camera,'left_wrist','right_wrist'],object_pose_commands_during_step=False,
                randomization_enabled=not args.no_randomization)
    writers={};joint_writer=None;shared=None;audit=None;grading_log=None;state_log=None;env=None
    try:
        import omni.usd
        from pxr import Gf,UsdGeom,UsdShade,Sdf
        from .env import ArenaTaskEnv
        from .evaluation import PrimitiveEvaluator
        from .interface import observation as wire_observation,action_chunk
        from .. import run as camera_runner
        from ..run_visual_aligned import configure
        from ..visual_alignment import apply_visual_alignment
        from ..head_camera import configure_head_camera,load_head_camera_calibration
        from ..recording import VideoWriter,JointStateWriter
        from ..policy_adapter import TrajectoryChunkExecutor,trajectory_executor_profile
        shared=configure()
        stage_path=ROOT/'scenes'/f'{args.task}.usda'
        context=omni.usd.get_context()
        if not context.open_stage(str(stage_path)):raise RuntimeError('Could not open task USD')
        stage=context.get_stage()
        head_file=ROOT.parent/'head_camera_online_aligned_v2.json'
        configure_head_camera(stage,load_head_camera_calibration(head_file))
        apply_visual_alignment(stage,intensity=2200.)
        if args.task!='phone':
            # The first three demos use a wood tabletop; preserve its collider.
            previous=stage.GetEditTarget();stage.SetEditTarget(stage.GetSessionLayer())
            UsdGeom.Imageable(stage.GetPrimAtPath('/World/VisualAlignment/ClothSurface')).CreateVisibilityAttr().Set('invisible')
            shader=UsdShade.Shader(stage.GetPrimAtPath('/World/Environment/Table/Looks/TableMaterial/PreviewSurface'))
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.28,.14,.06))
            stage.SetEditTarget(previous)
        for _ in range(10):app.update()
        env=ArenaTaskEnv(args.task,render=not args.headless,max_steps=args.steps,randomize=not args.no_randomization)
        env.configure_joint_command_profile('franka-panda-interface')
        state=env.reset(seed=args.seed,scenario_index=args.index)
        # Settle without policy or evaluator updates, then establish the trial origin.
        for _ in range(60):env.world.step(render=True)
        state=env.observe();env.evaluator=PrimitiveEvaluator(env.task,state)
        names=list(dict.fromkeys([args.camera,'head','left_wrist','right_wrist']))
        specs={name:camera_runner._camera_spec(name,head_file) for name in names}
        cameras={name:camera_runner._make_camera(specs[name],not args.headless,True) for name in names}
        for _ in range(10):env.world.render()
        for name in [args.camera,'left_wrist','right_wrist']:
            w,h=specs[name]['resolution'];writers[name]=VideoWriter(output/f'{name}.mp4',w,h,30)
        joint_writer=JointStateWriter(output/'joint_states.csv')
        audit=(output/'wrist_camera_poses.jsonl').open('x')
        grading_log=(output/'evaluation.jsonl').open('x');state_log=(output/'object_states.jsonl').open('x')
        from PIL import Image
        sample_index=0
        def sample(phase='physics'):
            nonlocal sample_index
            current=env.observe()
            camera_runner._sample(writers[args.camera],joint_writer,cameras[args.camera],env.world,specs[args.camera],0,sample_index,phase,env.policy_steps,current,
                extra_videos={n:w for n,w in writers.items() if n!=args.camera},cameras=cameras,specs=specs,camera_audit=audit)
            state_log.write(json.dumps(dict(sample_index=sample_index,physics_time_s=current['physics_time_s'],objects=current['objects']))+'\n');state_log.flush()
            # Current synchronized camera group; this read cannot advance physics.
            frame=shared.last['images'][args.camera]
            if sample_index==0:Image.fromarray(frame).save(output/'scene_preview.jpg')
            Image.fromarray(frame).save(output/'preview.part.jpg');(output/'preview.part.jpg').replace(output/'preview.jpg')
            sample_index+=1
        sample('reset')
        policy=None
        if args.policy_script:
            spec=importlib.util.spec_from_file_location('arena_user_policy',args.policy_script)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            policy=mod.Policy(str(args.checkpoint_dir))
        executor=TrajectoryChunkExecutor(policy_hz=10,**trajectory_executor_profile('franka-lookahead'))
        executor.reset(episode=0);old_index=env.evaluator.index;requests=0
        def predict(robot_state,step,episode):
            nonlocal requests
            images=camera_runner._policy_observation(robot_state,cameras,specs,env.world)['images']
            payload=wire_observation(robot_state,images,env.evaluator.prompt)
            # infer receives exactly the five official fields, with no object poses/contact/center image.
            result=policy.infer(payload);requests+=1
            return action_chunk(result,robot_state,args.adopt_rows)
        substeps_per_frame=env.task_config['physics_hz']//30
        for step in range(args.steps):
            if args.stop_file and args.stop_file.exists():
                env.evaluator.fail('operator_stop');report['user_stopped']=True;break
            if not app.is_running():env.evaluator.fail('application_closed');break
            if env.evaluator.index!=old_index:
                # Retain the policy instance/history; discard the previous instruction's pending actions.
                executor.reset(episode=0);old_index=env.evaluator.index
            action=executor.act(state,step,0,predict) if policy else {}
            state,_,terminated,truncated,info=env.step(action,on_physics_step=lambda tick:sample() if tick%substeps_per_frame==0 else None)
            grading_log.write(json.dumps(dict(step=step,physics_time_s=state['physics_time_s'],**info['task']),ensure_ascii=False)+'\n');grading_log.flush()
            if terminated or truncated:break
        grading=env.evaluator.summary()
        report.update(status='stopped' if report.get('user_stopped') else 'completed',gpu_backend=env.backend,
                      physics_hz=env.task_config['physics_hz'],policy_hz=10,recording_hz=30,video_frames=sample_index,
                      model_requests=requests,task=grading,final_objects=state['objects'],
                      episodes=[dict(policy_steps=env.policy_steps,success=grading['full_task_success'],task=grading)])
        if shared:report['shared_camera_render']=dict(shared.stats)
        # Explicitly report physical geometry/settling validation separately from any policy success.
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        print(report['traceback'],file=sys.stderr,flush=True)
    finally:
        for writer in writers.values():
            try:writer.close()
            except Exception as exc:report.update(status='failed',video_error=str(exc))
        if joint_writer:joint_writer.close()
        for stream in (audit,grading_log,state_log):
            if stream:stream.close()
        if shared:shared.close()
        write_json(output/'report.json',report)
        app.close()
    print(json.dumps({k:report.get(k) for k in ('status','task_id','error','video_frames')},ensure_ascii=False),flush=True)
    return 1 if report['status']=='failed' else 0

if __name__=='__main__':raise SystemExit(main())
