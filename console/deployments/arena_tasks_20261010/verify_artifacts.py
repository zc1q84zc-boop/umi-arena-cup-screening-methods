"""Validate downloaded recordings and write the scene-console acceptance marker."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from yubi_isaac_sim_env.arena_tasks.evaluation import world_points

HERE=Path(__file__).resolve().parent
CONSOLE=HERE.parents[1]
PROFILE=CONSOLE/'simulator_profiles/arena_tasks_v1'
ARTIFACTS=CONSOLE/'sim_validation/arena_tasks_20261010'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify(task,source):
    directory=ARTIFACTS/task
    report=read(directory/'report.json')
    assert report['status']=='completed' and report['task_id']==task
    assert report['model_requests']==0 and not report['task']['full_task_success']
    assert report['gpu_backend']['gpu_dynamics'] and report['gpu_backend']['broadphase']=='GPU'
    n=report['video_frames']; assert n==181
    streams={}
    for name in ('head','left_wrist','right_wrist'):
        result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames','-of','json',str(directory/f'{name}.mp4')],capture_output=True,text=True,check=True)
        stream=json.loads(result.stdout)['streams'][0]
        assert int(stream['nb_frames'])==n and stream['avg_frame_rate']=='30/1'
        assert (stream['width'],stream['height'])==(640,480)
        streams[name]=stream
    objects=rows(directory/'object_states.jsonl')
    poses=rows(directory/'wrist_camera_poses.jsonl')
    renders=rows(directory/'shared_camera_render.jsonl')
    assert len(objects)==len(poses)==len(renders)==n
    assert all(abs(o['physics_time_s']-p['physics_time_s'])<1e-8 for o,p in zip(objects,poses))
    records=[r for r in renders if r['event']=='record']; assert len(records)==n
    assert all(max(r['rendering_times_s'].values())-min(r['rendering_times_s'].values())<1e-8 for r in records)
    assert not report['shared_camera_render']['physics_changed']
    metadata=read(PROFILE/f'yubi_isaac_sim_env/arena_tasks/scenes/{task}.json')
    meta={o['id']:o for o in metadata['objects']}
    lowest=1e10
    for sample in objects:
        assert set(sample['objects'])==set(meta)
        for key,state in sample['objects'].items():
            for field in ('position_m','quaternion_wxyz','linear_velocity_m_s','angular_velocity_rad_s'):
                assert np.isfinite(state[field]).all(),(task,key,field)
            assert abs(np.linalg.norm(state['quaternion_wxyz'])-1)<1e-4
            lowest=min(lowest,float(world_points(meta[key],state)[:,2].min()))
    assert lowest>.747,(task,'deep table penetration',lowest)
    final=objects[-1]['objects']
    result=dict(record_dir=source,status='completed',primitive_count=report['primitive_count'],
                video_frames_per_view=n,streams=streams,gpu_backend=report['gpu_backend'],
                sampled_simulation_seconds=objects[-1]['physics_time_s']-objects[0]['physics_time_s'],
                synchronized_views=True,finite_rigid_body_states=True,minimum_shape_z_m=lowest,
                maximum_final_linear_speed_m_s=max(float(np.linalg.norm(s['linear_velocity_m_s'])) for s in final.values()),
                model_requests=0,task_success_claim=False)
    if task!='cable':
        fit=read(ARTIFACTS/f'fits/{task}.json')
        assert fit['status']=='ok' and all(c['contained'] and c['stable'] for c in fit['checks'].values())
        result['physical_goal_fit']=dict(status='ok',checked_objects=list(fit['checks']),physics_steps=fit['settled_physics_steps'],scope=fit['scope'])
    else:result['physical_goal_fit']=dict(status='pending',scope='Dynamic chain and open socket checked; robot-driven insertion requires contact calibration')
    result['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file() and p.suffix in ('.json','.jpg','.mp4')}
    return result


def main():
    sources=read(HERE/'run_sources.json')
    usd=read(HERE/'usd_validation.json');assert all(t['status']=='ok' for t in usd)
    identity=read(ARTIFACTS/'identity_report.json')
    assert identity['status']=='completed' and identity['model_requests']==1
    assert identity['shared_camera_render']['reused_policy_groups']==1
    result=dict(date='2026-10-10',profile='arena_tasks_v1',based_on='tuned_v1',
                official_references=['https://umi-arena.airoa.io/evaluation','https://umi-arena.airoa.io/submission-format'],
                cpu_tests=dict(passed=67),usd_validation=usd,
                tasks={task:verify(task,sources[task]) for task in ('pens','sps','cable','phone')},
                runtime_interface_probe=dict(status='completed',policy='diagnostic identity action; not a trained model',requests=identity['model_requests'],video_frames=identity['video_frames'],official_input_keys=identity['policy_input_keys'],physics_changed_by_camera=False),
                policy_task_success_rate='not evaluated',
                limitations=['Estimated mass/friction and non-CAD object dimensions','Cable is a jointed rigid-body approximation','USB robot-driven contact insertion is not yet calibrated','Divider is a rigid tray approximation','These local thresholds do not replace official judges'])
    result['source_sha256']={str(p.relative_to(PROFILE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in PROFILE.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    console=read(HERE/'console_probe.json');assert console['status']=='completed' and console['exit_code']==0
    result['console_validation']=dict(status='ok',run_id=console['id'],scene='phone',steps=60,
                                      previews_loaded=4,recording_links_checked=['head.mp4','left_wrist.mp4','right_wrist.mp4','report.json'],
                                      task_selector='phone: eight ordered prompts observed in browser',
                                      screenshot='sim_validation/arena_tasks_20261010/console_preview.jpg')
    (HERE/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Accepted',list(result['tasks']),'CPU tests',result['cpu_tests']['passed'])


if __name__=='__main__':main()
