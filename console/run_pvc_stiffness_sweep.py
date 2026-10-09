"""Sequential, bounded private-console experiments, hard to soft. No action assistance.

Generated validation markers are accepted only after per-profile physical evidence,
source hashes, video counts, successful main exit and an empty GPU are verified.
An interrupted sweep adopts existing runs; it never overwrites or duplicates them.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

from sim_console import SSH, SCP, ROOT, _PVC_REGISTRY, _require_verified_pvc_probe, atomic_json
from verify_pvc_stiffness import evaluate
from verify_tuned_online import verify

BASE = ROOT/'sim_validation/pvc_stiffness'
REMOTE = '/home/lrl/dual-franka-yubi-isaac-sim-deploy/runs/'
STATE = BASE/'sweep_20261008.json'
PROBE_FILES = ('report.json','deformation.jsonl','video.mp4','nodes_0119.npz','nodes_0479.npz','nodes_0959.npz')
REQUEST = dict(policy='pi05-cup-clean-30000',inference_backend='rtx5090',setup_index=0,
               seed=42,camera='head',steps=600,task_objective='plate_return')


def api(path, body=None):
    request = urllib.request.Request('http://127.0.0.1:8772'+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type':'application/json','Origin':'http://127.0.0.1:8772'})
    return json.load(urllib.request.urlopen(request,timeout=180))


def remote(command):
    p = subprocess.run([*SSH,'squirrel_5090',command],capture_output=True,text=True,timeout=35)
    if p.returncode:
        raise RuntimeError(f'Remote read/command failed ({p.returncode}): {p.stderr[:300]}')
    return p.stdout.strip()


def idle():
    identity = remote('hostname; id -un; nvidia-smi --query-compute-apps=pid --format=csv,noheader')
    if identity.splitlines() != ['slzl-System-Product-Name','lrl']:
        raise RuntimeError('GPU not free or wrong server; preserve the existing process')


def copy_new(source, target):
    target = Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([*SCP,'squirrel_5090:'+source,str(target)],check=True,timeout=90)


def verify_probe(key, name):
    folder = BASE/key/'physical_probe'
    unit = 'umi-'+name.replace('_','-')+'.service'
    state = remote('systemctl --user show '+shlex.quote(unit)+
                   ' -p MainPID -p ExecMainCode -p ExecMainStatus -p ControlGroup -p Result -p ActiveState')
    props = dict(line.split('=',1) for line in state.splitlines() if '=' in line)
    # User-manager/unit GC can clear ExecMainCode, just as with the console's
    # durable-runtime recovery. In that case terminal physical artifacts and
    # all safety checks below, not the cleared unit state, provide the proof.
    terminal = props.get('ActiveState') in ('inactive','failed')
    main_ok = (props.get('ExecMainCode')=='1' and props.get('ExecMainStatus')=='0'
               or props.get('ExecMainCode')=='0' and props.get('Result')=='success' and props.get('ActiveState')=='inactive')
    if props.get('MainPID') != '0' or not terminal or not main_ok or props.get('ControlGroup'):
        raise RuntimeError('Physical probe did not finish cleanly')
    idle()
    for file in PROBE_FILES:
        copy_new(REMOTE+name+'/'+file,folder/file)
    report = json.loads((folder/'report.json').read_text())
    assert report['status']=='completed' and report['purpose']=='physical_platen_compression_not_model_grasp'
    assert report['profile']['id']==key and report['prescribed_vertex_animation'] is False
    assert report['sample_count']==960 and report['video_frames']==240
    samples = [json.loads(x) for x in (folder/'deformation.jsonl').read_text().splitlines()]
    assert len(samples)==960 and [x['physics_step'] for x in samples]==list(range(960))
    assert all(x['platen_tracking_error_m'] < .005 and x['max_nodal_shape_change_m'] < .06 for x in samples)
    for field in ('unloaded_shape_preserved','physical_deformation_observed','recovered_after_release','platen_motion_verified'):
        assert report[field] is True,field
    package = ROOT/'simulator_profiles/tuned_v1/yubi_isaac_sim_env'
    for field,file in (('source_sha256','pvc_stiffness_probe.py'),('shell_sha256','pvc_shell.py'),('registry_sha256','pvc_stiffness.py')):
        assert report[field]==hashlib.sha256((package/file).read_bytes()).hexdigest()
    video = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(folder/'video.mp4')]))['streams'][0]
    assert int(video['nb_read_frames'])==240 and video['r_frame_rate']=='30/1'
    marker = BASE/key/'verified_probe.json'
    if marker.exists():
        assert json.loads(marker.read_text())==report,'Never overwrite a different validation marker'
    else:
        atomic_json(marker,report)
    _require_verified_pvc_probe(key)


def physical(key, record):
    if (BASE/key/'verified_probe.json').exists():
        _require_verified_pvc_probe(key)
        return
    idle()
    name = record.setdefault('probe_name','pvc_stiffness_'+key.split('_')[2]+'_20261008_v1')
    exists = remote('if test -e '+shlex.quote(REMOTE+name)+'; then echo EXISTS; else echo ABSENT; fi')
    if exists == 'ABSENT':
        remote('/home/lrl/dual-franka-yubi-isaac-sim-console-tuned-v1/run_pvc_stiffness_probe.sh '+
               shlex.quote(name)+' '+shlex.quote(key))
    unit = 'umi-'+name.replace('_','-')+'.service'
    deadline = time.monotonic()+660
    while True:
        text = remote('systemctl --user show '+shlex.quote(unit)+' -p MainPID -p ActiveState -p ControlGroup')
        props = dict(line.split('=',1) for line in text.splitlines() if '=' in line)
        pid = props.get('MainPID')
        if pid=='0' and props.get('ActiveState') in ('inactive','failed') and not props.get('ControlGroup'):
            break
        if time.monotonic()>deadline:
            raise RuntimeError('Physical probe exceeded its bounded deadline; inspect dedicated unit')
        print(json.dumps(dict(profile=key,phase='physical_probe',pid=pid)),flush=True)
        time.sleep(45)
    verify_probe(key,name)


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(
        protocol='hard_to_soft_same_pi05_30k_same_scene_seed_no_action_assistance',
        measured_material=False,request=REQUEST,stiffness_order_Pa=[v for _,v in _PVC_REGISTRY['STIFFNESS_SPECS']],
        trials=[],excluded_infrastructure_failure='f122d9c2d0d7')
    if state.get('error'):
        state.setdefault('resolved_observer_events',[]).append(state.pop('error'))
    state['status']='running'
    try:
        for key, modulus in _PVC_REGISTRY['STIFFNESS_SPECS']:
            record = next((r for r in state['trials'] if r['profile']==key),None)
            if record is None:
                record = dict(profile=key,youngs_modulus_Pa=modulus)
                state['trials'].append(record)
            atomic_json(STATE,state)
            if 'result' in record:
                if record['result']['stable_grasp_verified']:
                    state['stop_reason']='stable_grasp_verified_requires_video_review'
                    break
                continue
            runs = api('/api/runs')
            matching = [r for r in runs if r.get('contact_profile')==key and r.get('policy')==REQUEST['policy']
                        and r.get('steps')==600 and r['id']!=state['excluded_infrastructure_failure']]
            if not record.get('run_id') and matching:
                assert len(matching)==1,'Ambiguous existing trials; do not duplicate'
                record['run_id']=matching[0]['id']
            if not record.get('run_id'):
                if any(r['status'] in ('starting','running') for r in runs):
                    raise RuntimeError('Another console run is active; no contention')
                physical(key,record)
                idle()
                record['run_id']=api('/api/runs',{**REQUEST,'contact_profile':key})['id']
                atomic_json(STATE,state)
            run_id = record['run_id']
            deadline = time.monotonic()+50*60
            while True:
                run = api('/api/runs/'+run_id)
                if run['status'] not in ('starting','running'):
                    break
                print(json.dumps(dict(profile=key,phase='model_trial',run_id=run_id,status=run['status'])),flush=True)
                if time.monotonic()>deadline:
                    raise RuntimeError('Model trial tracking exceeded bounded deadline; do not relaunch')
                time.sleep(45)
            if run['status']!='completed':
                raise RuntimeError('Trial failed; inspect without continuing: '+run_id+' '+str(run.get('error')))
            assert run.get('inference_cleanup')=='stopped',run.get('inference_cleanup')
            idle()
            directory=ROOT/'sim_runs'/run_id
            validation=verify(directory)
            result=evaluate(directory)
            assert result['requests']==600 or result['task_success'] is True
            record.update(result=result,integration_validation=validation,cleanup_gpu_free=True,
                          report_sha256=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
                          manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest())
            atomic_json(STATE,state)
            print(json.dumps(dict(phase='verified_result',**result),allow_nan=False),flush=True)
            if result['stable_grasp_verified']:
                state['stop_reason']='stable_grasp_verified_requires_video_review'
                break
        else:
            state['stop_reason']='all_five_stiffness_levels_tested_no_verified_stable_grasp'
        state['status']='completed'
        atomic_json(STATE,state)
        print(json.dumps(dict(status=state['status'],stop_reason=state['stop_reason'])),flush=True)
    except Exception as exc:
        state.update(status='needs_inspection',error=f'{type(exc).__name__}: {exc}')
        atomic_json(STATE,state)
        raise


if __name__ == '__main__': main()
