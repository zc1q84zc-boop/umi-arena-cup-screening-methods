"""Publish serving readiness only after valid inference and owned cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ROOT=Path('/home/claude/workspace/umi_cup_intersection_models_4090_20261009')
CONSOLE=Path('/home/claude/umi-track1-console-4090-20261009')
TUNED=CONSOLE/'simulator_profiles/tuned_v1'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('family',choices=['pi05','openwam'])
    args=parser.parse_args(); family=args.family
    assert socket.gethostname()=='benyun-workstation'
    number='30000' if family=='pi05' else '5069'
    unit=f'umi-intersection-{family}-{number}-4090-console.service'
    state=dict(line.split('=',1) for line in subprocess.check_output(
        ['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState'],text=True).splitlines())
    assert state['MainPID']=='0' and state['ActiveState']=='inactive'
    data=json.loads((ROOT/f'validation/{family}_gpu_preflight.json').read_text())
    assert data['status']=='ok' and data['verified_requests']==2 and data['hardware']=='RTX4090 GPU1'
    assert data['health']['checkpoint']==str(ROOT/f'{family}/{number}')
    for row in data['requests']:
        assert row['model']==data['model'] and row['future_observation_used'] is False
        assert row['action_timing']['pose_rows']==row['action_timing']['gripper_rows']==[0]
    paths={
        'server_sha256':ROOT/'scripts/intersection_server.py',
        'adapter_sha256':TUNED/'adapters/intersection_adapter.py',
        'entry_sha256':TUNED/f'adapters/{family}_intersection_isaac_online_adapter.py',
        'launcher_sha256':ROOT/f'scripts/run_{family}_intersection_{number}.sh',
        'simulation_launcher_sha256':ROOT/'scripts/run_tuned_intersection_4090.sh',
    }
    data.update({key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()})
    if family=='openwam':
        data['cpu_offload_wrapper_sha256']=hashlib.sha256((ROOT/'scripts/intersection_4090_cpu_text_server.py').read_bytes()).hexdigest()
        data['cpu_text_helper_sha256']=hashlib.sha256(Path('/home/claude/workspace/umi_cup_models_4090_20261009/openwam/scripts/openwam_cpu_text_encoder_4090.py').read_bytes()).hexdigest()
    data.update(model_service_cleanup='stopped',task_success_evaluated=False)
    path=CONSOLE/f'deployment_validation/intersection_{family}_server.json'
    if path.exists():
        previous=path.read_bytes()
        archive=path.with_name(path.stem+'_'+hashlib.sha256(previous).hexdigest()[:12]+'.json')
        if not archive.exists():archive.write_bytes(previous)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)
    print(json.dumps({'family':family,'status':data['status'],'service_cleanup':'stopped','task_success_evaluated':False}))

if __name__=='__main__':main()
