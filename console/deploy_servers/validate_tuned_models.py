"""Sequential private-model integration probes; never a full-task benchmark."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sim_console import INFERENCE_BACKENDS, REMOTE_ROOT, TUNED_ROOT, SSH
from verify_tuned_online import verify


def api(backend, action):
    request = Request(f'http://127.0.0.1:8772/api/inference/{action}',
        data=json.dumps({'backend':backend}).encode(),
        headers={'Content-Type':'application/json','Origin':'http://127.0.0.1:8772'})
    with urlopen(request,timeout=240) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('backends',nargs='+')
    parser.add_argument('--steps',type=int,default=3)
    args = parser.parse_args()
    for key in args.backends:
        backend = INFERENCE_BACKENDS[key]
        assert backend['host']=='squirrel_5090'
        assert 1 <= args.steps <= 30
        family = 'pi05' if key=='rtx5090' or key.startswith('pi05_') else key.split('_')[0]
        checkpoint = '30000' if key=='rtx5090' else key.split('_')[1]
        run_id = __import__('secrets').token_hex(6)
        remote = str(REMOTE_ROOT/'runs'/f'console_{run_id}')
        unit = f'umi-tuned-validation-{run_id}'
        directory = ROOT/'sim_validation'/f'tuned_online_v1_{family}{checkpoint}_verified'
        if directory.exists():
            raise ValueError(f'Refusing to overwrite {directory}')
        print(f'START {key} run={run_id}',flush=True)
        try:
            status = api(key,'start')
            if not status.get('ready'):
                raise ValueError(status)
            command = shlex.join(['systemd-run','--user',f'--unit={unit}','--collect',
                '--property=RuntimeMaxSec=420','--property=TimeoutStopSec=10',
                'flock','-n',str(REMOTE_ROOT/'console.lock'),
                'env',f"UMI_MODEL_UNIT={backend['unit']}",f"{family.upper()}_ONLINE_URL={backend['adapter_url']}",
                'bash',str(TUNED_ROOT/'run_tuned_online.sh'),remote,f'{family}_isaac_online_adapter.py',
                str(REMOTE_ROOT/'runs'/f'.stop_{run_id}'),str(args.steps)])
            subprocess.run([*SSH,backend['host'],command],check=True,timeout=25)
            deadline = time.monotonic()+440
            while time.monotonic()<deadline:
                done = subprocess.run([*SSH,backend['host'],
                    f'test -f {shlex.quote(remote+"/manifest.json")}'],timeout=20)
                if done.returncode==0:
                    runtime_state = subprocess.run([*SSH,backend['host'],
                        f'systemctl --user show umi-tuned-runtime-{run_id}.service -p ActiveState --value'],
                        capture_output=True,text=True,timeout=20)
                    if runtime_state.stdout.strip() not in ('active','activating','deactivating'):
                        break
                state = subprocess.run([*SSH,backend['host'],
                    f'systemctl --user show {unit}.service -p ActiveState --value'],
                    capture_output=True,text=True,timeout=20)
                if state.returncode or state.stdout.strip() in ('failed','inactive'):
                    log = subprocess.check_output([*SSH,backend['host'],
                        f'journalctl --user -u {unit} -n 8 --no-pager'],text=True,timeout=20)
                    raise RuntimeError(log)
                time.sleep(5)
            else:
                raise TimeoutError(unit)
            # Manifest is written after all encoders and audit streams close.
            subprocess.run(['rsync','-a',f'{backend["host"]}:{remote}/',str(directory)+'/'],
                           check=True,timeout=120)
            result = verify(directory)
            result.update(backend=key,checkpoint=checkpoint,run_id=run_id,
                          policy_id=f'{family}-cup-clean-{checkpoint}',
                          private_checkpoint_path=backend['checkpoint'])
            (directory/'deployment_validation.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f'PASS {key} {json.dumps(result)}',flush=True)
        finally:
            # Scoped transient unit only; a failed probe must not leave Isaac
            # active when its model server is stopped.
            subprocess.run([*SSH,backend['host'],f'systemctl --user stop {unit}.service'],
                           timeout=25,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            print(f'STOP {key} {json.dumps(api(key,"stop"))}',flush=True)
            processes = subprocess.check_output([*SSH,backend['host'],
                'nvidia-smi --query-compute-apps=pid --format=csv,noheader'],text=True,timeout=20)
            if processes.strip():
                raise RuntimeError('GPU0 is not free after task-owned cleanup; refusing next model')


if __name__=='__main__': main()
