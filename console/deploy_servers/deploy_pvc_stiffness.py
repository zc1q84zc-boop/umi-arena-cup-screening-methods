"""Idle-only, hash-checked deployment of the isolated stiffness trial paths."""
import hashlib
import shlex
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim_console import SSH, SCP

REPO = '/home/lrl/dual-franka-yubi-isaac-sim-console-tuned-v1'
PACKAGE = ROOT/'simulator_profiles/tuned_v1/yubi_isaac_sim_env'
FILES = [
    (PACKAGE/'env.py','yubi_isaac_sim_env/env.py','2092535e4f55caf576d336260c97cd8c666b289c396be639f9bb32a91be9d83e'),
    (PACKAGE/'run.py','yubi_isaac_sim_env/run.py','3b66b6eeedc465da7e3fce0323bdf88c1e3202a1b17d89acdbf5e921edf29d79'),
    (PACKAGE/'pvc_stiffness.py','yubi_isaac_sim_env/pvc_stiffness.py',None),
    (PACKAGE/'pvc_stiffness_probe.py','yubi_isaac_sim_env/pvc_stiffness_probe.py',None),
    (ROOT/'deploy_servers/run_tuned_online_on_squirrel.sh','run_tuned_online.sh','df8d14e9d24909d0498012fb57cfba7f913dad3a95db2d9f1ccc0f1ba73cde4e'),
    (ROOT/'deploy_servers/run_pvc_stiffness_probe_on_squirrel.sh','run_pvc_stiffness_probe.sh',None),
]


def ssh(command):
    return subprocess.run([*SSH,'squirrel_5090',command],check=True,capture_output=True,text=True,timeout=40).stdout.strip()


def main():
    identity = ssh('hostname; id -un; nvidia-smi --query-compute-apps=pid --format=csv,noheader')
    if identity.splitlines() != ['slzl-System-Product-Name','lrl']:
        raise RuntimeError('Unexpected identity or GPU is occupied; no deployment')
    for local, relative, old in FILES:
        target = REPO+'/'+relative
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        current = ssh('if test -e '+shlex.quote(target)+'; then sha256sum '+shlex.quote(target)+'; else echo ABSENT; fi')
        actual = current.split()[0]
        if actual == digest:
            continue
        if actual != (old if old is not None else 'ABSENT'):
            raise RuntimeError('Remote file changed; preserve it: '+target)
        stage = ssh('mktemp '+shlex.quote(target+'.stiffness-stage.XXXXXX'))
        subprocess.run([*SCP,str(local),'squirrel_5090:'+stage],check=True,timeout=40)
        if ssh('sha256sum '+shlex.quote(stage)).split()[0] != digest:
            raise RuntimeError('Stage hash mismatch')
        # Recheck both ownership/idle and target immediately before the atomic promotion.
        command = 'set -eu; test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; '
        if old:
            command += 'test "$(sha256sum '+shlex.quote(target)+' | cut -d " " -f1)" = '+old+'; '
            command += 'cp -n '+shlex.quote(target)+' '+shlex.quote(target+'.before-stiffness-20261008')+'; '
        else:
            command += 'test ! -e '+shlex.quote(target)+'; '
        if target.endswith('.sh'):
            command += 'chmod 700 '+shlex.quote(stage)+'; '
        command += 'mv '+shlex.quote(stage)+' '+shlex.quote(target)+'; sha256sum '+shlex.quote(target)
        print(ssh(command),flush=True)


if __name__ == '__main__': main()
