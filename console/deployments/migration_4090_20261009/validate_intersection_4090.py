"""Bounded OpenWAM 5069 contract probe; no simulation or task-success claim."""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path('/home/claude/workspace/umi_cup_intersection_models_4090_20261009')
RUNTIME = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy')
UNIT = 'umi-intersection-openwam-5069-4090-console.service'
MODEL = 'openwam-cup-intersection-fullpass-5069'


def command(args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


def state():
    return dict(line.split('=', 1) for line in command([
        'systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '-p', 'MainPID',
        '-p', 'ExecStart', '-p', 'InvocationID', '-p', 'ControlGroup']).splitlines())


def check_resources(owned_pid=None):
    for gpu in (0, 1):
        pids = command(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                        '--format=csv,noheader']).splitlines()
        if any(pid.strip() and not (gpu == 1 and pid.strip() == str(owned_pid))
               for pid in pids):
            raise RuntimeError(f'GPU{gpu} has another compute job; stop only this probe')


def check_active_runs():
    for port in (8774, 8775):
        try:
            with urlopen(f'http://127.0.0.1:{port}/api/runs', timeout=5) as response:
                data = json.load(response)
        except (URLError, TimeoutError):
            if port == 8774:
                raise
            continue
        rows = data.get('runs', []) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise RuntimeError('Unrecognized console run inventory')
        if any(r.get('status') in ('starting', 'running') for r in rows):
            raise RuntimeError(f'Console {port} has an active run; refusing startup')


def check_process_identity(pid):
    args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    allowed = {str(ROOT / 'scripts/run_openwam_intersection_5069.sh').encode(),
               str(ROOT / 'scripts/intersection_4090_cpu_text_server.py').encode()}
    if not allowed.intersection(args) or Path(f'/proc/{pid}').stat().st_uid != os.getuid():
        raise RuntimeError('Unexpected model process identity')


def check_empty_cgroup(group):
    # An inactive unit may clear ControlGroup; never inspect the root cgroup.
    if not group or group == '/':
        return
    cgroup = Path('/sys/fs/cgroup') / group.lstrip('/') / 'cgroup.procs'
    if cgroup.exists() and cgroup.read_text().strip():
        raise RuntimeError('Owned unit cgroup is not empty')


def main():
    if socket.gethostname() != 'benyun-workstation':
        raise RuntimeError('Wrong deployment host')
    output = ROOT / 'validation/openwam_gpu_preflight.json'
    if output.exists():
        raise RuntimeError('Preserve prior probe; inspect its result before retrying')
    integrity = json.loads((ROOT / 'validation/openwam_checkpoint_integrity.json').read_text())
    if integrity.get('model') != MODEL or integrity.get('all_sha256_verified') is not True:
        raise RuntimeError('Checkpoint hash verification missing')
    before = state()
    if (before['MainPID'] != '0' or before['ActiveState'] != 'inactive'
            or str(ROOT / 'scripts/run_openwam_intersection_5069.sh') not in before['ExecStart']):
        raise RuntimeError('Dedicated model unit is not an idle, identified launcher')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Probe interrupted by signal {signum}')

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    with (RUNTIME / 'console.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_active_runs()
        check_resources()
        started = False
        owned_pid = None
        invocation = None
        owned_group = None
        probe = None
        try:
            subprocess.run(['systemctl', '--user', 'start', UNIT], check=True, timeout=20)
            started = True
            current = state()
            owned_pid = int(current['MainPID'])
            invocation = current['InvocationID']
            owned_group = current['ControlGroup']
            if not owned_pid or not invocation:
                raise RuntimeError('Dedicated unit did not acquire a process')
            check_process_identity(owned_pid)
            print(json.dumps({'event': 'loading', 'pid': owned_pid, 'unit': UNIT}), flush=True)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                current = state()
                if current['MainPID'] != str(owned_pid) or current['ActiveState'] != 'active':
                    raise RuntimeError('Owned model server exited while loading')
                check_process_identity(owned_pid)
                check_resources(owned_pid)
                try:
                    with urlopen('http://127.0.0.1:18862/health', timeout=2) as response:
                        health = json.load(response)
                except (URLError, TimeoutError):
                    time.sleep(2)
                    continue
                if (health.get('ready') is not True or health.get('model') != MODEL
                        or health.get('checkpoint') != str(ROOT / 'openwam/5069')):
                    raise RuntimeError('Unexpected serving model/checkpoint')
                break
            else:
                raise TimeoutError('OpenWAM startup exceeded 300 seconds')
            print(json.dumps({'event': 'ready', 'model': MODEL}), flush=True)
            probe = subprocess.Popen([
                str(RUNTIME / '.venv/bin/python'), '-u', str(ROOT / 'scripts/probe_inference.py'),
                '--fixture', str(ROOT / 'validation/observation.json'), '--family', 'openwam',
                '--port', '18862', '--output', str(output), '--hardware', 'RTX4090 GPU1'])
            deadline = time.monotonic() + 400
            while probe.poll() is None:
                check_resources(owned_pid)
                current = state()
                if current['MainPID'] != str(owned_pid) or current['InvocationID'] != invocation:
                    raise RuntimeError('Model process changed during probe')
                if time.monotonic() > deadline:
                    raise TimeoutError('Two-request probe exceeded 400 seconds')
                time.sleep(3)
            if probe.returncode:
                raise RuntimeError(f'Inference probe failed, exit {probe.returncode}')
        finally:
            if probe is not None and probe.poll() is None:
                probe.terminate()
                probe.wait(timeout=10)
            if started:
                current = state()
                if current['InvocationID'] != invocation:
                    raise RuntimeError('Unit was replaced; do not stop a different process')
                subprocess.run(['systemctl', '--user', 'stop', UNIT], check=True, timeout=40)
                current = state()
                if current['MainPID'] != '0':
                    raise RuntimeError('Owned model process did not exit')
                check_empty_cgroup(owned_group)
                print(json.dumps({'event': 'cleanup', 'model_main_pid': 0,
                                  'unit': UNIT, 'active_state': current['ActiveState']}), flush=True)
        check_resources()
        subprocess.run([str(RUNTIME / '.venv/bin/python'),
                        str(ROOT / 'scripts/finalize_intersection_probe.py'), 'openwam'],
                       check=True, timeout=30)


if __name__ == '__main__':
    main()
