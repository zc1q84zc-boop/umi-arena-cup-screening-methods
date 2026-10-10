#!/usr/bin/env python3
"""Bounded saved-observation serving check; not a live grasp/task success test."""
import argparse
import base64
import fcntl
import hashlib
import json
import math
from pathlib import Path
import socket
import signal
import subprocess
import time
from urllib.request import Request, urlopen

ROOT = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009')
RUNTIME = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy')
MODELS = Path('/home/claude/workspace/umi_cup_models_4090_20261009')


def command(args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(unit):
    return dict(line.split('=', 1) for line in command([
        'systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID',
        '-p', 'ExecStart']).splitlines())


def resource_check(owned_pid=None):
    for gpu in (0, 1):
        pids = command(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                        '--format=csv,noheader']).splitlines()
        if any(pid.strip() and not (gpu == 1 and pid.strip() == str(owned_pid)) for pid in pids):
            raise RuntimeError(f'GPU{gpu} has another compute job; refusing or stopping only this probe')


def validate(checkpoint, source):
    if socket.gethostname() != 'benyun-workstation' or ROOT != Path(__file__).parent:
        raise RuntimeError('wrong deployment host/directory')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'probe interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if source.resolve().parent != ROOT / 'sim_runs':
        raise RuntimeError('source must be a saved run of this console')
    unit = f'lingbot-{checkpoint}-squirrel.service'
    port = 18810 if checkpoint == '5000' else 18811
    model_id = f'lingbot-cup-clean-{checkpoint}'
    launcher = MODELS / f'lingbot/run_lingbot_{checkpoint}.sh'
    server = MODELS / 'lingbot/scripts/lingbot_sim_server.py'
    adapter = ROOT / 'simulator_profiles/tuned_v1/adapters/lingbot_isaac_online_adapter.py'
    if str(launcher) not in state(unit)['ExecStart']:
        raise RuntimeError('unit launcher identity mismatch')
    if state(unit)['ActiveState'] == 'active':
        raise RuntimeError('existing service is active; will not take ownership')
    rows = [json.loads(line) for line in (source / 'online_adapter.jsonl').read_text().splitlines()]
    if len(rows) < 3 or [r['step'] for r in rows[:3]] != [0, 1, 2]:
        raise RuntimeError('need three sequential saved observations')
    lock_path = RUNTIME / 'console.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource_check()
        owned_pid = None
        started = False
        results = []
        try:
            subprocess.run(['systemctl', '--user', 'start', unit], check=True, timeout=20)
            started = True
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                current = state(unit)
                owned_pid = int(current['MainPID'])
                if current['ActiveState'] != 'active' or not owned_pid:
                    raise RuntimeError('owned model server exited while loading')
                resource_check(owned_pid)
                try:
                    with urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                        health = json.load(response)
                    if health.get('model') != model_id or health.get('checkpoint') != str(MODELS / f'lingbot/{checkpoint}/hf_ckpt'):
                        raise RuntimeError('wrong model/checkpoint on serving port')
                    if health.get('prompt_protocol') != 'umi_arena_cup_primitives_20261008':
                        raise RuntimeError('server did not confirm current prompt protocol')
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('LingBot readiness deadline exceeded')
            for step, row in enumerate(rows[:3]):
                resource_check(owned_pid)
                if row['observation_origin'] != 'current_simulator_render_and_robot_state':
                    raise RuntimeError('unexpected saved observation origin')
                pose = row['model_input_pose']
                payload = {
                    'episode': 0, 'step': step, 'pose_frame': 'source_hand_reference_259632_v1',
                    'relative_pose_xyzw': row['relative_pose_xyzw'], 'gripper_rad': row['source_gripper_rad'],
                    'prompt': 'Pick up the cup with your right hand and set it on the plate',
                }
                for side in ('left', 'right'):
                    payload[f'{side}_pose_wxyz'] = [*pose[side]['position_m'], *pose[side]['quaternion_wxyz']]
                    payload[f'{side}_jpeg'] = base64.b64encode((source / f'input_{side}_{step:04d}.jpg').read_bytes()).decode('ascii')
                request = Request(f'http://127.0.0.1:{port}/infer', data=json.dumps(payload).encode(),
                                  headers={'Content-Type': 'application/json'}, method='POST')
                with urlopen(request, timeout=120) as response:
                    prediction = json.load(response)
                actions = prediction['actions']
                if (prediction.get('model_id') != model_id or prediction['episode'] != 0 or prediction['step'] != step
                        or prediction.get('pose_mapping') != payload['pose_frame']
                        or prediction.get('prompt') != payload['prompt']
                        or prediction.get('inference_mode') != 'native_lingbot_vla_v2_chunk'
                        or prediction.get('prompt_protocol') != 'umi_arena_cup_primitives_20261008'
                        or prediction.get('action_timing') != {'pose_rows': [1, 2, 3], 'gripper_rows': [0, 1, 2]}
                        or len(actions) != 3 or any(len(action) != 16 for action in actions)
                        or not all(math.isfinite(value) for action in actions for value in action)):
                    raise RuntimeError('invalid causal action/prompt/model identity response')
                results.append({'step': step, 'action_rows': 3, 'latency_ms': prediction['latency_ms']})
            resource_check(owned_pid)
        finally:
            if started:
                current = state(unit)
                # Never stop a replacement process belonging to another run.
                if current['MainPID'] != str(owned_pid or current['MainPID']):
                    raise RuntimeError('owned PID changed; cleanup needs identity review')
                subprocess.run(['systemctl', '--user', 'stop', unit], check=True, timeout=40)
                if state(unit)['MainPID'] != '0':
                    raise RuntimeError('owned model service did not exit')
        return {'status': 'causal_server_probe_passed', 'hostname': socket.gethostname(),
                'checkpoint': checkpoint, 'model_id': model_id, 'source_run': source.name,
                'source_sha256': sha(source / 'online_adapter.jsonl'), 'saved_sim_steps': results,
                'server_sha256': sha(server), 'adapter_sha256': sha(adapter),
                'launcher_sha256': sha(launcher), 'model_service_cleanup': 'stopped',
                'normalization_sha256': sha(MODELS / 'lingbot/norm_stats.json'),
                'live_simulation': False, 'task_success': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', choices=('5000', '10000'))
    parser.add_argument('source_run', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.checkpoint, args.source_run), indent=2))
