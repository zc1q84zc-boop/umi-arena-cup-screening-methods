"""Exercise both official stage prompts against the deployed HTTP contract."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen

from official_cup_prompts import RIGHT_PLACE, LEFT_RETURN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--family', choices=['pi05', 'openwam'], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hardware', required=True)
    args = parser.parse_args()
    model = {'pi05': 'pi05-cup-intersection-30000',
             'openwam': 'openwam-cup-intersection-fullpass-5069'}[args.family]
    fixture = json.loads(args.fixture.read_text())
    base = f'http://127.0.0.1:{args.port}'
    with urlopen(base + '/health', timeout=10) as response:
        health = json.load(response)
    assert health['ready'] and health['model'] == model
    evidence = {'model': model, 'hardware': args.hardware, 'health': health,
                'image_sha256': {side: hashlib.sha256(base64.b64decode(fixture[side + '_jpeg'])).hexdigest()
                                 for side in ('left', 'right')},
                'requests': [], 'status': 'running'}
    for step, prompt in enumerate((RIGHT_PLACE, LEFT_RETURN)):
        payload = {key: fixture[key] for key in ('left_jpeg', 'right_jpeg', 'relative_pose_xyzw', 'gripper_rad')}
        payload.update(prompt=prompt, episode=0, step=step)
        started = time.monotonic()
        request = Request(base + '/infer', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        assert result['model'] == model and result['prompt'] == prompt
        assert result['episode'] == 0 and result['step'] == step
        assert result['future_observation_used'] is False
        assert result['action_timing']['action_hz'] == 10
        assert result['action_timing']['pose_rows'] == result['action_timing']['gripper_rows'] == [0]
        assert len(result['actions']) == 1 and len(result['actions'][0]) == 16
        assert all(math.isfinite(value) for value in result['actions'][0])
        for offset in (3, 10):
            norm = math.sqrt(sum(value * value for value in result['actions'][0][offset:offset + 4]))
            assert abs(norm - 1) < .01, norm
        result['http_wall_ms'] = (time.monotonic() - started) * 1000
        evidence['requests'].append(result)
        args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'model': model, 'stage': step, 'latency_ms': result['latency_ms'],
                          'shape': [1, 16], 'finite': True}), flush=True)
    evidence['status'] = 'ok'
    evidence['verified_requests'] = len(evidence['requests'])
    evidence['task_success_evaluated'] = False
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
