"""Apply reviewed console defaults, with backups and concurrent-edit checks."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from urllib.request import urlopen

STAGE = Path(__file__).resolve().parent
CONSOLE = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009')
SERVICE = 'umi-track1-console-4090.service'
PLAN = json.loads((STAGE / 'deploy_plan.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get(path):
    with urlopen('http://127.0.0.1:8774' + path, timeout=30) as response:
        return json.load(response)


assert socket.gethostname() == 'benyun-workstation'
active = [r['id'] for r in get('/api/runs') if r['status'] in ('starting', 'running')]
assert not active, f'Native console runs active: {active}'
launcher = next(f for f in PLAN['files'] if f['name'] == 'run_tuned_intersection_4090.sh')
records = {}
for family in ('pi05', 'openwam'):
    path = CONSOLE / f'deployment_validation/intersection_{family}_server.json'
    if not path.exists():
        assert family != 'pi05'
        continue
    original = path.read_bytes()
    record = json.loads(original)
    assert record['simulation_launcher_sha256'] == launcher['expected_sha256'], path
    records[path] = original

for entry in PLAN['files']:
    target = Path(entry['target'])
    assert sha(target) == entry['expected_sha256'], f'Concurrent edit: {target}'
    staged = STAGE / entry['name']
    assert sha(staged) == entry['sha256'], f'Staged file changed: {staged}'
    if staged.suffix == '.py':
        ast.parse(staged.read_text())
    if staged.suffix == '.sh':
        subprocess.run(['bash', '-n', str(staged)], check=True)

backup = STAGE / 'remote_before'
backup.mkdir(mode=0o700, exist_ok=False)
for entry in PLAN['files']:
    shutil.copy2(entry['target'], backup / entry['name'])
for path in records:
    shutil.copy2(path, backup / path.name)

def replace(path, contents):
    temporary = path.with_name(path.name + '.precision-default.tmp')
    mode = path.stat().st_mode & 0o777
    with temporary.open('xb') as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())
    temporary.chmod(mode)
    temporary.replace(path)

subprocess.run(['systemctl', '--user', 'stop', SERVICE], check=True)
try:
    # Recheck after web requests have stopped; independent GPU jobs are untouched.
    for entry in PLAN['files']:
        assert sha(Path(entry['target'])) == entry['expected_sha256'], entry['target']
    for path, original in records.items():
        assert path.read_bytes() == original, f'Concurrent validation edit: {path}'
    for entry in PLAN['files']:
        replace(Path(entry['target']), (STAGE / entry['name']).read_bytes())
    for path, original in records.items():
        record = json.loads(original)
        record['simulation_launcher_sha256'] = launcher['sha256']
        # Serving/checkpoint evidence and readiness gates retain their values.
        replace(path, (json.dumps(record, indent=2, ensure_ascii=False) + '\n').encode())
finally:
    subprocess.run(['systemctl', '--user', 'start', SERVICE], check=True)

for attempt in range(30):
    try:
        catalog = get('/api/catalog')
        break
    except (OSError, ValueError):
        if attempt == 29:
            raise
        time.sleep(1)
assert catalog['default_contact_profile'] == 'pvc_shell_e3000mpa_i128_h240_v2'
assert catalog['default_policy_id'] == 'pi05-cup-intersection-30000'
assert catalog['default_run_mode'] == 'until_success'
policy = next(p for p in catalog['policies'] if p['id'] == catalog['default_policy_id'])
assert policy['ready'], policy
precision = next(p for p in catalog['stiffness_profiles'] if p['id'] == catalog['default_contact_profile'])
assert precision['ready'], precision
prior = get('/api/runs/8d5e16b6ec82')
assert prior['result']['episodes'][0]['full_task_success'] is False
audit = {
    'status': 'deployed_and_api_verified',
    'hostname': socket.gethostname(),
    'deployed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'defaults': {k: catalog[k] for k in ('default_contact_profile', 'default_policy_id', 'default_run_mode')},
    'files': [{**e, 'deployed_sha256': sha(Path(e['target']))} for e in PLAN['files']],
    'prior_run_full_task_success': False,
    'independent_gpu_jobs_untouched': True,
    'server_tests_passed': 25,
}
(STAGE / 'deployment_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: audit[k] for k in ('status', 'hostname', 'defaults')}, ensure_ascii=False))
