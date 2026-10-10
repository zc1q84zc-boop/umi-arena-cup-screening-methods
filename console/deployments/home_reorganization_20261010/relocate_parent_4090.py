"""Move the verified Zhangchi workspace beside Jiazhen without legacy aliases."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

HOME = Path('/home/claude')
SOURCE = HOME / 'umi_workspace_zhangchi'
DEST = HOME / 'Corl_Track_1/umi_workspace_zhangchi'
AUDIT_REL = Path('maintenance/parent_relocation_20261010')
NAMES = (
    'dual-franka-yubi-isaac-sim-deploy', 'umi-track1-console-4090-20261009',
    'umi_cup_intersection_v2_20261009', 'console-plate-radius-45mm-20261009',
    'console-precision-default-20261009-8d5e16b6ec82',
    'console-shared-camera-render-20261009', 'console_exports', 'console_snapshots',
)
UNITS = (
    'umi-track1-console-4090.service', 'umi-arena-tasks-4090.service',
    'umi-intersection-pi05-30000-4090-console.service',
    'umi-jaw-margin-pi05-20261010.service',
)
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'maintenance'}
TEXT_EXTS = {'.py', '.sh', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.ini',
             '.cfg', '.conf', '.pth', '.egg-link', '.service', '.html', '.js', '.css', '.usda'}
BINARY_EXTS = {'.safetensors', '.pt', '.bin', '.onnx', '.mp4', '.usdc', '.stl', '.npy', '.npz'}
PATTERN = re.compile(re.escape(str(SOURCE).encode()) + rb'(?=$|[^A-Za-z0-9_.-])')


def rewrite(data):
    return PATTERN.sub(lambda _: str(DEST).encode(), data)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def replace_bytes(path, data):
    mode = path.stat().st_mode & 0o777
    temp = path.with_name(path.name + '.parent-relocation-tmp')
    with temp.open('xb') as stream:
        stream.write(data)
    temp.chmod(mode)
    temp.replace(path)


def replace_link(path, target):
    temp = path.with_name(path.name + '.parent-relocation-tmp')
    os.symlink(target, temp)
    temp.replace(path)


def walk(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in EXCLUDED]
        for entry in (*dirs, *files):
            yield Path(directory) / entry


def own_edits():
    for path in walk(SOURCE):
        if path.is_symlink():
            old = os.readlink(path)
            new = rewrite(old.encode()).decode()
            if not os.path.isabs(old):
                # Moving the whole tree keeps internal relative links valid. External
                # relative links need the original resolved destination preserved.
                resolved = path.resolve()
                if not resolved.is_relative_to(SOURCE):
                    moved_parent = DEST / path.parent.relative_to(SOURCE)
                    new = os.path.relpath(resolved, moved_parent)
            if old != new:
                yield 'symlink', path, old, new
            continue
        if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            continue
        if path.relative_to(SOURCE).parts[0] not in NAMES[:3]:
            continue
        if path.suffix not in TEXT_EXTS and path.parent.name != 'bin' and path.name != 'activate':
            continue
        old = path.read_bytes()
        try:
            old.decode('utf-8')
        except UnicodeDecodeError:
            continue
        new = rewrite(old)
        if old != new:
            yield 'text', path, old, new


def external_edits():
    roots = [(HOME / 'workspace').resolve(),
             (HOME / 'umi_cup_official_prompt_20k_20261009').resolve(),
             HOME / '.config/systemd/user', HOME / '.local/bin']
    seen = set()
    for root in roots:
        assert not root.is_relative_to(SOURCE)
        for path in walk(root):
            if path in seen:
                continue
            seen.add(path)
            if path.is_symlink():
                old = os.readlink(path)
                new = rewrite(old.encode()).decode()
                if new == old:
                    # This catches relative links from shared environments into us.
                    resolved = str(path.resolve())
                    relocated = rewrite(resolved.encode()).decode()
                    if relocated != resolved:
                        new = relocated
                if old != new:
                    yield 'symlink', path, old, new
            elif path.is_file() and path.stat().st_size <= 1024 * 1024 and (
                path.suffix in {'.py', '.sh', '.cfg', '.pth', '.egg-link', '.service'}
                or path.parent.name == 'bin'
            ):
                old = path.read_bytes()
                try:
                    old.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                new = rewrite(old)
                if old != new:
                    yield 'text', path, old, new


def inventory(root):
    result = {}
    for path in walk(root):
        if not path.is_symlink() and path.is_file() and path.suffix in BINARY_EXTS:
            value = path.stat()
            result[str(path.relative_to(root))] = [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]
    return result


def protected_paths():
    result = {}
    for path in (HOME / 'umi_workspace_jiazhen', DEST.parent / 'umi_workspace_jiazhen',
                 DEST.parent / 'umi_cup_official_prompt_20k_20261009'):
        value = path.lstat()
        result[str(path)] = [value.st_dev, value.st_ino, os.readlink(path) if path.is_symlink() else None]
    return result


def preflight():
    assert SOURCE.is_dir() and not SOURCE.is_symlink()
    assert not os.path.lexists(DEST), f'Refuse merging: {DEST}'
    assert SOURCE.stat().st_dev == DEST.parent.stat().st_dev
    assert (DEST.parent / 'umi_workspace_jiazhen').is_dir()
    assert (SOURCE / 'maintenance/home_reorganization_20261010/receipt.json').is_file()
    actual = set(p.name for p in SOURCE.iterdir())
    assert actual == set(NAMES) | {'maintenance'}, f'Unexpected workspace contents: {actual}'
    return {'.': [SOURCE.stat().st_dev, SOURCE.stat().st_ino],
            **{name: [(SOURCE/name).stat().st_dev, (SOURCE/name).stat().st_ino] for name in NAMES}}


def check_active_consumers():
    unexpected = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ')
            paths = cmd + str((proc / 'exe').resolve()).encode() + str((proc / 'cwd').resolve()).encode()
            if str(SOURCE).encode() not in paths:
                continue
            group = (proc / 'cgroup').read_text()
            if not any(unit in group for unit in UNITS):
                unexpected.append({'pid': int(proc.name), 'command': cmd[:400].decode(errors='replace')})
        except (OSError, RuntimeError):
            pass
    assert not unexpected, f'Active consumers outside the managed services: {unexpected}'


def http(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return {'status': response.status, 'body': response.read(4096).decode(errors='replace')}
    except (OSError, urllib.error.URLError) as error:
        return {'status': getattr(error, 'code', None), 'error': str(error)}


def verify_http():
    urls = [f'http://127.0.0.1:{port}{path}' for port, path in (
        (8774, '/'), (8775, '/'), (8774, '/api/catalog'), (8775, '/api/catalog'),
        (18861, '/health'), (18863, '/health'))]
    values = {}
    for attempt in range(60):
        values = {url: http(url) for url in urls}
        if all(value['status'] == 200 for value in values.values()):
            return values
        if attempt % 10 == 0:
            print(json.dumps({'phase': 'waiting_for_services', 'attempt': attempt,
                              'http': {url: value['status'] for url, value in values.items()}}), flush=True)
        time.sleep(2)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    with (HOME / '.cache/umi_zhangchi_relocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        roots = preflight()
        own = list(own_edits())
        external = list(external_edits())
        plan = {'source': str(SOURCE), 'destination': str(DEST),
                'own_text_edits': sum(e[0] == 'text' for e in own),
                'own_link_edits': sum(e[0] == 'symlink' for e in own),
                'external_edits': [{'path': str(p), 'kind': kind} for kind, p, _, _ in external],
                'compatibility_links': False, 'move_shared_workspaces': False}
        if not args.apply:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        runtime_lock = (SOURCE / NAMES[0] / 'console.lock').open('a')
        waited = 0
        while True:
            try:
                fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if waited % 30 == 0:
                    print(json.dumps({'phase': 'waiting_for_evaluation', 'seconds': waited}), flush=True)
                time.sleep(10)
                waited += 10
        roots = preflight()
        check_active_consumers()
        own = list(own_edits())
        external = list(external_edits())
        protected = protected_paths()
        binary_before = inventory(SOURCE)
        active = [unit for unit in UNITS if subprocess.run(
            ['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0]
        audit = SOURCE / AUDIT_REL
        assert not audit.exists(), 'Audit already exists; inspect a previous attempt'
        audit.mkdir(parents=True)
        write_json(audit / 'plan.json', plan)
        write_json(audit / 'before_inventory.json', {'directories': roots, 'binary_files': binary_before,
                                                    'protected_paths': protected, 'active_units': active})
        changed = []
        hashes = {}
        moved = False
        try:
            if active:
                run('systemctl', '--user', 'stop', *active)
            check_active_consumers()
            SOURCE.rename(DEST)
            moved = True
            audit = DEST / AUDIT_REL
            for scope, edits in (('own', own), ('external', external)):
                for kind, old_path, old, new in edits:
                    path = DEST / old_path.relative_to(SOURCE) if scope == 'own' else old_path
                    backup_rel = old_path.relative_to(SOURCE if scope == 'own' else HOME)
                    record = {'scope': scope, 'kind': kind, 'path': str(path), 'old_path': str(old_path)}
                    if kind == 'symlink':
                        assert os.readlink(path) == old, f'Concurrent link change: {path}'
                        record.update(old_target=old, new_target=new)
                        replace_link(path, new)
                    else:
                        assert path.read_bytes() == old, f'Concurrent file change: {path}'
                        backup = audit / ('before_' + scope) / backup_rel
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        backup.write_bytes(old)
                        record.update(backup=str(backup.relative_to(audit)), old_sha256=digest(old), new_sha256=digest(new))
                        replace_bytes(path, new)
                        hashes[digest(old)] = digest(new)
                    changed.append(record)
            console = DEST / NAMES[1]
            # These readiness hashes change only because source paths changed.
            markers = [*(console / 'deployment_validation').glob('*.json'),
                       *(console / 'sim_validation').rglob('verified_probe.json')]
            rebased = []
            for marker in markers:
                old = marker.read_bytes()
                new = old
                for before, after in hashes.items():
                    new = new.replace(before.encode(), after.encode())
                if old != new:
                    backup = audit / 'before_marker_rebase' / marker.relative_to(DEST)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(old)
                    record = {'scope': 'marker', 'kind': 'text', 'path': str(marker),
                              'old_path': str(SOURCE / marker.relative_to(DEST)),
                              'backup': str(backup.relative_to(audit)),
                              'old_sha256': digest(old), 'new_sha256': digest(new)}
                    replace_bytes(marker, new)
                    changed.append(record)
                    rebased.append(str(marker.relative_to(DEST)))
            write_json(audit / 'changed_files.json', changed)
            for name, before in roots.items():
                value = (DEST / name).stat()
                assert [value.st_dev, value.st_ino] == before
            assert not os.path.lexists(SOURCE)
            assert inventory(DEST) == binary_before, 'Binary assets changed'
            assert protected_paths() == protected, 'Shared workspace roots changed'
            broken = [str(path) for path in walk(DEST) if path.is_symlink() and not path.exists()]
            assert not broken, f'Broken moved links: {broken}'
            assert not list(own_edits_at_destination()), 'Unresolved current workspace references'
            for scope, edits in (('own', own), ('external', external)):
                for kind, old_path, _, _ in edits:
                    path = DEST / old_path.relative_to(SOURCE) if scope == 'own' else old_path
                    if kind == 'symlink':
                        assert path.exists(), f'Broken updated reference: {path}'
            run('systemctl', '--user', 'daemon-reload')
            python = DEST / NAMES[0] / '.venv/bin/python'
            probe = "import sys; sys.path.insert(0,sys.argv[1]); import native_4090_console as n; n.configure(); print(n.app.ROOT); print(n.RUNTIME)"
            bootstrap = run(str(python), '-c', probe, str(console)).strip().splitlines()
            if active:
                run('systemctl', '--user', 'start', *active)
        except BaseException as error:
            # Restore exact pre-move configuration and location if mutation fails.
            if active:
                subprocess.run(['systemctl', '--user', 'stop', *active], capture_output=True)
            for item in reversed(changed):
                path = Path(item['path'])
                if item['kind'] == 'symlink':
                    replace_link(path, item['old_target'])
                else:
                    replace_bytes(path, (audit / item['backup']).read_bytes())
            write_json(audit / 'rollback.json', {'error': str(error), 'moved': moved,
                                                'changed_files_restored': len(changed)})
            if moved:
                DEST.rename(SOURCE)
            subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
            if active:
                subprocess.run(['systemctl', '--user', 'start', *active], check=True)
            raise
        http_values = verify_http()
        states = {unit: run('systemctl', '--user', 'show', unit, '-p', 'ActiveState', '--value').strip()
                  for unit in active}
        receipt = {**plan, 'moved': True, 'eight_owned_directories_preserved': True,
                   'directory_inodes_preserved': True, 'binary_files_unchanged': len(binary_before),
                   'jiazhen_and_shared_roots_unchanged': True, 'old_location_absent': not os.path.lexists(SOURCE),
                   'updated_files_and_links': len(changed), 'path_only_marker_rebase': rebased,
                   'console_bootstrap': bootstrap, 'service_states': states,
                   'http': http_values, 'no_new_gpu_evaluation_claim': True,
                   'completed': all(v['status'] == 200 for v in http_values.values())
                                and all(v == 'active' for v in states.values())}
        write_json(audit / 'receipt.json', receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k not in {'http', 'external_edits'}}, ensure_ascii=False, indent=2))
        assert receipt['completed'], 'Move completed but service verification needs repair; inspect receipt'


def own_edits_at_destination():
    # Reuse the exact candidate rules after the tree has moved.
    global SOURCE
    original = SOURCE
    SOURCE = DEST
    try:
        yield from own_edits()
    finally:
        SOURCE = original


if __name__ == '__main__':
    main()
