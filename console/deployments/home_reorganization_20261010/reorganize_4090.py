"""Relocate only the eight recorded Zhangchi UMI deployment directories."""
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

HOME_ROOT = Path('/home/claude')
SOURCE = HOME_ROOT / 'Corl_Track_1'
DEST = HOME_ROOT / 'umi_workspace_zhangchi'
NAMES = (
    'dual-franka-yubi-isaac-sim-deploy',
    'umi-track1-console-4090-20261009',
    'umi_cup_intersection_v2_20261009',
    'console-plate-radius-45mm-20261009',
    'console-precision-default-20261009-8d5e16b6ec82',
    'console-shared-camera-render-20261009',
    'console_exports', 'console_snapshots',
)
LIVE = NAMES[:3]
MODEL_UNIT = 'umi-intersection-pi05-30000-4090-console.service'
JAW_MODEL_UNIT = 'umi-jaw-margin-pi05-20261010.service'
UNITS = ('umi-track1-console-4090.service', 'umi-arena-tasks-4090.service', MODEL_UNIT, JAW_MODEL_UNIT)
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'maintenance'}
TEXT_EXTS = {'.py', '.sh', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.pth', '.egg-link', '.service', '.html', '.js', '.css', '.usda'}
BINARY_EXTS = {'.safetensors', '.pt', '.bin', '.onnx', '.mp4', '.usdc', '.stl', '.npy', '.npz'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def rewrite(data):
    for name in NAMES:
        new = str(DEST / name).encode()
        for old in (SOURCE / name, HOME_ROOT / name):
            data = re.sub(re.escape(str(old).encode()) + rb'(?=$|[^A-Za-z0-9_.-])', lambda _: new, data)
    return data


def snapshot():
    values = {}
    for name in NAMES:
        path = SOURCE / name
        assert path.is_dir() and not path.is_symlink(), f'Not an owned source directory: {path}'
        assert not os.path.lexists(DEST / name), f'Target already exists: {DEST / name}'
        legacy = HOME_ROOT / name
        assert not os.path.lexists(legacy) or legacy.is_symlink() and legacy.resolve() == path.resolve(), f'Unrelated legacy path: {legacy}'
        info = path.stat()
        values[name] = {'inode': info.st_ino, 'device': info.st_dev}
    assert not DEST.exists(), 'Refuse merging into an existing workspace'
    shared = SOURCE / 'umi_cup_official_prompt_20k_20261009'
    assert shared.is_symlink() and 'umi_workspace_jiazhen' in str(shared.resolve())
    return values, {'path': str(shared), 'target': os.readlink(shared), 'inode': shared.lstat().st_ino}


def candidates():
    for name in NAMES:
        for directory, dirs, files in os.walk(SOURCE / name, followlinks=False):
            dirs[:] = [d for d in dirs if d not in EXCLUDED]
            for entry in (*dirs, *files):
                path = Path(directory) / entry
                if path.is_symlink():
                    old = os.readlink(path)
                    new = rewrite(old.encode()).decode()
                    if old != new:
                        yield 'symlink', path, old, new
                    continue
                if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
                    continue
                if name not in LIVE:
                    continue
                if path.suffix not in TEXT_EXTS and path.parent.name != 'bin' and path.name != 'activate':
                    continue
                old = path.read_bytes()
                try:
                    old.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                new = rewrite(old)
                if path.name == 'native_4090_console.py' and path.parent.name == NAMES[1]:
                    expected = b"BASE = Path('/home/claude')"
                    assert old.count(expected) == 1
                    new = new.replace(expected, ("BASE = Path('" + str(DEST) + "')\nSHARED_HOME = Path('/home/claude')").encode())
                    new = new.replace(b"MODELS = BASE / 'workspace/", b"MODELS = SHARED_HOME / 'workspace/")
                    new = new.replace(b"INTERSECTION = BASE / 'workspace/", b"INTERSECTION = SHARED_HOME / 'workspace/")
                if old != new:
                    yield 'text', path, old, new


def binary_inventory(root):
    result = {}
    for name in NAMES:
        for directory, dirs, files in os.walk(root / name, followlinks=False):
            dirs[:] = [d for d in dirs if d not in EXCLUDED]
            for entry in files:
                path = Path(directory) / entry
                if path.is_symlink() or path.suffix not in BINARY_EXTS:
                    continue
                value = path.stat()
                result[str(path.relative_to(root))] = [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]
    return result


def shared_reference_edits():
    """Update references to relocated runtime, without moving shared content."""
    seen = set()
    for root in ((HOME_ROOT / 'workspace').resolve(), (HOME_ROOT / 'umi_cup_official_prompt_20k_20261009').resolve()):
        assert root.is_dir() and not any(root.is_relative_to(SOURCE / name) for name in NAMES)
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in EXCLUDED]
            for entry in (*dirs, *files):
                path = Path(directory) / entry
                if path in seen:
                    continue
                seen.add(path)
                if path.is_symlink():
                    old = os.readlink(path); new = rewrite(old.encode()).decode()
                    if new == old:
                        resolved = str(path.resolve())
                        relocated = rewrite(resolved.encode()).decode()
                        if relocated != resolved:
                            new = relocated
                    if old != new:
                        yield 'symlink', path, old, new
                elif path.is_file() and path.stat().st_size <= 1024 * 1024 and (path.suffix in {'.py', '.sh', '.cfg', '.pth', '.egg-link', '.service'} or path.parent.name == 'bin'):
                    old = path.read_bytes()
                    try:
                        old.decode('utf-8')
                    except UnicodeDecodeError:
                        continue
                    new = rewrite(old)
                    if old != new:
                        yield 'text', path, old, new


def check_gpu_scope():
    pids = run('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader').splitlines()
    for value in pids:
        pid = int(value.strip())
        group = (Path('/proc') / str(pid) / 'cgroup').read_text()
        assert any(unit in group for unit in (MODEL_UNIT, JAW_MODEL_UNIT)), f'Active GPU job outside the migration dependency: PID {pid}'


def unit_edits():
    edits = []
    for path in (HOME_ROOT / '.config/systemd/user').glob('*.service'):
        if path.name in UNITS:
            old = path.read_bytes(); new = rewrite(old)
            if old != new: edits.append((path, old, new))
    fragment = run('systemctl', '--user', 'show', JAW_MODEL_UNIT, '--property=FragmentPath', '--value').strip()
    if fragment.startswith('/run/user/'):
        path = HOME_ROOT / '.config/systemd/user' / JAW_MODEL_UNIT
        assert not path.exists(), 'Do not overwrite an existing model unit'
        edits.append((path, None, rewrite(Path(fragment).read_bytes())))
    return edits


def atomic_bytes(path, data):
    mode = path.stat().st_mode & 0o777
    temp = path.with_name(path.name + '.zhangchi-migration-tmp')
    with temp.open('xb') as stream:
        stream.write(data)
    temp.chmod(mode)
    temp.replace(path)


def http_status(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    with (HOME_ROOT / '.cache/umi_zhangchi_relocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        roots, shared = snapshot()
        edits = list(candidates())
        external = list(shared_reference_edits())
        units = unit_edits()
        plan = dict(destination=str(DEST), owned_directories=list(NAMES),
                    excluded_shared_reference=shared, text_edits=sum(e[0] == 'text' for e in edits),
                    symlink_edits=sum(e[0] == 'symlink' for e in edits), unit_edits=[p.name for p, _, _ in units],
                    shared_dependency_reference_updates=[str(e[1]) for e in external],
                    old_paths_will_be_removed=True, compatibility_links_will_be_created=False)
        if not args.apply:
            print(json.dumps(plan, ensure_ascii=False, indent=2)); return
        runtime_lock = (SOURCE / NAMES[0] / 'console.lock').open('a')
        waited = 0
        while True:
            try:
                fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if waited % 30 == 0:
                    print(json.dumps({'phase':'waiting_for_active_evaluation','waited_seconds':waited}), flush=True)
                time.sleep(10); waited += 10
        # Check again after the finished evaluation releases the filesystem lock.
        for attempt in range(6):
            try:
                check_gpu_scope(); break
            except AssertionError:
                if attempt == 5: raise
                time.sleep(5)
        # The evaluation may have created output while we waited; rescan after it ends.
        roots, shared = snapshot()
        edits = list(candidates()); external = list(shared_reference_edits()); units = unit_edits()
        plan.update(text_edits=sum(e[0] == 'text' for e in edits), symlink_edits=sum(e[0] == 'symlink' for e in edits),
                    unit_edits=[p.name for p, _, _ in units], shared_dependency_reference_updates=[str(e[1]) for e in external])
        active = [unit for unit in UNITS if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0]
        before_http = {str(port): http_status(f'http://127.0.0.1:{port}/') for port in (8774, 8775)}
        run('systemctl', '--user', 'stop', *active)
        binaries = binary_inventory(SOURCE)
        DEST.mkdir(mode=0o775)
        audit = DEST / 'maintenance/home_reorganization_20261010'
        audit.mkdir(parents=True)
        write_json(audit / 'plan.json', plan)
        for name in NAMES:
            (SOURCE / name).rename(DEST / name)
        changed = []; hash_changes = {}
        for kind, source_path, old, new in edits:
            relative = source_path.relative_to(SOURCE)
            target = DEST / relative
            if kind == 'symlink':
                assert os.readlink(target) == old
                temp = target.with_name(target.name + '.zhangchi-migration-tmp')
                os.symlink(new, temp); temp.replace(target)
                changed.append(dict(path=str(relative), kind=kind, old_target=old, new_target=new))
            else:
                assert target.read_bytes() == old, f'Concurrent file edit: {target}'
                backup = audit / 'before' / relative
                backup.parent.mkdir(parents=True, exist_ok=True); backup.write_bytes(old)
                atomic_bytes(target, new)
                hash_changes[digest(old)] = digest(new)
                changed.append(dict(path=str(relative), kind=kind, old_sha256=digest(old), new_sha256=digest(new)))
        external_changed = []
        for kind, target, old, new in external:
            relative = target.relative_to(HOME_ROOT)
            if kind == 'symlink':
                assert os.readlink(target) == old
                temp = target.with_name(target.name + '.zhangchi-migration-tmp')
                os.symlink(new, temp); temp.replace(target)
                external_changed.append(dict(path=str(target), kind=kind, old_target=old, new_target=new))
            else:
                assert target.read_bytes() == old, f'Concurrent dependency edit: {target}'
                backup = audit / 'before_shared_references' / relative
                backup.parent.mkdir(parents=True, exist_ok=True); backup.write_bytes(old)
                atomic_bytes(target, new)
                hash_changes[digest(old)] = digest(new)
                external_changed.append(dict(path=str(target), kind=kind, old_sha256=digest(old), new_sha256=digest(new)))
        # These live readiness markers must follow files whose only change was relocation.
        # Original markers remain in the migration backup; no new GPU validation is claimed.
        rebased = []
        console = DEST / NAMES[1]
        markers = [*(console / 'deployment_validation').glob('*.json'), *(console / 'sim_validation').rglob('verified_probe.json')]
        for marker in markers:
            old = marker.read_bytes(); new = old
            for previous, current in hash_changes.items():
                new = new.replace(previous.encode(), current.encode())
            if new != old:
                backup = audit / 'before_hash_rebase' / marker.relative_to(DEST)
                backup.parent.mkdir(parents=True, exist_ok=True); backup.write_bytes(old)
                atomic_bytes(marker, new); rebased.append(str(marker.relative_to(DEST)))
        for path, old, new in units:
            if old is None:
                assert not path.exists(), f'Concurrent unit creation: {path}'
                path.write_bytes(new)
            else:
                assert path.read_bytes() == old, f'Concurrent unit edit: {path}'
                (audit / ('before_' + path.name)).write_bytes(old)
                atomic_bytes(path, new)
        write_json(audit / 'changed_files.json', changed)
        for name, previous in roots.items():
            value = (DEST / name).stat()
            assert [value.st_dev, value.st_ino] == [previous['device'], previous['inode']]
            assert not os.path.lexists(SOURCE / name)
            legacy = HOME_ROOT / name
            if legacy.is_symlink() and legacy.resolve() == DEST / name:
                legacy.unlink()
            assert not os.path.lexists(legacy), f'Legacy entry remains: {legacy}'
        assert binary_inventory(DEST) == binaries, 'Model/video/mesh files changed'
        assert (SOURCE / 'umi_cup_official_prompt_20k_20261009').lstat().st_ino == shared['inode']
        assert os.readlink(SOURCE / 'umi_cup_official_prompt_20k_20261009') == shared['target']
        run('systemctl', '--user', 'daemon-reload')
        python = DEST / NAMES[0] / '.venv/bin/python'
        probe = "import sys; sys.path.insert(0,sys.argv[1]); import native_4090_console as n; n.configure(); print(n.app.ROOT); print(n.RUNTIME)"
        bootstrap = run(str(python), '-c', probe, str(console))
        if active:
            run('systemctl', '--user', 'start', *active)
        after_http = {}
        for attempt in range(15):
            try:
                after_http = {str(port): http_status(f'http://127.0.0.1:{port}/') for port in (8774, 8775)}
                if all(value == 200 for value in after_http.values()): break
            except OSError:
                pass
            time.sleep(1)
        receipt = dict(**plan, moved_directory_count=len(NAMES), original_directory_inodes_preserved=True,
                       binary_files_unchanged=len(binaries), external_shared_reference_unchanged=True,
                       old_source_directories_absent=True, old_home_aliases_absent=True,
                       path_changes=changed, rebased_path_only_readiness_markers=rebased,
                       shared_dependency_reference_changes=external_changed,
                       no_new_gpu_validation_claim=True, console_bootstrap=bootstrap.strip().splitlines(),
                       restarted_units=active, before_http=before_http, after_http=after_http,
                       all_console_pages_available=all(value == 200 for value in after_http.values()))
        write_json(audit / 'receipt.json', receipt)
        assert receipt['all_console_pages_available'], 'Console HTTP verification failed; inspect migration receipt'
        print(json.dumps({k: v for k, v in receipt.items() if k != 'path_changes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
