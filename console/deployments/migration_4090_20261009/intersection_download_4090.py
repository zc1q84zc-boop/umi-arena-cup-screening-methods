"""Single-writer direct LAN transfer; verified immutable PI payload may be reused.

No model imports, inference, GPU calls, SSH auth changes, or checkpoint writes.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
from urllib.request import Request, urlopen
from urllib.parse import quote, urlsplit

from intersection_export_readonly import checked_manifest, MANIFEST_SHA

ROOT = Path('/home/claude/workspace/umi_cup_intersection_models_4090_20261009')
PI_REUSE = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi_cup_intersection_v2_20261009/pi05/30000/inference_export')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def record(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def reuse_candidate(relative):
    parts = relative.parts
    if parts[:2] != ('pi05', '30000') or len(parts) < 3:
        return None
    if parts[2] not in ('params', 'assets', '_CHECKPOINT_METADATA'):
        return None
    candidate = PI_REUSE.joinpath(*parts[2:])
    if not candidate.is_file() or candidate.is_symlink() or candidate.stat().st_uid != os.getuid():
        return None
    if not candidate.resolve().is_relative_to(PI_REUSE.resolve()):
        raise ValueError('Reuse source escaped its dedicated checkpoint')
    return candidate


def transfer(access, root=ROOT, reuse_pi=False, pi_first=False):
    url = urlsplit(access['url'])
    if url.scheme != 'http' or url.hostname != '192.168.110.11' or url.path not in ('', '/'):
        raise ValueError('Export must be the verified A100 LAN source')
    if access['manifest_sha256'] != MANIFEST_SHA or access['expires_unix'] < time.time():
        raise ValueError('Invalid or expired transfer session')
    headers = {'Authorization': 'Bearer ' + access['token']}
    with urlopen(Request(access['url'] + '/manifest.json', headers=headers), timeout=30) as response:
        raw = response.read(1024 * 1024)
    manifest = checked_manifest(raw)
    (root / 'transfer_manifest.json').write_bytes(raw)
    if shutil.disk_usage(root).free < manifest['total_bytes'] + 2 * 1024**3:
        raise ValueError('Insufficient free space for independent verified payload')
    started = time.monotonic()
    deadline = min(access['expires_unix'], time.time() + 7100)
    completed = 0
    results = []
    last_progress = 0

    def progress(path, phase, partial_bytes=0):
        nonlocal last_progress
        now = time.monotonic()
        if now - last_progress < 5 and phase == 'downloading':
            return
        value = {'status': 'running', 'phase': phase, 'path': str(path),
                 'verified_files': len(results), 'verified_bytes': completed,
                 'partial_bytes': partial_bytes, 'total_bytes': manifest['total_bytes'],
                 'elapsed_seconds': now - started, 'updated_unix': time.time()}
        record(root / 'transfer_status.json', value)
        if phase == 'downloading':
            print(json.dumps(value), flush=True)
        last_progress = now

    # Scheduling only: immutable bytes/digests are still checked against the
    # original frozen manifest. Permit PI validation while OW is transferring.
    rows = manifest['files']
    if pi_first:
        rows = sorted(rows, key=lambda row: row['path'].startswith('openwam/'))
    for row in rows:
        if time.time() > deadline:
            raise TimeoutError('Bounded transfer deadline exceeded')
        relative = Path(row['path'])
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unexpected destination symlink')
        origin = 'direct_a100_lan_copy'
        measured = None
        progress(relative, 'checking')
        if path.exists():
            measured = digest(path)
            if path.stat().st_size != row['bytes'] or measured != row['sha256']:
                raise ValueError('Existing destination differs; will not overwrite: ' + str(relative))
            origin = 'verified_resume'
        elif reuse_pi and (candidate := reuse_candidate(relative)) is not None:
            if candidate.stat().st_size == row['bytes'] and digest(candidate) == row['sha256']:
                # Immutable model payload only; never link provenance or markers.
                # The original running model path/content is not modified.
                try:
                    os.link(candidate, path)
                    measured = digest(path)
                    origin = 'verified_existing_a100_export_hardlink'
                except OSError:
                    pass
        if measured is None:
            partial = path.with_name(path.name + '.part')
            if partial.is_symlink():
                raise ValueError('Unexpected partial symlink')
            if row['bytes'] == 0:
                partial.touch(exist_ok=True)
            for attempt in range(3):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > row['bytes']:
                    raise ValueError('Partial file exceeds expected size')
                if offset == row['bytes']:
                    break
                request_headers = dict(headers)
                if offset:
                    request_headers['Range'] = f'bytes={offset}-'
                try:
                    with urlopen(Request(access['url'] + '/' + quote(row['path'], safe='/'),
                                         headers=request_headers), timeout=60) as response, partial.open('ab') as stream:
                        if (offset and response.status != 206) or int(response.headers['Content-Length']) != row['bytes'] - offset:
                            raise ValueError('Export did not honor byte range')
                        while block := response.read(1024 * 1024):
                            if offset + len(block) > row['bytes']:
                                raise ValueError('Export exceeded expected size')
                            stream.write(block)
                            offset += len(block)
                            progress(relative, 'downloading', offset)
                            if time.time() > deadline:
                                raise TimeoutError('Bounded transfer deadline exceeded')
                        stream.flush()
                        os.fsync(stream.fileno())
                    if offset != row['bytes']:
                        raise OSError('Short file response')
                    break
                except (OSError, TimeoutError):
                    if attempt == 2 or time.time() > deadline:
                        raise
                    time.sleep(2)
            progress(relative, 'hashing', row['bytes'])
            measured = digest(partial)
            if partial.stat().st_size != row['bytes'] or measured != row['sha256']:
                raise ValueError('Downloaded checksum mismatch; partial retained: ' + str(relative))
            partial.replace(path)
        if measured != row['sha256']:
            raise ValueError('Reuse checksum mismatch')
        completed += row['bytes']
        results.append({**row, 'actual_sha256': measured, 'verified': True, 'origin': origin})
        record(root / 'transfer_file_verification.json', results)
    result = {'status': 'ok', 'files': len(results), 'bytes': completed,
              'all_sha256_verified': True, 'manifest_sha256': MANIFEST_SHA,
              'direct_source_host': 'kemove-SYS-420GP-TNAR', 'source_lan': '192.168.110.11',
              'target_host': socket.gethostname(), 'target_root': str(root),
              'new_copied_bytes': sum(row['bytes'] for row in results if row['origin'] == 'direct_a100_lan_copy'),
              'reused_verified_bytes': sum(row['bytes'] for row in results if 'hardlink' in row['origin']),
              'models': ['pi05-cup-intersection-30000', 'openwam-cup-intersection-fullpass-5069'],
              'elapsed_seconds': time.monotonic() - started, 'completed_unix': time.time(),
              'model_inference_started': False, 'source_weights_modified': False,
              'target_gpu_validation': 'pending; no readiness claim'}
    record(root / 'transfer_result.json', result)
    record(root / 'transfer_status.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-pi', action='store_true')
    parser.add_argument('--pi-first', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != 'benyun-workstation' or ROOT.is_symlink():
        raise RuntimeError('Not the verified isolated 4090 destination')
    if ROOT.stat().st_uid != os.getuid() or ROOT.stat().st_mode & 0o077:
        raise ValueError('Destination must be owned and private')
    access_path = ROOT / 'transfer_access.json'
    if access_path.stat().st_mode & 0o077:
        raise ValueError('Credentials must be private')
    with (ROOT / '.transfer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            transfer(json.loads(access_path.read_text()), reuse_pi=args.reuse_pi,
                     pi_first=args.pi_first)
        except Exception as error:
            record(ROOT / 'transfer_failure.json', {'status': 'failed', 'type': type(error).__name__,
                   'reason': str(error), 'time': time.time(), 'partial_files_retained': True})
            raise
        finally:
            access_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
