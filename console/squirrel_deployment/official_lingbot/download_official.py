"""Copy the pinned public model and verify every file before marking ready."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import urllib.request

config = json.loads(Path(__file__).with_name('transfer_config.json').read_text())
manifest = config['manifest']
target = Path(config['target'])
target.mkdir(parents=True, exist_ok=True)
start = time.monotonic()
def copy_one(item):
    name, expected = item
    path = target / name
    if path.is_file() and path.stat().st_size == expected['size']:
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual == expected['sha256']:
            print('VERIFIED_EXISTING', name, flush=True)
            return
    temporary = target / (name + '.partial')
    total = temporary.stat().st_size if temporary.is_file() else 0
    if total > expected['size']:
        raise RuntimeError('Partial official file exceeds expected size: ' + name)
    headers = {'Authorization': 'Bearer ' + config['token']}
    if total:
        headers['Range'] = f'bytes={total}-'
        print('RESUMING', name, total, flush=True)
    request = urllib.request.Request(config['source'] + '/' + name, headers=headers)
    last = total
    # A completed partial file only needs verification. A resumed HTTP copy
    # must explicitly confirm the requested range before appending any bytes.
    if total < expected['size']:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('ab') as f:
            if total and (response.status != 206 or response.headers.get('Content-Range') !=
                          f"bytes {total}-{expected['size']-1}/{expected['size']}"):
                raise RuntimeError('Source did not confirm requested byte range')
            while block := response.read(512 * 1024):
                f.write(block)
                total += len(block)
                if total - last >= 1024**3:
                    print('COPY_PROGRESS', name, total, 'bytes', flush=True)
                    last = total
    with temporary.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if total != expected['size'] or actual != expected['sha256']:
        raise RuntimeError('Official file size/SHA-256 mismatch: ' + name)
    os.replace(temporary, path)
    print('VERIFIED', name, total, flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(copy_one, manifest['files'].items()))
manifest['verified'] = True
manifest['verified_at_unix'] = time.time()
manifest['transfer_elapsed_s'] = time.monotonic() - start
(target / 'official_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('OFFICIAL_CHECKPOINT_VERIFIED', manifest['revision'], flush=True)
