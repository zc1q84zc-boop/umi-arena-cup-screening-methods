"""Copy the last official shard independently while the main copy continues."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

config = json.loads(Path(__file__).with_name('transfer_config.json').read_text())
name = 'model-00006-of-00006.safetensors'
expected = config['manifest']['files'][name]
target = Path(config['target']) / name
temporary = target.with_name(name + '.tail.partial')
request = urllib.request.Request(config['source'] + '/' + name,
                                 headers={'Authorization': 'Bearer ' + config['token']})
start = time.monotonic()
digest = hashlib.sha256()
total = 0
with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as f:
    while block := response.read(8 * 1024 * 1024):
        f.write(block)
        digest.update(block)
        total += len(block)
if total != expected['size'] or digest.hexdigest() != expected['sha256']:
    raise RuntimeError('Last official shard size/SHA-256 mismatch')
os.replace(temporary, target)
print('TAIL_FILE_VERIFIED', name, total, 'elapsed_s', time.monotonic() - start, flush=True)
