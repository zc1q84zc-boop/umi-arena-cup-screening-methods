"""Download only the authorized export manifest; hash before atomic publish."""
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request,urlopen

ROOT=Path('/home/lrl/workspace/umi_cup_intersection_models_20261009')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):h.update(block)
    return h.hexdigest()


def main():
    access=json.loads((ROOT/'transfer_access.json').read_text())
    headers={'Authorization':'Bearer '+access['token']}
    with urlopen(Request(access['url']+'/manifest.json',headers=headers),timeout=20) as response:
        manifest=json.load(response)
    (ROOT/'transfer_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.monotonic();completed=0;last=0
    for row in manifest['files']:
        relative=Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
            completed+=row['bytes'];continue
        partial=path.with_name(path.name+'.part')
        if row['bytes']==0:partial.touch(exist_ok=True)
        for attempt in range(3):
            offset=partial.stat().st_size if partial.exists() else 0
            assert offset<=row['bytes']
            if offset==row['bytes']:break
            request_headers=dict(headers)
            if offset:request_headers['Range']=f'bytes={offset}-'
            try:
                with urlopen(Request(access['url']+'/'+row['path'],headers=request_headers),timeout=60) as response, partial.open('ab') as stream:
                    if offset:assert response.status==206
                    while block:=response.read(8*1024*1024):
                        stream.write(block);offset+=len(block)
                        now=time.monotonic()
                        if now-last>5:
                            print(json.dumps({'path':row['path'],'file_bytes':offset,
                                'completed_GiB':(completed+offset)/2**30,'total_GiB':manifest['total_bytes']/2**30,
                                'seconds':now-started}),flush=True);last=now
                    stream.flush();os.fsync(stream.fileno())
                break
            except (OSError,TimeoutError):
                if attempt==2:raise
                time.sleep(2)
        assert partial.stat().st_size==row['bytes'] and sha(partial)==row['sha256'],row['path']
        partial.replace(path);completed+=row['bytes']
    result={'status':'ok','files':len(manifest['files']),'bytes':completed,'seconds':time.monotonic()-started,
        'all_sha256_verified':True,'models':['openwam-cup-intersection-fullpass-5069','pi05-cup-intersection-30000']}
    (ROOT/'transfer_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
