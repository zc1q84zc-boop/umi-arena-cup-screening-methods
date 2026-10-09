"""Temporary manifest-only LAN export, with bearer auth and resumable reads."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import unquote,urlsplit

ROOT=Path('/mnt/data/benyun/workspace/umi_cup_intersection_export_20261009')
OW=Path('/mnt/data/benyun/workspace/openwam_cup_fullpass_20261009')
PI=Path('/mnt/data/benyun/workspace/pi05_cup_intersection_20261008')
REPO=Path('/mnt/data/benyun/workspace/openwam_charger_full_20260922/OpenWAM')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):h.update(block)
    return h.hexdigest()


def prepare():
    assert json.loads((OW/'training/coverage.json').read_text())['full_pass_completed']
    assert json.loads((PI/'status.json').read_text())['completed_steps']==30000
    sources={}
    def add_tree(src,dst,filter_code=False):
        for path in sorted(src.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                if filter_code and path.suffix not in ('.py','.yaml','.yml','.json','.txt','.cpp','.cu','.h','.hpp'):continue
                sources[str(Path(dst)/path.relative_to(src))]=path
    ckpt=OW/'training/checkpoints/2026-10-09_04-44-22'
    for name in ('checkpoint_step_5069.safetensors','config.yaml','normalization_stats.npy'):
        sources['openwam/5069/'+name]=ckpt/name
    add_tree(ckpt/'tokenizer','openwam/5069/tokenizer')
    add_tree(OW/'training/official_contract','openwam/5069/official_contract')
    for name in ('coverage.json','result.json'):
        sources['openwam/5069/provenance/'+name]=OW/'training'/name
    sources['openwam/5069/provenance/training_plan.json']=OW/'training_plan.json'
    pi_ckpt=PI/'checkpoints/pi05_cup_intersection/pi05_cup_intersection_v2/30000'
    for name in ('params','assets'):add_tree(pi_ckpt/name,'pi05/30000/'+name)
    sources['pi05/30000/_CHECKPOINT_METADATA']=pi_ckpt/'_CHECKPOINT_METADATA'
    for name in ('status.json','provenance.json','validation.jsonl'):
        sources['pi05/30000/provenance/'+name]=PI/name
    marker={'model':'pi05-cup-intersection-30000','source_step':30000,'export_type':'params_and_assets_only',
        'asset_id':'intersection_v2','action_hz':10,'future_action_row':0,'paired':False}
    (ROOT/'inference_export.json').write_text(json.dumps(marker,indent=2)+'\n')
    sources['pi05/30000/inference_export.json']=ROOT/'inference_export.json'
    for name in ('openwam','configs','third_party'):add_tree(REPO/name,'source/OpenWAM/'+name,filter_code=True)
    add_tree(PI/'evaluation/umi_arena','source/evaluation/umi_arena',filter_code=True)
    files=[]
    for name,path in sources.items():
        files.append({'path':name,'source':str(path),'bytes':path.stat().st_size,'sha256':digest(path)})
    manifest={'created_unix':time.time(),'total_bytes':sum(row['bytes'] for row in files),
        'files':[{k:v for k,v in row.items() if k!='source'} for row in files]}
    (ROOT/'source_files.json').write_text(json.dumps(files,indent=2)+'\n')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return files,manifest


def main():
    ROOT.mkdir(mode=0o700,exist_ok=False)
    files,manifest=prepare()
    mapping={row['path']:Path(row['source']) for row in files}
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.headers.get('Authorization')!='Bearer '+token:
                self.send_error(403);return
            name=unquote(urlsplit(self.path).path).lstrip('/')
            if name=='manifest.json':path=ROOT/'manifest.json'
            elif name in mapping:path=mapping[name]
            else:self.send_error(404);return
            size=path.stat().st_size;offset=0
            range_header=self.headers.get('Range')
            if range_header:
                if not range_header.startswith('bytes=') or not range_header.endswith('-'):
                    self.send_error(416);return
                offset=int(range_header[6:-1])
                if not 0<=offset<size:self.send_error(416);return
            self.send_response(206 if offset else 200)
            self.send_header('Content-Length',str(size-offset))
            if offset:self.send_header('Content-Range',f'bytes {offset}-{size-1}/{size}')
            self.end_headers()
            try:
                with path.open('rb') as stream:
                    stream.seek(offset)
                    while block:=stream.read(8*1024*1024):self.wfile.write(block)
            except (BrokenPipeError,ConnectionResetError):pass
    server=ThreadingHTTPServer(('192.168.110.11',0),Handler)
    access={'url':f'http://192.168.110.11:{server.server_port}','token':token,
        'pid':os.getpid(),'file_count':len(files),'total_bytes':manifest['total_bytes']}
    path=ROOT/'access.json';path.write_text(json.dumps(access));path.chmod(0o600)
    print(json.dumps({k:v for k,v in access.items() if k not in ('token','url')}),flush=True)
    # Export access expires; explicit cleanup also follows successful migration.
    threading.Timer(7200,server.shutdown).start()
    server.serve_forever()


if __name__=='__main__':main()
