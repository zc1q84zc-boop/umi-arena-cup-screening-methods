#!/usr/bin/env python3
"""Scene-check console for the four new tasks; uses the existing GPU0 lock."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

CONSOLE=Path(__file__).resolve().parents[2]
PROFILE=CONSOLE/'simulator_profiles/arena_tasks_v1'
CONFIGS=PROFILE/'yubi_isaac_sim_env/arena_tasks/configs'
RUNTIME=Path(os.environ.get('UMI_ISAAC_RUNTIME','/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy'))
RUNS=CONSOLE/'arena_runs'
IDS=('pens','sps','cable','phone')
RUN_ID=re.compile(r'^[0-9a-f]{12}$')
lock=threading.Lock();active=None


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def catalog():
    result=[]
    for k in IDS:
        task=json.loads((CONFIGS/f'{k}.json').read_text())
        result.append(dict(id=k,label=task['label'],steps=[p['prompt'] for p in task['primitives']],count=len(task['primitives'])))
    return result


def execute(metadata):
    global active
    out=RUNS/metadata['id'];output=Path(metadata['record_dir'])
    try:
        metadata['status']='running';save(out/'metadata.json',metadata)
        with (out/'run.log').open('w') as log:
            result=subprocess.run(['bash',str(PROFILE/'run_arena.sh'),metadata['task'],str(output),str(metadata['seed']),'0',str(metadata['steps'])],stdout=log,stderr=subprocess.STDOUT)
        metadata['status']='completed' if result.returncode==0 else 'failed'
        metadata['exit_code']=result.returncode
        if (output/'report.json').is_file():metadata['report']=json.loads((output/'report.json').read_text())
        else:metadata['error']='场景未启动；请查看运行日志和 GPU 占用。'
    except Exception as exc:metadata.update(status='failed',error=str(exc))
    finally:
        save(out/'metadata.json',metadata)
        with lock:active=None


def start(data):
    global active
    if set(data)!={'task','seed','steps'} or data['task'] not in IDS:raise ValueError('Invalid task request')
    for key,upper in [('seed',2147483647),('steps',300)]:
        if type(data[key]) is not int or not 0<=data[key]<=upper:raise ValueError('Invalid numeric field')
    if data['steps']<1:raise ValueError('Steps must be positive')
    with lock:
        if active:raise ValueError('已有场景检查正在运行')
        rid=uuid.uuid4().hex[:12];out=RUNS/rid;out.mkdir(parents=True,exist_ok=False)
        metadata=dict(id=rid,task=data['task'],seed=data['seed'],steps=data['steps'],status='starting',
                      record_dir=str(RUNTIME/'runs'/f'arena_console_{rid}'))
        save(out/'metadata.json',metadata);active=rid
        threading.Thread(target=execute,args=(metadata,),daemon=True).start()
    return metadata


class Handler(BaseHTTPRequestHandler):
    def send(self,value,status=200):
        data=json.dumps(value,ensure_ascii=False).encode();self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

    def file(self,path,kind):
        if not path.is_file():self.send({'error':'尚无预览'},404);return
        data=path.read_bytes();self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

    def do_GET(self):
        path=self.path.split('?',1)[0]
        if path=='/':return self.file(Path(__file__).with_name('arena.html'),'text/html; charset=utf-8')
        if path=='/api/catalog':return self.send(dict(tasks=catalog(),active=active))
        if path.startswith('/api/runs/'):
            rid=path.rsplit('/',1)[-1]
            if RUN_ID.fullmatch(rid) and (RUNS/rid/'metadata.json').is_file():return self.send(json.loads((RUNS/rid/'metadata.json').read_text()))
        if path.startswith('/preview/'):
            key=path.rsplit('/',1)[-1]
            markers=Path(__file__).with_name('validation.json')
            records=json.loads(markers.read_text()).get('tasks',{}) if markers.is_file() else {}
            if key in records:return self.file(Path(records[key]['record_dir'])/'scene_preview.jpg','image/jpeg')
        match=re.fullmatch('/runs/([0-9a-f]{12})/(preview.jpg|head.mp4|left_wrist.mp4|right_wrist.mp4|report.json|run.log)',path)
        if match:
            rid,name=match.groups();meta=RUNS/rid/'metadata.json'
            if not meta.is_file():return self.send({'error':'Run not found'},404)
            value=json.loads(meta.read_text());root=RUNS/rid if name=='run.log' else Path(value['record_dir'])
            kinds={'.jpg':'image/jpeg','.mp4':'video/mp4','.json':'application/json','.log':'text/plain; charset=utf-8'}
            return self.file(root/name,kinds[Path(name).suffix])
        self.send({'error':'Not found'},404)

    def do_POST(self):
        try:
            if self.path!='/api/runs':raise ValueError('Unknown action')
            origin=self.headers.get('Origin')
            if origin and origin.split('://',1)[-1]!=self.headers.get('Host'):raise ValueError('Invalid request origin')
            if not self.headers.get('Content-Type','').startswith('application/json'):raise ValueError('JSON required')
            length=int(self.headers.get('Content-Length',0))
            if not 1<=length<=2048:raise ValueError('Invalid request size')
            data=json.loads(self.rfile.read(length));self.send(start(data),202)
        except (ValueError,KeyError,TypeError) as exc:self.send({'error':str(exc)},400)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8775);args=parser.parse_args()
    RUNS.mkdir(parents=True,exist_ok=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
