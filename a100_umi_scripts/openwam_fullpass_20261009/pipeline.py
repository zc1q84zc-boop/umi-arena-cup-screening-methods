#!/usr/bin/env python3
"""Measure, audit, save/read back two updates, then train one complete pass."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/data/benyun/workspace/openwam_cup_fullpass_20261009')
OLD=Path('/mnt/data/benyun/workspace/openwam_charger_full_20260922')
PY=str(OLD/'.venv/bin/python')
PREPARED=Path('/mnt/data/benyun/workspace/openwam_cup_merged_official_20261008/prepared_v2')
RECIPE='/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/openwam/checkpoints/2026-09-25_18-55-32/config.yaml'
GPUS=[1,2]


def write(name,value):
    path=ROOT/name;tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def status(phase,**extra):
    value={'phase':phase,'time':time.time(),'gpus':GPUS,**extra}
    write('pipeline_status.json',value);print(json.dumps(value),flush=True)


def alive(pid):
    path=Path(f'/proc/{pid}/status')
    return path.exists() and '\nState:\tZ' not in path.read_text()


def wait_job(pid,result_path,timeout=3600):
    start=time.monotonic()
    while alive(pid):
        if time.monotonic()-start>timeout:
            raise TimeoutError(f'job {pid}; inspect its log (process left running)')
        time.sleep(5)
    if not result_path.exists():raise RuntimeError(f'job {pid} exited without {result_path}')
    result=json.loads(result_path.read_text())
    if result.get('status') not in ('ok','oom'):raise RuntimeError(result)
    return result


def gpu_guard():
    # Only observe the requested cards; never terminate unrelated jobs.
    for _ in range(30):
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).splitlines()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'],text=True)
        observed=[]
        for row in rows:
            index,uuid,memory,util=[s.strip() for s in row.split(',')]
            if int(index) in GPUS:
                observed.append({'index':int(index),'memory_mib':int(memory),'util':int(util),'has_app':uuid in apps})
        if len(observed)==2 and all(r['memory_mib']<500 and r['util']<=2 and not r['has_app'] for r in observed):return
        time.sleep(1)
    raise RuntimeError('requested cards became occupied; no process was stopped: '+str(observed))


def run(mode,batch,accum,name):
    assert not (ROOT/name).exists(),f'refusing to overwrite {name}'
    gpu_guard();status(mode,name=name,batch=batch,accum=accum)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1,2',CUDA_HOME='/usr/local/cuda-12.8',
        TORCH_EXTENSIONS_DIR=str(ROOT/'torch_extensions'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',DS_BUILD_OPS='0',
        MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='2')
    env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH']
    script=ROOT/'scripts'/('benchmark/run.py' if mode=='benchmark' else 'run.py')
    cmd=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(script),
        '--repo',str(OLD/'OpenWAM'),'--prepared',str(PREPARED),'--recipe',RECIPE,
        '--foundation',str(OLD/'foundation'),'--output',str(ROOT/name),'--mode',mode,
        '--batch',str(batch),'--accum',str(accum)]
    if mode=='benchmark':cmd+=['--steps','8','--warmup','2']
    else:cmd+=['--save-every-updates','1250']
    with (ROOT/(name+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,cwd=ROOT/'scripts',stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,env=env)
        (ROOT/(name+'.pid')).write_text(str(child.pid)+'\n')
        write(name+'_launch.json',{'pid':child.pid,'command':cmd,'time':time.time(),'gpus':GPUS})
        status(mode,name=name,pid=child.pid,batch=batch,accum=accum)
        code=child.wait()
    path=ROOT/name/'result.json'
    if not path.exists():raise RuntimeError(f'{name} exited {code} without result; inspect log')
    result=json.loads(path.read_text())
    if result.get('status') not in ('ok','oom'):raise RuntimeError(result)
    if code and result.get('status')!='oom':raise RuntimeError(f'{name} exit={code}: {result}')
    return result


def readback():
    import torch
    from safetensors import safe_open
    weights=list((ROOT/'smoke/checkpoints').rglob('checkpoint_step_*.safetensors'))
    bases=list((OLD/'foundation').glob('*.safetensors'))
    assert len(weights)==len(bases)==1,(weights,bases)
    checked=changed=0
    with safe_open(str(weights[0]),framework='pt',device='cpu') as saved, safe_open(str(bases[0]),framework='pt',device='cpu') as base:
        assert set(saved.keys())==set(base.keys()),'checkpoint structure differs from foundation'
        keys=[k for k in saved.keys() if k.startswith('action_backbone.') and len(saved.get_slice(k).get_shape())==2]
        assert keys
        for key in keys[:12]:
            tensor=saved.get_tensor(key);before=base.get_tensor(key)
            assert torch.isfinite(tensor).all() and tensor.shape==before.shape
            changed+=int(not torch.equal(tensor,before));checked+=1
    assert changed>0,'no observable action weight update'
    norms=list((ROOT/'smoke/checkpoints').rglob('normalization_stats.npy'))
    assert norms,'normalization stats missing from saved checkpoint'
    result={'status':'ok','checkpoint':str(weights[0]),'tensors_checked':checked,
        'changed_action_tensors':changed,'normalization_stats':list(map(str,norms))}
    write('smoke_readback.json',result)


def main():
    status('waiting_for_b8_benchmark')
    initial=wait_job(int((ROOT/'benchmark_b8_v2.pid').read_text()),ROOT/'benchmark_b8_v2/result.json')
    measured=[]
    if initial['status']=='ok' and initial['peak_reserved_mib']<=.85*81920:measured.append(initial)
    if initial['status']=='ok' and initial['peak_reserved_mib']<.55*81920:
        larger=run('benchmark',16,1,'benchmark_b16')
        if larger['status']=='ok' and larger['peak_reserved_mib']<=.85*81920:measured.append(larger)
    if not measured:
        for batch in (4,2):
            smaller=run('benchmark',batch,1,f'benchmark_b{batch}')
            if smaller['status']=='ok' and smaller['peak_reserved_mib']<=.85*81920:
                measured.append(smaller);break
    assert measured,'no batch with enough memory headroom'
    for value in measured:
        assert value['optimizer_updates']==8 and value['world_size']==2
        assert value['samples_per_second']>0 and math.isfinite(value['samples_per_second'])
    write('benchmarks.json',measured)
    best=max(measured,key=lambda value:value['samples_per_second'])
    batch=best['micro_batch'];accum=32//(2*batch)
    assert batch*2*accum==32
    status('waiting_for_dataset_audit',batch=batch,accum=accum)
    audit=wait_job(int((ROOT/'dataset_audit_v2.pid').read_text()),ROOT/'dataset_audit.json')
    assert audit['status']=='ok'
    smoke=run('smoke',batch,accum,'smoke')
    assert smoke['status']=='ok' and smoke['optimizer_updates']==2 and smoke['consumed_slots']==64
    rows=[json.loads(row) for row in (ROOT/'smoke/steps.jsonl').read_text().splitlines()]
    assert any(value>0 for row in rows for key,value in row['metrics'].items() if 'grad_norm' in key)
    status('checkpoint_readback');readback()
    slots=math.ceil(audit['balanced_entries']/32)*32
    plan={'gpus':GPUS,'world_size':2,'micro_batch_per_gpu':batch,'gradient_accumulation':accum,
        'effective_batch':32,'training_records':audit['training_records'],'validation_records':audit['validation_records'],
        'training_unique_windows':audit['training_unique_windows'],'balanced_entries':audit['balanced_entries'],
        'sample_presentations':slots,'tail_padding_entries':slots-audit['balanced_entries'],
        'optimizer_updates':slots//32,'micro_steps_per_rank':slots//(batch*2),
        'epoch_count':1,'time_cap_hours':None,'sampler':'seed42 complete permutation, tail padding only',
        'measured_samples_per_second':best['samples_per_second'],
        'estimated_compute_hours':slots/best['samples_per_second']/3600,
        'eta_note':'benchmark estimate; model initialization, checkpoints, data/I/O variation add time',
        'checkpoint_every_optimizer_updates':1250,
        'checkpoint_optimizer_updates':list(range(1250,slots//32+1,1250))+([slots//32] if slots//32%1250 else []),
        'foundation':str(OLD/'foundation'),'prepared':str(PREPARED),
        'start_from':'fresh foundation; benchmark/smoke weights discarded',
        'selection':'quality/IK intersection; complete adjacent right-then-left episode pairs merged',
        'official_inputs':'left/right wrists, inter-hand pose7 + finger joints2, current official phase prompt',
        'action':'16D bilateral 10Hz future composed SE3, horizon32, phase-tail masks',
        'model':'video backbone frozen; action/proprio parameters trained, BF16, ZeRO2 CPU optimizer offload',
        'learning_rate':1e-5,'optimizer_snapshots':False,'checkpoint_weights_and_normalization':True,
        'coverage_success_condition':'every balanced entry and every unique physical window consumed, actual rank order verified'}
    write('training_plan.json',plan)
    result=run('train',batch,accum,'training')
    assert result['status']=='ok'
    coverage=json.loads((ROOT/'training/coverage.json').read_text())
    assert coverage['full_pass_completed'] and coverage['all_training_windows_seen']
    status('full_pass_complete',result=result,coverage=coverage)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('stopped_on_error',error=repr(error));raise
