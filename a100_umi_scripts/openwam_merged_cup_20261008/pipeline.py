#!/usr/bin/env python3
"""Gate the authorized pilot on data checks, measured batches, and checkpoint readback."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/data/benyun/workspace/openwam_cup_merged_official_20261008')
OLD=Path('/mnt/data/benyun/workspace/openwam_charger_full_20260922')
PY=str(OLD/'.venv/bin/python')
PREPARED=ROOT/'prepared_v2'


def write(name,value):
    p=ROOT/name
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(p)


def status(phase,**extra):
    value={'phase':phase,'time':time.time(),**extra}
    write('pipeline_status.json',value)
    print(json.dumps(value),flush=True)


def wait_result(path,pid,timeout=1800):
    start=time.monotonic()
    while not path.exists():
        proc=Path(f'/proc/{pid}/status')
        if not proc.exists() or '\nState:\tZ' in proc.read_text():
            raise RuntimeError(f'job {pid} exited without {path}')
        if time.monotonic()-start>timeout: raise TimeoutError(str(path))
        time.sleep(5)
    return json.loads(path.read_text())


def gpu_guard():
    for _ in range(12):
        row=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        memory,util=[int(x.strip()) for x in row.split(',')]
        if memory<128 and util==0:return
        time.sleep(1)
    raise RuntimeError('GPU1 occupied; no other process will be stopped: '+row)


def run(mode,batch,accum,steps,name,save_every=1000):
    gpu_guard()
    status(mode,name=name,batch=batch,accum=accum,micro_steps=steps)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUDA_HOME='/usr/local/cuda-12.8',
             TORCH_EXTENSIONS_DIR=str(ROOT/'torch_extensions'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',DS_BUILD_OPS='0',MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='2')
    env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH']
    cmd=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(ROOT/'scripts/run_v2.py'),
         '--repo',str(OLD/'OpenWAM'),'--prepared',str(PREPARED),
         '--recipe','/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/openwam/checkpoints/2026-09-25_18-55-32/config.yaml',
         '--foundation',str(OLD/'foundation'),'--output',str(ROOT/name),'--mode',mode,
         '--batch',str(batch),'--accum',str(accum),'--steps',str(steps),'--warmup','2','--save-every',str(save_every)]
    with (ROOT/(name+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,cwd=ROOT/'scripts',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        (ROOT/(name+'.pid')).write_text(str(child.pid)+'\n')
        code=child.wait()
    result_path=ROOT/name/'result.json'
    if not result_path.exists(): raise RuntimeError(f'{name} exited {code} without result; see log')
    result=json.loads(result_path.read_text())
    if code and result.get('status')!='oom': raise RuntimeError(f'{name}: {result}')
    return result


def readback(smoke):
    import torch
    from safetensors import safe_open
    weights=list((ROOT/smoke/'checkpoints').rglob('checkpoint_step_*.safetensors'))
    assert len(weights)==1,'expected one completed smoke checkpoint'
    bases=list((OLD/'foundation').glob('*.safetensors'))
    assert len(bases)==1
    checked,changed=0,0
    with safe_open(str(weights[0]),framework='pt',device='cpu') as saved, safe_open(str(bases[0]),framework='pt',device='cpu') as base:
        assert set(saved.keys())==set(base.keys()),'saved model structure differs from foundation'
        keys=[k for k in saved.keys() if k.startswith('action_backbone.') and len(saved.get_slice(k).get_shape())==2]
        assert keys,'no action parameters in checkpoint'
        for key in keys[:12]:
            tensor=saved.get_tensor(key)
            assert torch.isfinite(tensor).all(),f'nonfinite saved tensor {key}'
            assert tensor.shape==base.get_tensor(key).shape
            changed+=int(not torch.equal(tensor,base.get_tensor(key)))
            checked+=1
    assert changed>0,'smoke produced no observable action weight update'
    result={'status':'ok','checkpoint':str(weights[0]),'tensors_checked':checked,'changed_action_tensors':changed}
    write('smoke_readback.json',result)
    return result


def main():
    status('waiting_for_existing_benchmark')
    initial=wait_result(ROOT/'benchmark_b2/result.json',int((ROOT/'benchmark_b2.pid').read_text()))
    if initial.get('status')!='ok': raise RuntimeError('initial real-gradient benchmark failed: '+str(initial))
    assert initial['optimizer_updates']>0
    status('checking_final_pairing_dataset')
    assert (PREPARED/'summary.json').exists(),'final pairing preparation incomplete'
    subprocess.run([PY,str(ROOT/'scripts/check_data.py'),'--prepared',str(PREPARED)],check=True,cwd=ROOT/'scripts')
    checks=json.loads((PREPARED/'data_checks.json').read_text())
    # Measure final data, including actual two-view video decoding. A smaller
    # draft pairing set was used only for the first batch/cardinality probe.
    results=[]
    for batch in ([4,8] if initial['peak_reserved_mib']<.60*81920 else [2]):
        value=run('benchmark',batch,1,8,f'final_benchmark_b{batch}')
        if value['status']=='oom':break
        if value['peak_reserved_mib']<=.80*81920:results.append(value)
        if value['peak_reserved_mib']>.65*81920:break
    if not results:
        value=run('benchmark',2,1,8,'final_benchmark_fallback_b2')
        assert value['status']=='ok' and value['peak_reserved_mib']<.85*81920
        results.append(value)
    best=max(results,key=lambda r:r['samples_per_second'])
    batch=best['micro_batch'];accum=32//batch
    assert batch*accum==32
    smoke=run('smoke',batch,accum,2*accum,'smoke_final')
    assert smoke['status']=='ok' and smoke['optimizer_updates']==2
    rows=[json.loads(x) for x in (ROOT/'smoke_final/steps.jsonl').read_text().splitlines()]
    assert any(v>0 for r in rows for k,v in r['metrics'].items() if 'grad_norm' in k),'no positive gradient audit'
    status('checkpoint_readback')
    readback('smoke_final')
    # Initial pilot: at most one balanced pass, with a conservative 24-GPU-hour
    # planning cap. Leave 15% for initialization, checkpoint I/O and variance.
    samples_cap=min(checks['training_balanced_windows'],int(best['samples_per_second']*24*3600*.85))
    steps=max(2*accum,(samples_cap//32)*accum)
    save_every=max(accum,(steps//4//accum)*accum)
    plan={'micro_batch':batch,'gradient_accumulation':accum,'world_size':1,'effective_batch':32,
          'measured_samples_per_second':best['samples_per_second'],'max_micro_steps':steps,
          'optimizer_updates':steps//accum,'sample_presentations':steps*batch,
          'training_unique_windows':checks['training_unique_windows'],
          'training_balanced_windows':checks['training_balanced_windows'],
          'expected_window_exposure_passes':steps*batch/checks['training_unique_windows'],
          'planning_gpu_hours':24,'planning_time_reserve_fraction':.15,
          'checkpoint_every_micro_steps':save_every,'prepared':str(PREPARED),
          'selection':'intersection only; full adjacent right-then-left pairs; existing quality speed limits at merge boundary',
          'independent_validation_split':True,'validation_metric_note':'held-out replay errors are diagnostics; closed-loop task success still required',
          'official_input_checked':True,'official_5070ti_16gb_runtime_verified':False,
          'optimizer_snapshots':False,'start_from':'foundation; benchmark and smoke updates discarded'}
    write('training_plan.json',plan)
    result=run('train',batch,accum,steps,'training',save_every)
    status('pilot_complete' if result['status']=='ok' else 'pilot_failed',result=result)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('stopped_on_error',error=repr(error))
        raise
