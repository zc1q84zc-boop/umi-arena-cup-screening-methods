from pathlib import Path
import subprocess,json,time,signal,os
running=True

def stop(*args):
    global running
    running=False
signal.signal(signal.SIGTERM,stop)
p=Path(__file__).parent/os.environ.get('UMI_GPU_SAMPLE_FILE','gpu_samples.jsonl')
with p.open('a',buffering=1) as f:
    while running:
        r=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
        f.write(json.dumps({'utc':time.time(),'gpus':r.stdout.strip().splitlines(),'loadavg':Path('/proc/loadavg').read_text().split()[:3]})+'\n')
        time.sleep(1)
