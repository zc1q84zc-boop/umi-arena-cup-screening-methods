import json
from pathlib import Path
import subprocess

ROOT = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/runs/jaw_margin005_sequential_home_20261010_v2')


def latest(name):
    p = ROOT / name
    if not p.exists():
        return None
    if p.suffix != '.jsonl':
        return json.loads(p.read_text())
    with p.open('rb') as stream:
        stream.seek(max(0, p.stat().st_size - 131072))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass


service = subprocess.check_output(['systemctl','--user','show',
    'umi-sequential-home005-v2-20261010.service','-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],text=True)
evaluation = latest('evaluation.jsonl')
progress = latest('progress.json')
row = {'service':service.strip(),'progress':progress}
if evaluation:
    keys = ('policy_step','physics_time_s','requested_task_stage','task_stage','plate_placed',
            'full_task_success','cup_position_m','gripper_open_fraction','jaw_joint_position_rad')
    row['evaluation'] = {key:evaluation.get(key) for key in keys}
if (ROOT/'report.json').exists():
    report = latest('report.json')
    row['report'] = {'status':report['status'],'error':report.get('error'),
        'episodes':[{k:e.get(k) for k in ('success','full_task_success','policy_steps','stop_reason')}
            for e in report.get('episodes',[])],
        'sequencing':report.get('sequential_homing_intervention')}
print(json.dumps(row))
