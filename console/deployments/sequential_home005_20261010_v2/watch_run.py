"""Emit compact progress every 45 seconds until the owned trial exits."""
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1].parent / 'dual-franka-yubi-isaac-sim-deploy/runs/jaw_margin005_sequential_home_20261010_v2'
while True:
    service = subprocess.check_output(['systemctl', '--user', 'show',
        'umi-sequential-home005-v2-20261010.service', '-p', 'ActiveState', '--value'], text=True).strip()
    row = {'service': service}
    if (ROOT/'progress.json').exists():
        p = json.loads((ROOT/'progress.json').read_text())
        row.update({k: p.get(k) for k in ('physics_time_s', 'wall_s', 'servo_actions', 'model_requests',
            'phase', 'plate_placed', 'right_home_verified', 'handoff_time_s',
            'first_left_policy_step', 'right_home_position_error_m', 'right_home_orientation_error_rad',
            'right_home_max_joint_error_rad', 'right_max_joint_velocity_rad_s', 'right_home_stable_s',
            'cup_position_m', 'cup_quaternion_wxyz', 'max_shape_change_mm')})
    print(json.dumps(row), flush=True)
    if service != 'active':
        break
    time.sleep(45)
