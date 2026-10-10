import hashlib
import json
from pathlib import Path

source = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/runs/console_612620c03d8a')
output = Path(__file__).resolve().parent/'frozen_targets.json'
report = json.loads((source/'report.json').read_text())
rows = []
with (source/'online_adapter.jsonl').open() as stream:
    for line in stream:
        row = json.loads(line)
        if row['step'] >= 1600:
            break
        assert row['step'] == len(rows)
        assert row['substep_waypoints'] == [row['waypoint']]*3
        rows.append({k:row[k] for k in ('step','waypoint','prompt','task_stage')})
assert len(rows) == 1600
package = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1/yubi_isaac_sim_env')
files = ['continuous_targets.py','env.py','policy_adapter.py','config.json','pvc_shell.py','pvc_numerics.py',
         'scenes/dual_franka_yubi_official_fingertip_friction_trial.usda','head_camera_online_aligned_v2.json',
         'wrist_camera_visual_aligned_v3.json','shared_camera_render.py']
data = dict(source_run='612620c03d8a',source_sha256=hashlib.sha256((source/'online_adapter.jsonl').read_bytes()).hexdigest(),
            initial_observation=report['episodes'][0]['initial_observation'],
            source_status=report['status'],cup_physics_profile=report['cup_physics_profile'],
            expected_sha256={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in files},
            controller=report['trajectory_controller_profile'],rows=rows)
output.write_text(json.dumps(data))
print(json.dumps(dict(output=str(output),requests=len(rows),bytes=output.stat().st_size,
                     source_sha256=data['source_sha256'])))
