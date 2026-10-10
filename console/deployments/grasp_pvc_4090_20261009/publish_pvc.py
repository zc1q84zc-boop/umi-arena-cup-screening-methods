"""Publish terminal FEM comparison videos only; no online policy API calls."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket

ROOT = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source',type=Path);parser.add_argument('run_id')
    args = parser.parse_args()
    assert socket.gethostname() == 'benyun-workstation'
    assert re.fullmatch(r'[0-9a-f]{12}',args.run_id)
    assert args.source.parent == Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/runs')
    assert args.source.name == 'grasp_pvc_'+args.run_id
    report = json.loads((args.source/'report.json').read_text())
    assert report['status'] in ('completed','failed','stopped') and report['model_inference_requests'] == 0
    diagnostic = report['frozen_pvc_diagnostic']
    assert diagnostic['original_world_pose_and_jaw_commands_unchanged']
    profile = report['cup_physics_profile']
    target = ROOT/'sim_runs'/args.run_id
    if target.exists():
        assert json.loads((target/'metadata.json').read_text())['diagnostic_source_dir'] == str(args.source)
        assert hashlib.sha256((target/'report.json').read_bytes()).digest() == hashlib.sha256((args.source/'report.json').read_bytes()).digest()
        print('already published',args.run_id);return
    target.mkdir()
    for name in ('report.json','manifest.json','video.mp4','video_left_wrist.mp4','video_right_wrist.mp4',
                 'joints.csv','cup_deformation.jsonl','frozen_command_audit.jsonl','continuous_targets.jsonl',
                 'wrist_camera_poses.jsonl','safety_abort.json','final_nodes.npz'):
        if (args.source/name).exists():shutil.copy2(args.source/name,target/name)
    metadata = dict(id=args.run_id,status=report['status'],
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        policy='frozen-c3b23931ada5-pvc-material',
        policy_label=f'冻结动作物理对照 · PVC 弹性杯 {profile["youngs_modulus_Pa"]/1e9:g} GPa（非在线推理）',
        seed=42,setup_index=0,steps=360,run_until_success=False,task_objective='plate_return',
        camera='head',gpu=0,inference_backend=None,contact_profile=profile['id'],
        left_return_diagnostic=False,left_extra_closure_fraction=0.,
        evaluation_class=diagnostic['classification'],recorded_views=['head','left_wrist','right_wrist'],
        simulator_profile='tuned_online_v1',source_run='c3b23931ada5',diagnostic_source_dir=str(args.source),
        model_inference_requests=0,frozen_pvc_diagnostic=diagnostic,inference_cleanup='not_started',
        description='未实测 PVC 类弹性壳，1 mm / 128 次迭代 / 240 Hz；仅改材料硬度的冻结动作对照，非在线推理。',
        error=report.get('error'))
    with (target/'metadata.json').open('x') as stream:json.dump(metadata,stream,indent=2)
    print('published',args.run_id,metadata['policy_label'])


if __name__ == '__main__':main()
