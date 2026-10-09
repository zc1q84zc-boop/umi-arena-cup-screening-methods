"""Import completed diagnostics only; never create or start an online policy."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import socket

ROOT=Path('/home/claude/umi-track1-console-4090-20261009')
PAIR=Path('/home/claude/dual-franka-yubi-isaac-sim-deploy/runs/grasp_margin_pair_b71a604f93c2')


def main():
    assert socket.gethostname()=='benyun-workstation'
    for mode,run_id in [('baseline','b71a604f93c3'),('extra02','b71a604f93c4')]:
        source=PAIR/mode
        if not (source/'report.json').exists():continue
        report=json.loads((source/'report.json').read_text())
        if report['status'] not in ('completed','stopped','failed'):continue
        target=ROOT/'sim_runs'/run_id
        if target.exists():
            assert json.loads((target/'metadata.json').read_text())['diagnostic_source_dir']==str(source)
            assert hashlib.sha256((target/'report.json').read_bytes()).digest()==hashlib.sha256((source/'report.json').read_bytes()).digest()
            print('already published',run_id);continue
        target.mkdir()
        files=['report.json','manifest.json','video.mp4','video_left_wrist.mp4','video_right_wrist.mp4',
               'joints.csv','gripper_contact_audit.jsonl','frozen_command_audit.jsonl','continuous_targets.jsonl',
               'wrist_camera_poses.jsonl','safety_abort.json']
        for name in files:
            if (source/name).exists():shutil.copy2(source/name,target/name)
        metadata={
            'id':run_id,'status':report['status'],'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'policy':'frozen-c3b23931ada5-grasp-margin',
            'policy_label':'冻结动作物理对照 · '+('原动作基线' if mode=='baseline' else '右夹爪额外闭合 2%')+'（非在线推理）',
            'seed':42,'setup_index':0,'steps':360,'run_until_success':False,
            'task_objective':'plate_return','camera':'head','gpu':0,'inference_backend':None,
            'contact_profile':'baseline','left_return_diagnostic':False,'left_extra_closure_fraction':0.,
            'evaluation_class':'frozen_model_action_physics_comparison_not_online_inference',
            'recorded_views':['head','left_wrist','right_wrist'],'simulator_profile':'tuned_online_v1',
            'source_run':'c3b23931ada5','diagnostic_source_dir':str(source),
            'model_inference_requests':0,'right_extra_closure_fraction':report['grasp_margin_diagnostic']['max_right_extra_closure_fraction'],
            'grasp_margin_diagnostic':report['grasp_margin_diagnostic'],'inference_cleanup':'not_started',
            'description':'固定已保存动作、同一杯子材质与控制器；不使用杯位辅助、不调用模型。对照不等于纯模型成功。',
            'error':report.get('error')}
        with (target/'metadata.json').open('x') as stream:json.dump(metadata,stream,indent=2)
        print('published',run_id,mode)


if __name__=='__main__':main()
