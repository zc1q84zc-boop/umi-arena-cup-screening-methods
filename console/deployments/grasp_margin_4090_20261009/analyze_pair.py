"""Offline evidence for the isolated jaw-only comparison, not model success."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from margin_core import filtered_waypoints

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'migration_4090_20261009'))
from verify_intersection_grasp import evaluate


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(path, original=None):
    report = json.loads((path/'report.json').read_text())
    contacts = read_jsonl(path/'gripper_contact_audit.jsonl')
    result = evaluate(contacts)
    endpoints = {r['action_index']: r for r in contacts if r['request_step'] >= 0}
    commands = read_jsonl(path/'frozen_command_audit.jsonl')
    extra = report['grasp_margin_diagnostic']['max_right_extra_closure_fraction']
    assert [c['step'] for c in commands] == list(range(len(commands)))
    for c in commands:
        assert c['applied_waypoints'] == filtered_waypoints(c['source_waypoints'], extra)
        if original is not None:
            assert c['source_waypoints'] == original[c['step']]['substep_waypoints']
    result.update(report_status=report['status'], requests=len(commands),
        maximum_extra_closure_fraction=extra, numeric_command_preservation_verified=True,
        model_inference_requests=0, arm_targets_changed=False, physics_materials_changed=False)
    maximum_streak = streak = 0
    friction_peak = {p: 0. for p in ('left_finger', 'right_finger')}
    samples = []
    previous = None
    for action, r in sorted(endpoints.items()):
        w,x,y,z = r['cup_quaternion_wxyz'];cosine=1-2*(x*x+y*y);sine=math.sqrt(max(0,1-cosine*cosine))
        p=r['cup_position_m']
        clearance=min(p[2]-.027*sine,p[2]+.075*cosine-.04*sine)-.75
        state=r['robots']['right'];force=state['cup_contacts']
        both=all(force[k]['positive_normal_force_sum_N']>.02 for k in friction_peak)
        contiguous=previous is not None and action==previous+1
        streak=streak+1 if contiguous and both and clearance>=.005 else int(both and clearance>=.005)
        maximum_streak=max(maximum_streak,streak)
        for part in friction_peak:
            friction_peak[part]=max(friction_peak[part],force[part]['friction']['net_tangential_force_norm_N'])
        if 870 <= action <= 930:
            samples.append({'action_index':action, 'time_s':r['physics_time_s'],
                'clearance_mm':clearance*1000, 'cup_origin_z_m':p[2],
                'cup_tilt_deg':math.degrees(math.acos(max(-1,min(1,cosine)))),
                'jaw_requested_rad':state['jaw_requested_rad'], 'jaw_actual_rad':state['jaw_actual_rad'],
                'normal_force_N':{k:force[k]['positive_normal_force_sum_N'] for k in friction_peak},
                'net_tangential_force_world_N':{k:force[k]['friction']['net_tangential_force_world_N'] for k in friction_peak}})
        previous=action
    result['right_bilateral_contact_above_5mm_longest_actions']=maximum_streak
    result['right_bilateral_contact_above_5mm_longest_s']=maximum_streak/30
    result['this_duration_is_not_a_stable_grasp_criterion']=True
    result['right_peak_net_tangential_force_norm_N']=friction_peak
    result['friction_utilization_ratio_computed']=False
    result['friction_anchor_normal_contact_one_to_one_mapping_available']=False
    result['drop_window']=samples
    abort=path/'safety_abort.json'
    result['safety_abort']=json.loads(abort.read_text()) if abort.exists() else None
    result['input_sha256']=report.get('input_sha256',{})
    result['raw_sha256']={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in
                         ('gripper_contact_audit.jsonl','frozen_command_audit.jsonl','continuous_targets.jsonl')}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('pair',type=Path)
    parser.add_argument('--original',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    original=read_jsonl(args.original) if args.original else None
    results={name:analyze(args.pair/name,original) for name in ('baseline','extra02') if (args.pair/name/'report.json').exists()}
    if len(results)==2:
        a,b=results['baseline'],results['extra02']
        common={p:h for p,h in a['input_sha256'].items() if p in b['input_sha256']}
        assert all(b['input_sha256'][p]==h for p,h in common.items()),'Common controller/scene code changed between variants'
    output={'classification':'frozen_action_jaw_bias_comparison_not_online_model_success',
            'source_run':'c3b23931ada5','tested_source_prefix_requests':360,
            'tested_source_prefix_servo_actions':1080,'real_calibration_measured':False,
            'results':results,'default_controller_or_usd_modified':False}
    if args.output:
        with args.output.open('x') as f:json.dump(output,f,indent=2)
    compact={name:{k:v for k,v in r.items() if k not in ('drop_window','input_sha256','raw_sha256','safety_abort')}
             for name,r in results.items()}
    print(json.dumps(compact,indent=2))


if __name__=='__main__':main()
