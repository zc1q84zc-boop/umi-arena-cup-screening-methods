"""Read completed telemetry and verify manipulation waited for right homing."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = json.loads((root/'report.json').read_text())
episode = report['episodes'][0]
sequence = report['sequential_homing_intervention']
phases, sources, prompts = Counter(), Counter(), Counter()
first_left = None
for line in (root/'handoff.jsonl').open():
    row = json.loads(line)
    phases[row['phase']] += 1
    sources[row['source']] += 1
    if row['phase'] == 'left_return' and row['source'] == 'model' and first_left is None:
        first_left = row
for line in (root/'online_adapter.jsonl').open():
    row = json.loads(line)
    prompts[row['prompt']] += 1

home_event = next((e for e in sequence.get('phase_events', []) if e['to']=='left_return'), None)
valid_handoff = False
if first_left is not None:
    assert home_event is not None
    m = home_event['metrics']
    assert first_left['right_home_verified']
    assert first_left['physics_time_s'] >= home_event['physics_time_s']
    assert m['right_home_position_error_m'] <= .005
    assert m['right_home_orientation_error_rad'] <= math.radians(2)
    assert m['right_home_max_joint_error_rad'] <= .02
    assert m['right_max_joint_velocity_rad_s'] <= .05
    assert m['right_gripper_open_fraction'] >= .95
    assert m['right_home_stable_s'] >= .5-1e-6
    valid_handoff = True

max_margin = peak_shape = 0.
for line in (root/'left_jaw_margin.jsonl').open():
    row = json.loads(line)
    max_margin = max(max_margin, row['applied_closure_margin_rad'])
    peak_shape = max(peak_shape, row['cup'].get('deformation', {}).get('max_nodal_shape_change_m', 0.)*1000)
summary = {'status': report['status'], 'error': report.get('error'),
    'success': episode['success'], 'full_task_success': episode['full_task_success'],
    'plate_placed': episode['plate_placed'], 'stop_reason': episode['stop_reason'],
    'policy_steps': episode['policy_steps'], 'frames': episode['video_frames'],
    'wrist_frames': episode['wrist_video_frames'], 'max_left_margin_rad': max_margin,
    'peak_shape_change_mm': peak_shape, 'phase_request_counts': dict(phases),
    'action_source_counts': dict(sources), 'prompt_request_counts': dict(prompts),
    'right_home_event': home_event, 'first_left_request': first_left,
    'right_home_before_left_verified': valid_handoff,
    'final_cup_position_m': episode['final_observation']['objects']['cup']['position_m'],
    'final_return_xy_error_mm': 1000*math.dist(
        episode['final_observation']['objects']['cup']['position_m'][:2],
        episode['initial_observation']['objects']['cup']['position_m'][:2])}
(root/'evaluation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
