"""Frozen c3b23931ada5 commands with an audited, optional right-jaw bias."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from margin_core import filtered_waypoints, load_source

ROWS = load_source(os.environ['UMI_MARGIN_SOURCE'], os.environ['UMI_MARGIN_SOURCE_SHA256'])
EXTRA = float(os.environ['UMI_MARGIN_EXTRA_FRACTION'])
OUT = Path(os.environ['SIM_ADAPTER_AUDIT_DIR'])
NEXT_STEP = 0


def predict(observation, step, episode):
    global NEXT_STEP
    if episode != 0 or step != NEXT_STEP or not 0 <= step < len(ROWS):
        raise ValueError('Nonsequential frozen replay request')
    original = ROWS[step]['substep_waypoints']
    applied = filtered_waypoints(original, EXTRA)
    with (OUT/'frozen_command_audit.jsonl').open('a') as stream:
        stream.write(json.dumps({'episode': episode, 'step': step,
            'source_run': 'c3b23931ada5', 'source_sha256': os.environ['UMI_MARGIN_SOURCE_SHA256'],
            'source_waypoints': original, 'applied_waypoints': applied,
            'model_inference_called': False, 'object_feedback_used': False,
            'arm_world_targets_unchanged': True, 'left_jaw_unchanged': True,
            'extra_closure_fraction': EXTRA, 'action_dt_s': 1/30, 'execute_steps': 3})+'\n')
    NEXT_STEP += 1
    return {'action_dt_s': 1/30, 'waypoints': applied, 'execute_steps': 3}
