"""Task specifications are explicit JSON; importing needs no Isaac runtime."""
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASK_IDS = ('pens', 'sps', 'cable', 'phone')
COUNTS = dict(pens=12, sps=15, cable=1, phone=8)


def load_task(task_id):
    if task_id not in TASK_IDS:
        raise ValueError(f'Unknown task: {task_id}; choose {TASK_IDS}')
    value = json.loads((ROOT/'configs'/f'{task_id}.json').read_text())
    ids = [o['id'] for o in value['objects']]
    if len(set(ids)) != len(ids) or len(value['primitives']) != COUNTS[task_id]:
        raise ValueError('Invalid object IDs or official primitive count')
    for obj in value['objects']:
        if any(not math.isfinite(v) or v <= 0 for v in obj['size_m']):
            raise ValueError(f'Invalid dimensions: {obj["id"]}')
        if obj['mass_kg'] <= 0 or not math.isfinite(obj['mass_kg']):
            raise ValueError('Mass must be positive and finite')
    for p in value['primitives']:
        if p['object'] not in ids or (p.get('target') and p['target'] not in ids):
            raise ValueError('Primitive references a missing object')
    return copy.deepcopy(value)


def task_catalog():
    return [dict(task_id=k, label=v['label'], primitive_count=len(v['primitives']),
                 source_url=v['source_url'], geometry_status=v['geometry_status'])
            for k in TASK_IDS for v in [load_task(k)]]
