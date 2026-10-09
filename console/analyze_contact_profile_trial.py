"""Compare immutable run artifacts; do not rewrite their original reports."""
import json
import math
from pathlib import Path
import argparse


def summarize(path):
    path=Path(path)
    report=json.loads((path/'report.json').read_text());episode=report['episodes'][0]
    rows=[json.loads(x) for x in (path/'gripper_contact_audit.jsonl').read_text().splitlines()]
    initial=rows[0]['cup_position_m'][2]
    maximum=max(rows,key=lambda r:r['cup_position_m'][2])
    upright=max((r['cup_position_m'][2]-initial for r in rows if
                  1-2*sum(x*x for x in r['cup_quaternion_wxyz'][1:3]) >= math.cos(math.radians(15))), default=0)
    return dict(id=path.name,status=report['status'],task_success=episode['full_task_success'],
        plate_placed=episode['plate_placed'],requests=episode['policy_steps'],
        max_cup_root_lift_mm=1000*(maximum['cup_position_m'][2]-initial),
        max_upright_root_lift_mm=1000*upright,
        contact_summary=episode.get('gripper_contact_audit_summary'),
        note='Contact magnitudes are uncalibrated PhysX telemetry; cup-root lift alone is not stable grasp proof.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+');parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=dict(runs=[summarize(p) for p in args.runs])
    text=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:f.write(text+'\n')
