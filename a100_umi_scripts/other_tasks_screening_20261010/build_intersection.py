"""Intersect quality views with completed original three-reference IK results."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from prepare_selection import TASKS,manifest,write

FLAGS=('franka_fr3_left','franka_fr3_right','openarm_v2_left','openarm_v2_right','agibot_g2_reference_dual')


def validated_passes(records,full_rows,completion):
    expected=defaultdict(set);frames=defaultdict(int)
    for r in full_rows:expected[r['uuid']].add(r['episode_index']);frames[r['uuid']]+=r['length']
    # A pass-only allowlist cannot establish that missing records were checked.
    if completion.get('all_records_processed') is not True:raise ValueError('IK job completion not certified by its run metadata')
    if set(completion.get('processed_record_ids',[]))!=set(expected):raise ValueError('IK completion scope does not match every target UUID')
    for k in ('source_script_sha256','reference_config_sha256'):
        value=completion.get(k,'')
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Original colleague script/config hashes required')
    if completion.get('method')!='original_colleague_three_reference_full_frame':raise ValueError('Different IK method cannot be presented as original colleague filtering')
    seen=set();passed=set()
    for r in records:
        uid=r['uuid']
        if uid in seen or uid not in expected:raise ValueError('Duplicate or out-of-scope IK record')
        seen.add(uid)
        if set(map(int,r['episode_indices']))!=expected[uid]:raise ValueError('Incomplete full-record IK scope')
        if int(r['frames'])!=frames[uid]:raise ValueError('IK result frame count differs from source record')
        if r.get('reference_only') is not True:raise ValueError('Reference interpretation must be explicit')
        if any(type(r.get(k)) is not bool for k in (*FLAGS,'three_reference_intersection_pass')):raise ValueError('Missing boolean IK flags')
        ok=all(r[k] for k in FLAGS)
        if r['three_reference_intersection_pass']!=ok:raise ValueError('Inconsistent three-reference conjunction')
        if ok:passed.update(expected[uid])
    return passed


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--ik-record-labels',type=Path,required=True);p.add_argument('--ik-completion',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);a=p.parse_args()
    full=json.loads((a.run_root/'selection/full_record_manifest.json').read_text())['episodes']
    results=[json.loads(line) for line in a.ik_record_labels.read_text().splitlines() if line]
    completion=json.loads(a.ik_completion.read_text());passed=validated_passes(results,full,completion)
    out=a.run_root/'quality_ik_intersection'
    if out.exists():raise FileExistsError(out)
    summaries={}
    for key in TASKS:
        quality_path=a.run_root/f'quality/{key}/clean_dataset/manifest.json'
        source=json.loads(quality_path.read_text())['episodes'];selected=[r for r in source if r['episode_index'] in passed]
        view=out/key
        write(view/'manifest.json',manifest(selected,'quality-clean episode allowlist INTERSECTION original full-frame three-reference IK record allowlist'))
        write(view/'episode_indices.json',[r['episode_index'] for r in selected])
        write(view/'episode_prompts.json',{str(r['episode_index']):r['primitive_actions'] for r in selected})
        for directory in ('data','videos','meta'):os.symlink(a.dataset_root/directory,view/directory,target_is_directory=True)
        membership=defaultdict(list)
        for r in selected:membership[r['uuid']].append(r['episode_index'])
        original=defaultdict(list)
        for r in full:
            if r['task_id']==key:original[r['uuid']].append(r['episode_index'])
        records=[dict(uuid=uid,original_episode_indices=ids,retained_episode_indices=membership[uid],
                      partial_record=bool(membership[uid]) and set(membership[uid])!=set(ids)) for uid,ids in sorted(original.items())]
        (view/'record_membership.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        summary=dict(task_name=TASKS[key],quality_clean_episodes=len(source),ik_pass_episodes=sum(r['episode_index'] in passed for r in full if r['task_id']==key),
                     intersection_episodes=len(selected),intersection_frames=sum(r['length'] for r in selected),
                     intersection_records=len(membership),partial_records=sum(r['partial_record'] for r in records),
                     reference_only=True,continuity_certified=False,collision_certified=False,raw_source_modified=False,
                     full_task_success_certified=False,pairing_or_merge_applied=False)
        write(view/'summary.json',summary);summaries[key]=summary
        (view/'README.md').write_text('Use manifest.json as the mandatory allowlist. Quality filtering is episode-level, IK is full-record reference filtering. The intersection may retain only some task phases. No pairing/merge, prompt rewrite or raw data rewrite is applied. Reference IK does not certify continuous, collision-free or real-robot execution.\n')
    write(out/'summary.json',summaries)
    write(out/'provenance.json',dict(ik_completion=completion,inputs_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.ik_record_labels,a.ik_completion,a.run_root/'selection/full_record_manifest.json']},method='quality/IK exact episode intersection'))
    write(a.run_root/'status.json',dict(phase='complete',quality_completed=True,ik_completed=True,intersection_completed=True,tasks=summaries))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
