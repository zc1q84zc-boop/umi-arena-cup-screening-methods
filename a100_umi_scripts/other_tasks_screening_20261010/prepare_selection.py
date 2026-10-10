"""Select exact dataset task names; keep related task variants explicitly separate."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

TASKS={'phone':'Pack Smartphone into Box','chain_sps':'Set Parts Supply (chain)'}
COLUMNS=['episode_index','uuid','length','data/chunk_index','data/file_index',
         'dataset_from_index','dataset_to_index','primitive_action','short_horizon_task',
         'success_short_horizon_task','task_success']
OFFICIAL_PHONE=[
    'Pick up and insert the divider into the box with the right gripper.',
    'Stand the smartphone upright with the left gripper.',
    'Pick up and lift the smartphone with the right gripper.',
    'Place the smartphone into the box with the right gripper.',
    'Close the lid with the right gripper.',
    'Open the lid with the right gripper.',
    'Take the smartphone out of the box with the right gripper.',
    'Remove the divider with the right gripper.']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def row_manifest(row,task_id):
    return dict(episode_index=int(row['episode_index']),uuid=row['uuid'],task_id=task_id,
                task=TASKS[task_id],length=int(row['length']),
                data_chunk_index=int(row['data/chunk_index']),data_file_index=int(row['data/file_index']),
                dataset_from_index=int(row['dataset_from_index']),dataset_to_index=int(row['dataset_to_index']),
                primitive_actions=row['primitive_action'],original_short_horizon_tasks=row['short_horizon_task'],
                success_short_horizon_task=bool(row['success_short_horizon_task']),
                recorded_task_success=bool(row['task_success']))


def select(rows):
    all_rows={};successful=[];target_records={};counts=Counter();success_counts=Counter()
    for row in rows:
        idx=int(row['episode_index'])
        if idx in all_rows:raise ValueError('Duplicate metadata episode')
        all_rows[idx]=row
        for t in row['short_horizon_task']:counts[t]+=1;success_counts[t]+=bool(row['success_short_horizon_task'])
        match=[key for key,t in TASKS.items() if t in row['short_horizon_task']]
        if len(match)>1:raise ValueError('Ambiguous target task')
        if match:
            key=match[0]
            if row['uuid'] in target_records and target_records[row['uuid']]!=key:raise ValueError('Ambiguous record task')
            target_records[row['uuid']]=key
            if row['success_short_horizon_task']:successful.append(row_manifest(row,key))
    full_records=[]
    for row in all_rows.values():
        if row['uuid'] in target_records:
            full_records.append(row_manifest(row,target_records[row['uuid']]))
    return sorted(successful,key=lambda r:r['episode_index']),sorted(full_records,key=lambda r:r['episode_index']),counts,success_counts


def manifest(rows,policy):
    counts={key:dict(task_name=t,episodes=sum(r['task_id']==key for r in rows),
                     frames=sum(r['length'] for r in rows if r['task_id']==key)) for key,t in TASKS.items()}
    return dict(summary=dict(selection_policy=policy,tasks=counts,total_episodes=len(rows),
                             total_frames=sum(r['length'] for r in rows),fps=30),episodes=rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    target=a.output/'selection'
    if target.exists():raise FileExistsError(target)
    rows=[];hashes={}
    for p in sorted((a.dataset_root/'meta/episodes').rglob('*.parquet')):
        rows.extend(pq.read_table(p,columns=COLUMNS,use_threads=False).to_pylist())
        hashes[str(p.relative_to(a.dataset_root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    kept,full,counts,success_counts=select(rows)
    phone_actions={p for r in full if r['task_id']=='phone' for p in r['primitive_actions']}
    if phone_actions!=set(OFFICIAL_PHONE):raise ValueError('Phone primitive labels differ from expected eight-stage task')
    source=manifest(kept,'exact short_horizon_task AND successful; related chain SPS kept separate from official YUBI SPS')
    write(target/'manifest.json',source)
    write(target/'full_record_manifest.json',manifest(full,'all PA episodes of target UUIDs, including unsuccessful episodes; IK input only'))
    discovery=dict(source_dataset=str(a.dataset_root.resolve()),tasks=[dict(task=t,episodes=n,successful=success_counts[t]) for t,n in sorted(counts.items())],
                   official_scope=dict(phone=dict(status='exact_match',task_name=TASKS['phone']),
                                       pens=dict(status='missing',reason='No pen-holder task or matching primitives in release metadata'),
                                       sps=dict(status='missing',reason='Chain/sprocket/bolt SPS is a different variant from official YUBI parts SPS'),
                                       cable=dict(status='missing',reason='Phone charge is different from YUBI MCU USB-C insertion')),
                   related_scope=dict(chain_sps=dict(status='related_variant',task_name=TASKS['chain_sps'])),
                   source_metadata_sha256=hashes,selection=source['summary'],full_record_ik_input=manifest(full,'IK input')['summary'])
    write(a.output/'discovery.json',discovery)
    print(json.dumps(source['summary'],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
