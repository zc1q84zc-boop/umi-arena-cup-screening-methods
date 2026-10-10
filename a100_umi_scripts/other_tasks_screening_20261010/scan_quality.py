"""Resumable motion-quality scan using exactly the existing cup metric function."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time
import pyarrow.parquet as pq
from quality_rules import COLUMNS,episode_metrics,adjacent_shards
from prepare_selection import TASKS,manifest,write


@lru_cache(maxsize=3)
def table_rows(root,chunk,index,allowed_ids):
    path=Path(root)/f'data/chunk-{chunk:03d}/file-{index:03d}.parquet'
    if not path.is_file():return {}
    groups=defaultdict(list)
    # Keep identical per-frame metrics while avoiding Python conversion of
    # unrelated episodes packed in the same shard. Arrow still reads raw data.
    for row in pq.read_table(path,columns=COLUMNS,filters=[('episode_index','in',list(allowed_ids))],use_threads=False).to_pylist():
        groups[int(row['episode_index'])].append(row)
    return groups


def scan_shard(root,chunk,index,episodes):
    allowed_ids=tuple(sorted(r['episode_index'] for r in episodes))
    groups=table_rows(root,chunk,index,allowed_ids);results=[]
    for meta in episodes:
        idx=meta['episode_index'];rows=list(groups.get(idx,[]));corrected=[]
        if len(rows)!=meta['length']:
            for nearby_chunk,nearby_index in adjacent_shards(chunk,index):
                more=table_rows(root,nearby_chunk,nearby_index,allowed_ids).get(idx,[])
                if more:rows.extend(more);corrected.append([nearby_chunk,nearby_index])
                if len(rows)>=meta['length']:break
        rows.sort(key=lambda r:r['frame_index'])
        if rows:result=episode_metrics(rows,idx,meta['length'])
        else:result=dict(episode_index=idx,frames=0,flags_for_review_only=['episode_missing_from_shard'],candidate_events=[])
        result.update(task_id=meta['task_id'],uuid=meta['uuid'])
        if corrected:result['corrected_source_shards']=corrected
        results.append(result)
    return dict(chunk=chunk,file_index=index,episodes=results)


def export(root,rows,results,human,output):
    selected={r['episode_index']:r for r in rows};scanned={r['episode_index']:r for r in results}
    assert len(selected)==len(rows) and len(scanned)==len(results) and set(scanned)==set(selected)
    human_ids={int(r['episode_index']) for r in human if r.get('review_status')=='confirmed_anomaly'}
    if not human_ids<=set(selected):raise ValueError('Human labels outside selected task scope')
    summaries={}
    for key in TASKS:
        source=[r for r in rows if r['task_id']==key]
        bad={r['episode_index'] for r in source if scanned[r['episode_index']]['flags_for_review_only'] or r['episode_index'] in human_ids}
        kept=[r for r in source if r['episode_index'] not in bad];quarantine=[r for r in source if r['episode_index'] in bad]
        out=output/'quality'/key
        if out.exists():raise FileExistsError(out)
        for name,items in [('clean_dataset',kept),('quarantined_dataset',quarantine)]:
            view=out/name
            write(view/'manifest.json',manifest(items,'unchanged cup quality rules; conservative reversible episode quarantine'))
            write(view/'episode_indices.json',[r['episode_index'] for r in items])
            write(view/'episode_prompts.json',{str(r['episode_index']):r['primitive_actions'] for r in items})
            for directory in ('data','videos','meta'):os.symlink(Path(root)/directory,view/directory,target_is_directory=True)
            (view/'README.md').write_text('Index-backed selection view: use manifest.json as the mandatory training allowlist. Raw linked files still include excluded episodes. Original primitive labels are preserved.\n')
        decisions=[]
        for r in source:
            item=scanned[r['episode_index']]
            decisions.append(dict(episode_index=r['episode_index'],uuid=r['uuid'],decision='quarantine' if r['episode_index'] in bad else 'keep',
                                  machine_flags=item['flags_for_review_only'],candidate_events=item.get('candidate_events',[]),
                                  human_confirmed=r['episode_index'] in human_ids,corrected_source_shards=item.get('corrected_source_shards',[]),
                                  evidence_tier='human_confirmed' if r['episode_index'] in human_ids else 'machine_candidate_only' if item['flags_for_review_only'] else 'no_rule_trigger'))
        (out/'episode_decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in decisions))
        summaries[key]=dict(task_name=TASKS[key],input_episodes=len(source),input_frames=sum(r['length'] for r in source),
                            retained_episodes=len(kept),retained_frames=sum(r['length'] for r in kept),
                            quarantined_episodes=len(bad),quarantined_frames=sum(r['length'] for r in quarantine),
                            human_confirmed_episodes=len(human_ids & {r['episode_index'] for r in source}),
                            machine_flagged_episodes=sum(bool(scanned[r['episode_index']]['flags_for_review_only']) for r in source),
                            scanned_frames=sum(scanned[r['episode_index']]['frames'] for r in source),
                            nonexclusive_flag_counts=dict(Counter(flag for r in source for flag in scanned[r['episode_index']]['flags_for_review_only'])),
                            source_location_corrected_episodes=sum(bool(scanned[r['episode_index']].get('corrected_source_shards')) for r in source),
                            reference_ik_status='pending_original_colleague_method',final_intersection_status='pending_reference_ik',
                            raw_source_modified=False)
        write(out/'summary.json',summaries[key])
    write(output/'quality_summary.json',summaries)
    return summaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--human-labels',type=Path);a=p.parse_args()
    assert 1<=a.workers<=4
    os.environ.setdefault('OMP_NUM_THREADS','1')
    source=json.loads((a.output/'selection/manifest.json').read_text());by_shard=defaultdict(list)
    for r in source['episodes']:by_shard[r['data_chunk_index'],r['data_file_index']].append(r)
    progress=a.output/'quality_progress.jsonl';completed=set();results=[]
    if progress.exists():
        for line in progress.read_text().splitlines():
            value=json.loads(line);key=(value['chunk'],value['file_index'])
            if key in completed:raise ValueError('Duplicate completed shard')
            completed.add(key);results.extend(value['episodes'])
    todo=[(key,eps) for key,eps in by_shard.items() if key not in completed]
    write(a.output/'status.json',dict(phase='quality_scanning',started_unix_s=time.time(),total_shards=len(by_shard),completed_shards=len(completed),input_episodes=len(source['episodes']),workers=a.workers))
    with ProcessPoolExecutor(max_workers=a.workers) as pool,progress.open('a') as stream:
        futures={pool.submit(scan_shard,str(a.dataset_root),*key,eps):key for key,eps in todo}
        for future in as_completed(futures):
            value=future.result();stream.write(json.dumps(value)+'\n');stream.flush();os.fsync(stream.fileno())
            completed.add((value['chunk'],value['file_index']));results.extend(value['episodes'])
            status=dict(phase='quality_scanning',completed_shards=len(completed),total_shards=len(by_shard),scanned_episodes=len(results),input_episodes=len(source['episodes']),workers=a.workers)
            write(a.output/'status.json',status)
            if len(completed)%20==0:print(json.dumps(status),flush=True)
    human=[json.loads(line) for line in a.human_labels.read_text().splitlines() if line] if a.human_labels else []
    summaries=export(a.dataset_root,source['episodes'],results,human,a.output)
    rules=Path(__file__).with_name('quality_rules.py')
    write(a.output/'quality_provenance.json',dict(source_manifest_sha256=hashlib.sha256((a.output/'selection/manifest.json').read_bytes()).hexdigest(),
         original_quality_rule_sha256=hashlib.sha256(rules.read_bytes()).hexdigest(),all_selected_frames_scanned=True,quality_method='same episode_metrics as cup screen',new_human_review_claim=False,
         successful_scope=source['summary'],source_dataset=str(a.dataset_root.resolve())))
    write(a.output/'status.json',dict(phase='quality_completed_waiting_for_ik',quality_completed=True,ik_completed=False,intersection_completed=False,total_shards=len(by_shard),scanned_episodes=len(results)))
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
