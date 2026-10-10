"""Audit exported selections against every completed per-episode scan."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from prepare_selection import TASKS, write

RULE_SHA256 = '7b09e84996cbd829e8731a5713b95c41fe3a99df33c6c5e8a49ae688e86a7a44'


def load(path):
    return json.loads(path.read_text())


def unique(rows):
    values = {r['episode_index']: r for r in rows}
    assert len(values) == len(rows), 'Duplicate episode'
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    status = load(root / 'status.json')
    assert status['quality_completed'] is True
    selected = unique(load(root / 'selection/manifest.json')['episodes'])
    progress = [json.loads(line) for line in (root / 'quality_progress.jsonl').read_text().splitlines()]
    assert len({(r['chunk'], r['file_index']) for r in progress}) == len(progress) == status['total_shards']
    scanned = unique([r for shard in progress for r in shard['episodes']])
    assert set(scanned) == set(selected), 'Scan coverage differs from selected episodes'
    missing_frames = [i for i in selected if scanned[i]['frames'] != selected[i]['length']]
    summaries = load(root / 'quality_summary.json')
    verified = {}
    review = {}
    for task in TASKS:
        source = {i: r for i, r in selected.items() if r['task_id'] == task}
        out = root / 'quality' / task
        clean = unique(load(out / 'clean_dataset/manifest.json')['episodes'])
        quarantine = unique(load(out / 'quarantined_dataset/manifest.json')['episodes'])
        assert not set(clean) & set(quarantine)
        assert set(clean) | set(quarantine) == set(source)
        assert {**clean, **quarantine} == source, 'Source labels or metadata changed'
        decisions = unique([json.loads(line) for line in (out / 'episode_decisions.jsonl').read_text().splitlines()])
        assert set(decisions) == set(source)
        for idx, row in decisions.items():
            assert row['machine_flags'] == scanned[idx]['flags_for_review_only']
            assert (row['decision'] == 'quarantine') == (idx in quarantine)
            assert (idx in quarantine) == bool(row['machine_flags'] or row['human_confirmed'])
        for name, values in [('clean_dataset', clean), ('quarantined_dataset', quarantine)]:
            view = out / name
            assert load(view / 'episode_indices.json') == list(values)
            assert load(view / 'episode_prompts.json') == {str(i): r['primitive_actions'] for i, r in values.items()}
            for directory in ('data', 'meta', 'videos'):
                link = view / directory
                assert link.is_symlink() and link.resolve().is_dir()
        summary = summaries[task]
        counts = dict(input_episodes=len(source), input_frames=sum(r['length'] for r in source.values()),
                      retained_episodes=len(clean), retained_frames=sum(r['length'] for r in clean.values()),
                      quarantined_episodes=len(quarantine), quarantined_frames=sum(r['length'] for r in quarantine.values()),
                      scanned_frames=sum(scanned[i]['frames'] for i in source))
        assert all(summary[k] == value for k, value in counts.items())
        assert summary['nonexclusive_flag_counts'] == dict(Counter(flag for i in source for flag in scanned[i]['flags_for_review_only']))
        assert load(out / 'summary.json') == summary
        verified[task] = {**counts, 'retained_percent': round(100 * len(clean) / len(source), 2)}
        angular_flags = {'left_fr3_angular_speed_conflict', 'right_fr3_angular_speed_conflict'}
        by_primitive = {}
        for idx, row in source.items():
            for primitive in row['primitive_actions']:
                tally = by_primitive.setdefault(primitive, {'input_episodes': 0, 'retained_episodes': 0})
                tally['input_episodes'] += 1
                tally['retained_episodes'] += idx in clean
        review[task] = dict(
            angular_only_quarantined_episodes=sum(bool(scanned[i]['flags_for_review_only']) and set(scanned[i]['flags_for_review_only']) <= angular_flags for i in quarantine),
            other_flag_quarantined_episodes=sum(bool(set(scanned[i]['flags_for_review_only']) - angular_flags) for i in quarantine),
            primitive_counts=by_primitive,
            interpretation='Machine candidates under unchanged cup thresholds; not newly human-confirmed anomalies. Raw quarantine remains recoverable.')
    rule_path = Path(__file__).with_name('quality_rules.py')
    assert hashlib.sha256(rule_path.read_bytes()).hexdigest() == RULE_SHA256
    provenance_path = root / 'quality_provenance.json'
    provenance = load(provenance_path)
    assert provenance['original_quality_rule_sha256'] == RULE_SHA256
    assert provenance['source_manifest_sha256'] == hashlib.sha256((root / 'selection/manifest.json').read_bytes()).hexdigest()
    # The receipt derives coverage from actual output, including any missing rows.
    provenance['all_selected_frames_scanned'] = not missing_frames
    provenance['episodes_with_length_mismatch'] = missing_frames
    write(provenance_path, provenance)
    receipt = dict(quality_export_verified=True, tasks=verified,
                   selected_episodes=len(selected), scanned_episodes=len(scanned), scanned_shards=len(progress),
                   expected_frames=sum(r['length'] for r in selected.values()),
                   scanned_frames=sum(r['frames'] for r in scanned.values()),
                   all_selected_frames_scanned=not missing_frames,
                   episodes_with_length_mismatch=missing_frames,
                   original_quality_rule_sha256=RULE_SHA256,
                   raw_labels_and_prompts_preserved=True, views_use_raw_source_symlinks=True,
                   ik_completed=status['ik_completed'], intersection_completed=status['intersection_completed'])
    write(root / 'verification.json', receipt)
    write(root / 'quality_review_summary.json', review)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
