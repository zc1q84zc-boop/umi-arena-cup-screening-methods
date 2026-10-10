import copy
import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from quality_rules import COLUMNS,episode_metrics
from prepare_selection import select,TASKS
from scan_quality import scan_shard,export,table_rows
from build_intersection import validated_passes,FLAGS


def motion(episode=11):
    rows=[]
    for i in range(4):
        values=[episode,i,i/30,[i*.001,0,0,0,0,0,1],[.3+i*.001,0,0,0,0,0,1],
                [.3,0,0,0,0,0,1],[.2,.2],[.2,.2],
                [.001 if i else 0,0,0,0,0,0,1],[.001 if i else 0,0,0,0,0,0,1]]
        rows.append(dict(zip(COLUMNS,values)))
    return rows


def meta(idx,uuid='record',success=True,task='phone'):
    return dict(episode_index=idx,uuid=uuid,length=4,**{'data/chunk_index':0,'data/file_index':0},
                dataset_from_index=4*idx,dataset_to_index=4*idx+4,
                primitive_action=['stage'],short_horizon_task=[TASKS[task]],
                success_short_horizon_task=success,task_success=False)


class Tests(unittest.TestCase):
    def test_ik_pass_requires_full_scope_and_original_provenance(self):
        selected,full,_,_=select([meta(11),meta(12,success=False)])
        label=dict(uuid='record',episode_indices=[11,12],frames=8,reference_only=True,
                   three_reference_intersection_pass=True,**{k:True for k in FLAGS})
        complete=dict(all_records_processed=True,processed_record_ids=['record'],
                      source_script_sha256='a'*64,reference_config_sha256='b'*64,
                      method='original_colleague_three_reference_full_frame')
        self.assertEqual(validated_passes([label],full,complete),{11,12})
        label['episode_indices']=[11]
        with self.assertRaises(ValueError):validated_passes([label],full,complete)

    def test_missing_ik_run_not_treated_as_a_completed_screen(self):
        _,full,_,_=select([meta(11)])
        with self.assertRaises(ValueError):validated_passes([],full,{})

    def test_same_rules_valid_motion_and_jump(self):
        self.assertEqual(episode_metrics(motion(),11,4)['flags_for_review_only'],[])
        rows=motion();rows[2][COLUMNS[4]][0]+=.15
        flags=episode_metrics(rows,11,4)['flags_for_review_only']
        self.assertIn('right_large_one_tick_translation',flags)
        self.assertIn('right_action_observation_misaligned',flags)

    def test_nonfinite_is_quality_candidate(self):
        rows=motion();rows[1][COLUMNS[3]][0]=float('nan')
        self.assertIn('nonfinite_signal',episode_metrics(rows,11,4)['flags_for_review_only'])

    def test_success_scope_and_full_record_are_distinct(self):
        a=meta(11);b=meta(12,success=False);c=meta(13,uuid='unrelated');c['short_horizon_task']=['Phone charge']
        selected,full,_,_=select([a,b,c])
        self.assertEqual([r['episode_index'] for r in selected],[11])
        self.assertEqual([r['episode_index'] for r in full],[11,12])
        self.assertFalse(selected[0]['recorded_task_success'])

    def test_duplicate_episode_metadata_rejected(self):
        with self.assertRaises(ValueError):select([meta(11),meta(11)])

    def test_cross_variant_record_rejected(self):
        with self.assertRaises(ValueError):select([meta(11),meta(12,task='chain_sps')])

    def test_neighboring_shard_recovery_preserves_frame_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'data/chunk-000').mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(motion(99)),root/'data/chunk-000/file-000.parquet')
            pq.write_table(pa.Table.from_pylist(motion(11)),root/'data/chunk-000/file-001.parquet')
            selected,_,_,_=select([meta(11)])
            result=scan_shard(str(root),0,0,selected)['episodes'][0]
            self.assertEqual(result['flags_for_review_only'],[])
            self.assertEqual(result['corrected_source_shards'],[[0,1]])
            self.assertEqual(result['frames'],4)
            table_rows.cache_clear()

    def test_episode_quarantine_is_union_and_source_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'raw';root.mkdir();out=Path(temp)/'out'
            for key in ('data','videos','meta'):(root/key).mkdir()
            preserved=root/'data/source.txt';preserved.write_text('immutable')
            selected,_,_,_=select([meta(11),meta(12),meta(13)])
            results=[dict(episode_index=i,frames=4,flags_for_review_only=['high_speed'] if i==12 else []) for i in (11,12,13)]
            summary=export(root,selected,results,[dict(episode_index=13,review_status='confirmed_anomaly')],out)
            self.assertEqual(summary['phone']['retained_episodes'],1)
            m=json.loads((out/'quality/phone/clean_dataset/manifest.json').read_text())
            self.assertEqual([r['episode_index'] for r in m['episodes']],[11])
            self.assertEqual(preserved.read_text(),'immutable')
            self.assertTrue((out/'quality/phone/clean_dataset/data').is_symlink())


if __name__=='__main__':unittest.main()
