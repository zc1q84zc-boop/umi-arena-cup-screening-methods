"""Fail-closed checks for a tuned online run, not a grasp-success classifier."""
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np

_spec = importlib.util.spec_from_file_location('tuned_rig_verifier',
    Path(__file__).parent/'simulator_profiles/tuned_v1/yubi_isaac_sim_env/wrist_rig.py')
_rig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rig)


def verify_right_hold_corridor(reset_q, hold, continuous):
    reset_q, hold = np.asarray(reset_q, dtype=np.float32), np.asarray(hold)
    assert reset_q.shape == hold.shape == (7,)
    assert np.max(np.abs(reset_q-hold)) <= .02, 'held right reset drift >.02 rad'
    lower, upper = np.minimum(reset_q, hold), np.maximum(reset_q, hold)
    for row in continuous:
        command = np.asarray(row['q'][1][:7])
        assert np.isfinite(command).all()
        assert np.all(command >= lower-2e-6) and np.all(command <= upper+2e-6), 'right governor left reset-to-hold corridor'
    np.testing.assert_allclose(continuous[-1]['q'][1][:7], hold, atol=2e-6)


def verify_closure_trace(trace, adapter_rows, extra):
    assert np.isfinite(extra) and 0 < extra <= .05
    for row in trace:
        index = row['action_index']
        model = adapter_rows[index//3]['substep_waypoints'][index%3]['left']['gripper_open_fraction']
        expected = max(0., model-extra*min(1., max(0., (.65-model)/.10)))
        audit = row['closure_command']
        np.testing.assert_allclose([audit['model_open_fraction'], audit['command_open_fraction'],
                                   audit['extra_closure_fraction_applied'], audit['max_extra_closure_fraction']],
                                  [model, expected, model-expected, extra], atol=1e-9, rtol=0)
        assert audit['force_limit_changed'] is False and audit['friction_changed'] is False
        assert audit['oracle_feedback'] is False
        assert row['closure_bias_is_diagnostic_assistance'] is True
        if model >= .65:
            assert audit['command_open_fraction'] == model, 'release command was altered'


def verify_official_lingbot_prompts(rows, episode):
    from deploy_servers.official_cup_prompts import cup_instruction, PROMPT_SOURCE, PROMPT_PROTOCOL
    counts = {'place_on_plate': 0, 'return_to_origin': 0}
    assert len(rows) == len(episode['transitions'])
    for row, transition in zip(rows, episode['transitions']):
        stage = transition['requested_task_stage']
        assert row['task_stage'] == stage
        expected = cup_instruction(stage)
        assert row['prompt'] == row['server_prompt'] == expected
        assert row['prompt_source'] == PROMPT_SOURCE and row['prompt_protocol'] == PROMPT_PROTOCOL
        mode = row['inference_mode']
        assert mode in ('native_lingbot_vla_v2_chunk','future_aligned_10hz_row0')
        if mode == 'future_aligned_10hz_row0':
            assert row['action_timing']['pose_rows'] == row['action_timing']['gripper_rows'] == [0]
            assert row['action_timing']['action_hz'] == 10
            assert row['target_update_hz'] == 10
            assert all(w == row['substep_waypoints'][0] for w in row['substep_waypoints'])
        previous = (episode['transitions'][row['step']-1]['task_stage'] if row['step'] else
                    'return_to_origin' if episode.get('left_return_diagnostic') else 'place_on_plate')
        assert stage == ('return_to_origin' if previous in ('return_to_origin', 'complete') else 'place_on_plate')
        counts[stage] += 1
    return {'verified': True, 'source': PROMPT_SOURCE, 'protocol': PROMPT_PROTOCOL,
            'requests_per_primitive': counts}


def verify(directory):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    assert report['status'] in ('completed','stopped'), report.get('error', report['status'])
    assert manifest['simulator_profile'] == 'tuned_online_v1'
    assert report['action_execution_hz'] == 30 and report['model_request_hz'] == 10
    n = report['episodes'][0]['policy_steps']
    assert n > 0
    rows = [json.loads(x) for x in (directory/'online_adapter.jsonl').read_text().splitlines()]
    cameras = [json.loads(x) for x in (directory/'wrist_camera_poses.jsonl').read_text().splitlines()]
    continuous = [json.loads(x) for x in (directory/'continuous_targets.jsonl').read_text().splitlines()]
    assert len(rows) == n and [r['step'] for r in rows] == list(range(n))
    executed_actions = len(cameras)-1
    interrupted = report['status']=='stopped' and report['episodes'][0].get('stop_reason')=='user_stop'
    # A stop marker can arrive after the first or second action of the final
    # three-action chunk. All previous chunks and all recorded frames must
    # still be complete. Never hide missing actions in a completed run.
    assert (3*(n-1)+1 <= executed_actions <= 3*n if interrupted else executed_actions == 3*n), \
        f'actions={executed_actions}, requests={n}, interrupted={interrupted}'
    from sim_console import SHELL_PROFILE_IDS, _PVC_NUMERICS
    shell_trial = report.get('cup_physics_profile', {}).get('id') in SHELL_PROFILE_IDS
    physics_hz = report.get('physics_hz', 60)
    expected_hz = _PVC_NUMERICS['physics_hz_for'](report['cup_physics_profile']['id']) if shell_trial else 60
    assert physics_hz == expected_hz, 'unexpected physics rate'
    assert len(continuous) == (physics_hz//30)*executed_actions, 'physics commands must match executed actions'
    if shell_trial:
        assert report['cup_physics_profile']['rigid_cup_colliders_disabled'] is True
        assert report['cup_physics_profile']['measured'] is False
        shapes = [json.loads(x) for x in (directory/'cup_deformation.jsonl').read_text().splitlines()]
        assert len(shapes) >= executed_actions
        assert all(np.isfinite(s['max_nodal_shape_change_m']) for s in shapes)
    for i, row in enumerate(rows):
        assert row['observation_origin'] == 'current_simulator_render_and_robot_state'
        assert row['future_observation_used'] is False
        assert row['calibration']['id'] == 'tuned_online_v1'
        assert row['calibration']['oracle_action_feedback'] is False
        assert row['calibration']['replay_specific_offsets'] is False
        assert row['gripper_calibration']['sim_closed_rad'] == -.1
        assert row['gripper_calibration']['sim_open_rad'] == .7
        assert row['execution_hz'] == 30 and len(row['substep_waypoints']) == 3
        for waypoint in row['substep_waypoints']:
            for side in ('left','right'):
                target = waypoint[side]
                p, q = np.asarray(target['position_m']), np.asarray(target['quaternion_wxyz'])
                fraction = float(target['gripper_open_fraction'])
                assert p.shape == (3,) and q.shape == (4,)
                assert np.isfinite(p).all() and np.isfinite(q).all()
                assert abs(np.linalg.norm(q)-1) < 1e-5
                assert np.isfinite(fraction) and 0 <= fraction <= 1
        for side in ('left','right'):
            meta = row['image_metadata'][side+'_wrist']
            view = cameras[i*3]['views'][side+'_wrist']
            assert view['robot_side'] == side
            np.testing.assert_allclose(meta['pose_world']['position_m'], view['camera_position_m'], atol=1e-5)
            a, b = np.asarray(meta['pose_world']['quaternion_wxyz']), np.asarray(view['camera_quaternion_wxyz'])
            assert abs(np.dot(a,b)) > 1-1e-5
    previous_v = np.zeros((2,8))
    for row in continuous:
        q, v = np.asarray(row['q']), np.asarray(row['v'])
        assert q.shape == (2,9) and v.shape == (2,8) and np.isfinite(q).all() and np.isfinite(v).all()
        assert np.max(abs(q[:,-2]+q[:,-1])) < 1e-6
        assert np.max(abs(v)) <= .8+1e-8
        assert np.max(abs(v-previous_v))*physics_hz <= 1.5+1e-7
        previous_v = v
    for index, row in enumerate(cameras):
        assert row['sample_index'] == index
        if index:
            assert abs(row['physics_time_s']-cameras[index-1]['physics_time_s']-1/30) < 1e-6
        for name, view in row['views'].items():
            p,q = _rig.camera_pose(view['base_pose'],manifest['cameras'][name]['rigid_mount'])
            np.testing.assert_allclose(p,view['camera_position_m'],atol=1e-5)
            assert abs(np.dot(q,view['camera_quaternion_wxyz'])) > 1-1e-5
            assert abs(view['clipping_range_m'][0]-.01) < 1e-6
    for filename in ('video.mp4','video_left_wrist.mp4','video_right_wrist.mp4'):
        data = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                 '-show_entries','stream=nb_read_frames,avg_frame_rate','-of','json',str(directory/filename)]))
        stream = data['streams'][0]
        assert int(stream['nb_read_frames']) == len(cameras) and stream['avg_frame_rate'] == '30/1'
    diagnostic = report.get('left_return_diagnostic', False)
    assert manifest.get('left_return_diagnostic', False) == diagnostic
    extra = report.get('left_extra_closure_fraction', 0.)
    assert np.isfinite(extra) and 0 <= extra <= .05
    assert manifest.get('left_extra_closure_fraction', 0.) == extra
    assert diagnostic or extra == 0, 'closure assistance was applied to a baseline run'
    if diagnostic:
        episode = report['episodes'][0]
        assert episode['full_task_success'] is False and episode['success'] is False
        initial = episode['initial_observation']
        cup, plate = initial['objects']['cup'], initial['objects']['plate']
        assert np.linalg.norm(np.array(cup['position_m'][:2])-plate['position_m'][:2]) <= .02
        assert .001 <= cup['position_m'][2]-plate['position_m'][2] <= .010
        right = initial['robots']['right']
        hold = [right['joint_positions'][right['joint_names'].index(f'panda_joint{i}')] for i in range(1,8)]
        for transition in episode['transitions']:
            np.testing.assert_allclose(transition['action']['right']['arm_joint_targets_rad'], hold, atol=1e-9)
        # The governor starts at commanded reset joints, while the hold target
        # is the actual gravity-settled reset observation. Permit only the
        # bounded transition BETWEEN those two known values, not arbitrary
        # right-arm commands or a relaxed global hold tolerance.
        verify_right_hold_corridor(initial['scenario']['initial_arm_joint_rad']['right'], hold, continuous)
        for state in (episode['final_observation'],):
            measured = [state['robots']['right']['joint_positions'][right['joint_names'].index(f'panda_joint{i}')]
                        for i in range(1,8)]
            assert np.max(np.abs(np.array(measured)-hold)) <= .02, 'held right arm drifted >.02 rad'
        trace = [json.loads(x) for x in (directory/'left_return_diagnostic.jsonl').read_text().splitlines()]
        assert [row['action_index'] for row in trace] == list(range(executed_actions))
        assert all(row['full_task_success'] is False and row['plate_placed_by_reset'] for row in trace)
        if extra:
            verify_closure_trace(trace, rows, extra)
            for transition in episode['transitions']:
                index = transition['policy_step']*3+2
                # A manually stopped chunk may contain only one or two actions.
                index = min(index, len(trace)-1)
                np.testing.assert_allclose(transition['action']['left']['gripper_open_fraction'],
                    trace[index]['closure_command']['command_open_fraction'], atol=1e-9, rtol=0)
    official_prompts = (verify_official_lingbot_prompts(rows, report['episodes'][0])
                        if rows[0].get('prompt_protocol') else None)
    return {'validated': True, 'model_requests': n, 'executed_actions': executed_actions,
            'predicted_actions': 3*n, 'last_chunk_actions_executed': executed_actions-3*(n-1),
            'interrupted_final_chunk': executed_actions != 3*n,
            'video_frames_per_view': len(cameras), 'simulator_profile': 'tuned_online_v1',
            'current_camera_pose_matches_model_input': True,
            'mirrored_jaw_command_verified': True, 'continuous_limits_verified': True,
            'left_phase_diagnostic': diagnostic,
            'left_extra_closure_fraction': extra,
            'closure_intervention_audited': bool(extra),
            'official_primitive_prompts': official_prompts,
            'task_success': report['episodes'][0]['success'], 'integration_audit_not_success_claim': True}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
