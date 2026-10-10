"""Use the existing runner with explicit right-home-before-left sequencing."""
import copy
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
CONSOLE = HERE.parents[1]
RUNTIME = CONSOLE.parent / 'dual-franka-yubi-isaac-sim-deploy'
TUNED = CONSOLE / 'simulator_profiles/tuned_v1'
sys.path.insert(0, str(TUNED))
from jaw_margin import adjust
from sequencing import Sequencer
from yubi_isaac_sim_env import run, run_visual_aligned, two_stage_task
from yubi_isaac_sim_env.interarm_guard import limit_interarm_motion

OUTPUT = Path(os.environ['SIM_ADAPTER_AUDIT_DIR'])
MARGIN = .005


def main():
    model_id = os.environ['UMI_EVAL_MODEL_ID']
    model_unit = os.environ['UMI_EVAL_MODEL_UNIT']
    model_url = os.environ['UMI_EVAL_MODEL_URL']
    simulation_gpu = int(os.environ['ISAAC_ACTIVE_GPU'])
    inference_gpu = int(os.environ['UMI_EVAL_MODEL_GPU'])
    assert model_id in ('pi05-cup-clean-10000', 'pi05-cup-clean-20000', 'pi05-cup-clean-30000', 'pi05-cup-intersection-10000', 'pi05-cup-intersection-20000', 'pi05-cup-intersection-30000')
    lock_name = f'gpu-eval-{simulation_gpu}.lock' if (os.environ.get('UMI_EVAL_PARALLEL_PROTECTED') == '1' or os.environ.get('UMI_EVAL_CONSOLE_LEASE') == '1') else 'console.lock'
    simulation_slot='primary'
    if os.environ.get('UMI_EVAL_PARALLEL_LANE')=='secondary':
        assert os.environ.get('UMI_EVAL_PARALLEL_PROTECTED')=='1' and simulation_gpu==0 and model_id=='pi05-cup-intersection-20000'
        buddy='umi-fast-full-30000-20261010-r2.service'
        assert buddy in os.environ.get('UMI_EVAL_PARALLEL_ALLOWED_UNITS','').split(',')
        buddy_pid=int(subprocess.check_output(['systemctl','--user','show',buddy,'-p','MainPID','--value']))
        if buddy_pid:
            cmd=Path(f'/proc/{buddy_pid}/cmdline').read_bytes().decode().replace('\0',' ')
            environ=Path(f'/proc/{buddy_pid}/environ').read_bytes().split(b'\0')
            assert str(HERE/'full_task_sequential.py') in cmd and 'fast_housing_pi05_30000_full_20261010' in cmd and b'ISAAC_ACTIVE_GPU=0' in environ
            simulation_slot='secondary';lock_name='gpu-eval-0-secondary.lock'
    lock = (RUNTIME / lock_name).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    health = json.load(urlopen(model_url.rsplit('/', 1)[0]+'/health', timeout=8))
    assert health['ready'] and health['model'] == model_id
    model_pid = int(subprocess.check_output(['systemctl', '--user', 'show', model_unit, '-p', 'MainPID', '--value']))
    model_cmd = Path(f'/proc/{model_pid}/cmdline').read_bytes().decode().split('\0')
    assert model_id in model_cmd and model_url.split(':')[-1].split('/')[0] in model_cmd
    free_mib = int(subprocess.check_output(['nvidia-smi', '-i', str(simulation_gpu),
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
    assert free_mib >= 7000, 'Insufficient simulation GPU memory'
    if os.environ.get('UMI_EVAL_PARALLEL_PROTECTED') == '1':
        gpu_pids = {int(p) for p in subprocess.check_output(['nvidia-smi', '-i', str(simulation_gpu),
            '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()}
        allowed_pids={model_pid}
        verified_units=[]
        authorized={
            'umi-intersection-pi05-30000-4090-console.service':('pi05-cup-intersection-30000','intersection_server.py'),
            'umi-fast-full-30000-20261010-r2.service':('fast_housing_pi05_30000_full_20261010','full_task_sequential.py'),
            'umi-fast-intersection-pi05-10000-20261010.service':('pi05-cup-intersection-10000','intersection_step_server.py'),
            'umi-fast-intersection-pi05-20000-20261010.service':('pi05-cup-intersection-20000','intersection_step_server.py'),
            'umi-fast-intersection-full-10000-20261010.service':('fast_housing_pi05_intersection_10000_full_20261010','full_task_sequential.py'),
            'umi-fast-intersection-full-20000-20261010.service':('fast_housing_pi05_intersection_20000_full_20261010','full_task_sequential.py'),
        }
        for unit in filter(None,os.environ.get('UMI_EVAL_PARALLEL_ALLOWED_UNITS','').split(',')):
            assert unit in authorized, 'Unknown parallel service identity'
            pid=int(subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value']))
            if not pid:continue
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
            identity,entry=authorized[unit]
            assert identity in cmd and entry in cmd and ('/home/claude/workspace/umi_cup_intersection_models_4090_20261009/' in cmd or str(HERE) in cmd)
            allowed_pids.add(pid);verified_units.append({'unit':unit,'pid':pid})
        assert gpu_pids <= allowed_pids, 'Another compute process is using this evaluation GPU'
        (HERE/f'parallel_preflight_{model_id}.json').write_text(json.dumps({'allowed_units':verified_units,'gpu':simulation_gpu,'free_mib':free_mib,'observed_compute_pids':sorted(gpu_pids)},indent=2)+'\n')

    # Change only stage selection in this process. Save the actual main
    # function used for reproducibility; production run.py is not modified.
    source = inspect.getsource(run.main)
    old = "requested_task_stage = ('return_to_origin' if two_stage.plate_placed else 'place_on_plate')"
    assert source.count(old) == 1, 'Runner stage selector changed; review before deployment'
    source = source.replace(old, 'requested_task_stage = two_stage.requested_task_stage')
    snapshot = HERE / 'runner_main_snapshot.py'
    snapshot.write_text(source)
    exec(compile(source, str(snapshot), 'exec'), run.__dict__)
    shared = run_visual_aligned.configure()
    original_create, original_write, original_load = run.create_sim, run._write_json, run._load_callable
    original_evaluator = two_stage_task.CupPlateReturnEvaluator
    holder = {}
    started = time.monotonic()
    audit = {'classification': 'live_model_manipulation_with_explicit_sequential_homing',
        'model': model_id, 'inference_gpu': inference_gpu, 'simulation_gpu': simulation_gpu,
        'inference_pid': model_pid, 'simulation_slot':simulation_slot, 'left_extra_closure_rad': MARGIN,
        'physics_hz': 240, 'solver_iterations': 128, 'response_gain': float(os.environ.get('UMI_ONLINE_RESPONSE_GAIN', '6')), 'jaw_response_gain': float(os.environ.get('UMI_ONLINE_JAW_RESPONSE_GAIN', '4')),
        'velocity_rad_s': float(os.environ.get('UMI_ONLINE_VELOCITY_RAD_S', '1.2')),
        'acceleration_rad_s2': float(os.environ.get('UMI_ONLINE_ACCELERATION_RAD_S2', '2.4')),
        'training_prompt': ('Place the cup on the plate, then put it back to its original position' if 'cup-clean' in model_id else 'official stage instructions'),
        'model_policy_wall_s': 0., 'world_step_wall_s': 0., 'world_step_calls': 0, 'failure_reason': None,
        'model_request_hz': 10, 'action_execution_hz': 30, 'full_task_reset': True,
        'phase_reset': False, 'step_limit': None, 'stop_on_full_task_success': True,
        'stages': ['right_place_model', 'right_release_controller', 'right_lift_controller',
            'right_home_controller', 'left_return_model'],
        'original_placement_policy_preserved': True,
        'left_held_at_placement_end_during_right_homing': True,
        'right_held_at_initial_joints_during_left': True, 'cup_state_used_for_grasp_control': False,
        'home_gate': {'position_m': .005, 'orientation_deg': 2., 'joint_rad': .02,
            'joint_velocity_rad_s': .05, 'gripper_open_fraction': .95, 'stable_s': .5},
        'right_retreat_vertical_m': .12, 'servo_actions': 0, 'model_requests': 0,
        'max_applied_closure_rad': 0.,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), HERE/'sequencing.py', HERE/'jaw_margin.py', snapshot)}}

    class SequentialEvaluator(original_evaluator):
        def __init__(self, observation):
            super().__init__(observation)
            holder['seq'] = Sequencer(observation)
            holder['observation'] = observation

        @property
        def requested_task_stage(self):
            return holder['seq'].requested_task_stage

        def update(self, observation, *, plate_success):
            result = super().update(observation, plate_success=plate_success)
            seq = holder['seq']
            if self.plate_placed:
                seq.placed(observation)
            result['full_task_success'] &= seq.right_home_verified and seq.first_left_policy_step is not None
            result['stage'] = 'complete' if result['full_task_success'] else seq.requested_task_stage
            result['sequential_handoff'] = seq.audit()
            return result

    def load(path, name):
        policy = original_load(path, name)
        if name != 'predict':
            return policy

        def predict(observation, step, episode):
            seq = holder['seq']
            seq.last_policy_phase = seq.phase
            if seq.phase in ('right_place', 'left_return'):
                actual = {**observation, 'task_stage': seq.requested_task_stage}
                prediction_start = time.monotonic()
                chunk = policy(actual, step, episode)
                audit['model_policy_wall_s'] += time.monotonic()-prediction_start
                audit['model_requests'] += 1
                if seq.phase == 'left_return':
                    for waypoint in chunk['waypoints']:
                        waypoint['right'] = {**copy.deepcopy(seq.home['right']['tool_pose']), 'gripper_open_fraction': 1.}
                if seq.phase == 'left_return':
                    assert seq.right_home_verified
                    if seq.first_left_policy_step is None:
                        seq.first_left_policy_step = step
            else:
                chunk = seq.policy_waypoints(observation)
            with (OUTPUT/'handoff.jsonl').open('a') as stream:
                stream.write(json.dumps({'policy_step': step, 'physics_time_s': observation['physics_time_s'],
                    'source': 'model' if seq.phase in ('right_place', 'left_return') else 'home_controller',
                    'requested_task_stage': seq.requested_task_stage, **seq.audit()})+'\n')
            return chunk
        return predict

    def create(*args, **kwargs):
        app, env = original_create(*args, **kwargs)
        assert env.cup_physics_profile['physics_hz'] == 240
        assert env.cup_physics_profile['solver_position_iterations'] == 128
        assert abs(env.q_closed_rad + .1) < 1e-9 and abs(env.q_open_rad - .7) < 1e-9
        holder['env'] = env
        original_world_step = env.world.step
        def timed_world_step(*args, **kwargs):
            t = time.monotonic()
            result = original_world_step(*args, **kwargs)
            audit['world_step_wall_s'] += time.monotonic()-t
            audit['world_step_calls'] += 1
            return result
        env.world.step = timed_world_step
        original_step = env.step
        margin_stream = (OUTPUT/'left_jaw_margin.jsonl').open('x', buffering=1)
        sequence_stream = (OUTPUT/'sequencing_actions.jsonl').open('x', buffering=1)

        def step(action, *args, **kwargs):
            seq = holder['seq']
            before = holder['observation']
            phase = seq.phase
            command = seq.filter_action(action, before)
            if phase == 'right_place':
                # The trajectory executor already applied the same original
                # inter-arm guard. Leave the proven placement commands intact.
                guard = {'active': False, 'source': 'original_trajectory_executor_guard'}
            else:
                command, guard = limit_interarm_motion(before, command)
            adjusted, margin = adjust(command, MARGIN, env.q_closed_rad, env.q_open_rad)
            outcome = original_step(adjusted, *args, **kwargs)
            state = outcome[0]
            holder['observation'] = state
            seq.observe(state)
            cup = state['objects']['cup']
            # A cup fallen below the table cannot be recovered by the tabletop
            # task. There is no policy-step or wall-clock limit.
            if cup['position_m'][2] < .65:
                audit['failure_reason'] = 'cup_fell_below_table'
                Path(os.environ['UMI_EVAL_STOP_FILE']).touch()
            audit['servo_actions'] += 1
            audit['max_applied_closure_rad'] = max(audit['max_applied_closure_rad'], margin['applied_closure_margin_rad'])
            index = audit['servo_actions']-1
            margin_stream.write(json.dumps({'servo_step': index, 'physics_time_s': state['physics_time_s'],
                **margin, 'left_actual_jaw_rad': state['robots']['left']['gripper_joint_position_rad'], 'cup': cup})+'\n')
            sequence_stream.write(json.dumps({'servo_step': index, 'physics_time_s': state['physics_time_s'],
                'applied_phase': phase, 'action': adjusted, 'guard': guard, **seq.audit()})+'\n')
            if audit['servo_actions'] % 30 == 0:
                progress = {'servo_actions': audit['servo_actions'], 'physics_time_s': state['physics_time_s'],
                    'wall_s': time.monotonic()-started, 'model_requests': audit['model_requests'],
                    'model': model_id, 'model_policy_wall_s': audit['model_policy_wall_s'],
                    'world_step_wall_s': audit['world_step_wall_s'],
                    'shared_render_stats': dict(shared.stats) if shared is not None else {},
                    'cup_position_m': cup['position_m'], 'cup_quaternion_wxyz': cup['quaternion_wxyz'],
                    'max_shape_change_mm': 1000*cup.get('deformation', {}).get('max_nodal_shape_change_m', 0),
                    'left_jaw_margin': margin, **seq.audit()}
                temp = OUTPUT/'progress.tmp'
                temp.write_text(json.dumps(progress)+'\n')
                temp.replace(OUTPUT/'progress.json')
                if audit['servo_actions'] % 150 == 0:
                    print(json.dumps(progress), flush=True)
            return outcome
        env.step = step
        return app, env

    def write(path, value):
        if Path(path).name in ('manifest.json', 'report.json'):
            details = dict(audit)
            details['wall_s'] = time.monotonic()-started
            details['shared_render_stats'] = dict(shared.stats) if shared is not None else {}
            if 'seq' in holder:
                details.update(holder['seq'].audit())
                details['phase_events'] = holder['seq'].events
            value['sequential_homing_intervention'] = details
        original_write(path, value)

    two_stage_task.CupPlateReturnEvaluator = SequentialEvaluator
    run.create_sim, run._write_json, run._load_callable = create, write, load
    try:
        return run.main()
    finally:
        if shared is not None:
            shared.close()
        if 'env' in holder:
            stream = getattr(holder['env'], '_continuous_audit_stream', None)
            if stream is not None: stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
