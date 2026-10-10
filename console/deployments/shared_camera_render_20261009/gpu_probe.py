"""Same-state RTX A/B check; no model inference or task-success claim."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from yubi_isaac_sim_env import run, run_visual_aligned


class Video:
    def __init__(self):
        self.last = None
    def write(self, pixels):
        self.last = np.array(pixels, copy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, exist_ok=False)
    lock = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/console.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    processes = subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid',
        '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()
    assert not processes, 'GPU0 is occupied'
    os.environ['SIM_ADAPTER_AUDIT_DIR'] = str(args.output)
    shared = run_visual_aligned.configure()
    pkg = run.PACKAGE_DIR
    config = json.loads((pkg/'config.json').read_text())
    config['physics_hz'] = 240
    config['policy_hz'] = 30
    app = env = None
    result = {'status':'running', 'classification':'same_state_render_performance_and_alignment_probe'}
    calls = 0
    try:
        app, env = run.create_sim(scene='dual_franka_yubi_official_fingertip_friction_trial',
            gui=False, setup='online_aligned_v2', seed=42, task_config=config,
            head_camera_calibration=pkg/'head_camera_online_aligned_v2.json')
        assert env.cup_physics_profile['solver_position_iterations'] == 128
        assert env.cup_physics_profile['physics_hz'] == 240
        result['cup_physics_profile'] = env.cup_physics_profile
        result['success_config'] = config['success']
        original_render = env.world.render
        def counted_render():
            nonlocal calls
            calls += 1
            return original_render()
        env.world.render = counted_render
        specs = {n:run._camera_spec(n,pkg/'head_camera_online_aligned_v2.json')
                 for n in ('head','left_wrist','right_wrist')}
        cameras = {n:run._make_camera(spec,gui=False,recording_video=True) for n,spec in specs.items()}
        state = env.observe()
        shared.original_policy(state,cameras,specs,env.world)  # Exclude initialization warmup.
        base_wall = opt_wall = 0.
        base_calls = opt_calls = 0
        comparisons = []
        videos = {n:Video() for n in cameras}
        def sample(function, index):
            function(videos['head'],None,cameras['head'],env.world,specs['head'],
                     0,index,'step',index,state,{n:videos[n] for n in ('left_wrist','right_wrist')},
                     cameras,specs)
        for index in range(9):
            # Small real articulation motion tests frame invalidation at changing poses.
            targets = env.current_targets.clone()
            targets[:,env.arm_dof_indices[0]] += .003
            env.robots.apply_action(env.ArticulationActions(
                joint_positions=targets[:,env.command_dof_indices],joint_indices=env.command_dof_indices))
            for _ in range(8):
                env.world.step(render=False)
            state = env.observe()
            t = time.perf_counter(); count = calls
            sample(shared.original_sample,index)
            baseline = {n:videos[n].last.copy() for n in cameras}
            base_wall += time.perf_counter()-t; base_calls += calls-count
            t = time.perf_counter(); count = calls
            sample(run._sample,index)
            opt_wall += time.perf_counter()-t; opt_calls += calls-count
            for name in cameras:
                diff = np.abs(baseline[name].astype(float)-videos[name].last.astype(float))
                comparisons.append({'sample':index,'view':name,'physics_time_s':state['physics_time_s'],
                    'mean_absolute_pixel_difference':float(diff.mean()),
                    'p99_absolute_pixel_difference':float(np.percentile(diff,99))})
            if index % 3 == 2:
                t = time.perf_counter(); count = calls
                baseline_obs = shared.original_policy(state,cameras,specs,env.world)
                base_wall += time.perf_counter()-t; base_calls += calls-count
                t = time.perf_counter(); count = calls
                optimized_obs = run._policy_observation(state,cameras,specs,env.world)
                opt_wall += time.perf_counter()-t; opt_calls += calls-count
                for name in cameras:
                    meta = optimized_obs['image_metadata'][name]
                    assert meta['shared_render_reused'] is True
                    assert abs(meta['rendering_time_s']-state['physics_time_s']) < 1e-5
                    assert np.array_equal(optimized_obs['images'][name],videos[name].last)
                    pose = cameras[name].get_world_pose(camera_axes='usd')
                    np.testing.assert_allclose(meta['pose_world']['position_m'],run._cpu_array(pose[0]),atol=1e-7)
                assert len({m['shared_render_group'] for m in optimized_obs['image_metadata'].values()}) == 1
        assert base_calls >= 216 and opt_calls == 54, (base_calls,opt_calls)
        # RTX temporal accumulation can vary; inspect bounded pixel differences.
        assert max(c['mean_absolute_pixel_difference'] for c in comparisons) < 2., comparisons
        result.update(status='passed', baseline_render_calls=base_calls,
            optimized_render_calls=opt_calls, baseline_capture_and_policy_wall_s=base_wall,
            optimized_capture_and_policy_wall_s=opt_wall, measured_render_work_speedup=base_wall/opt_wall,
            render_calls_reduction_fraction=1-opt_calls/base_calls,
            comparisons=comparisons, shared_stats=dict(shared.stats),
            sampled_rgb_exactly_reused_by_model=True,
            rendering_timestamps_match_physics=True, current_camera_poses_verified=True,
            actual_physics_seconds=env.world.current_time)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        shared.close()
        if app is not None:
            app.close()
        lock.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','cup_physics_profile')}),flush=True)


if __name__ == '__main__':
    main()
