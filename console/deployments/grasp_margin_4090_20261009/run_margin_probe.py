"""Independent entry: existing tuned control/visuals plus read-only friction data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
TUNED = Path('/home/claude/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1')
sys.path.insert(0, str(TUNED))
from margin_core import closure_target, decode_friction
from yubi_isaac_sim_env import run as runner
from yubi_isaac_sim_env import run_visual_aligned as visual_runner


def main():
    extra = float(os.environ['UMI_MARGIN_EXTRA_FRACTION'])
    closure_target(.5, extra)
    output = Path(os.environ['SIM_ADAPTER_AUDIT_DIR'])
    original_create, original_write = runner.create_sim, runner._write_json
    provenance = {
        'classification': 'frozen_model_action_physics_comparison_not_online_inference',
        'source_run': 'c3b23931ada5', 'source_sha256': os.environ['UMI_MARGIN_SOURCE_SHA256'],
        'max_right_extra_closure_fraction': extra, 'full_stroke_rad': .8,
        'maximum_extra_joint_closure_rad': extra*.8,
        'right_jaw_fade_fraction': [.55, .65], 'arm_targets_changed': False,
        'left_jaw_changed': False, 'physics_materials_changed': False,
        'drive_force_limit_changed': False, 'model_weights_changed': False,
        'object_feedback_in_actions': False, 'real_aperture_calibration_measured': False,
        'friction_api': 'RigidPrim.get_friction_data(dt=1/physics_hz)',
        'normal_force_abort_N_per_finger': 40., 'normal_force_abort_consecutive_samples': 3,
        'does_not_claim_pure_model_or_complete_task_success': True,
        'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [Path(__file__), HERE/'margin_core.py', HERE/'replay_margin.py']}}

    def write(path, value):
        if Path(path).name in ('report.json', 'manifest.json'):
            value['grasp_margin_diagnostic'] = provenance
            value['evaluation_class'] = provenance['classification']
            value['model_inference_requests'] = 0
        return original_write(path, value)

    def create(*args, **kwargs):
        app, env = original_create(*args, **kwargs)
        snapshot = env.contact_audit_snapshot
        normal_streak = 0
        last_resource_check = 0.

        def audit():
            nonlocal normal_streak, last_resource_check
            row = snapshot()
            dt = 1/env.task_config['physics_hz']
            maximum = 0.
            for side in ('left', 'right'):
                for part in ('left_finger', 'right_finger', 'base'):
                    contact = row['robots'][side]['cup_contacts'][part]
                    contact['friction'] = decode_friction(
                        env.robot_links[side][part].get_friction_data(clone=False, dt=dt))
                    if part != 'base':
                        maximum = max(maximum, contact['positive_normal_force_sum_N'])
            normal_streak = normal_streak+1 if maximum > 40 else 0
            reason = 'sustained_normal_force_limit' if normal_streak >= 3 else None
            if time.monotonic()-last_resource_check > 10:
                last_resource_check = time.monotonic()
                pids = subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid',
                    '--format=csv,noheader'], text=True, timeout=5).split()
                if any(p != str(os.getpid()) for p in pids):
                    reason = 'other_compute_process_on_simulation_gpu'
            if reason:
                output.mkdir(parents=True, exist_ok=True)
                with (output/'safety_abort.json').open('x') as stream:
                    json.dump({'reason': reason, 'action_index': row['action_index'],
                               'maximum_finger_normal_force_N': maximum, 'row': row}, stream)
                raise RuntimeError('Diagnostic safety stop: '+reason)
            return row

        env.contact_audit_snapshot = audit
        return app, env

    runner.create_sim, runner._write_json = create, write
    return visual_runner.main()


if __name__ == '__main__':
    raise SystemExit(main())
