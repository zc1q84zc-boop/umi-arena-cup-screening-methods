"""Deploy the verified opt-in governor and PI0.5 intersection launch setting.

Checks original source hashes and idle simulation GPU, preserving unrelated
models' gain 4 setting and all motion/physics/camera parameters.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile

TASK = Path(__file__).resolve().parent
CONSOLE = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009')
PKG = CONSOLE / 'simulator_profiles/tuned_v1/yubi_isaac_sim_env'
MODELS = Path('/home/claude/workspace/umi_cup_intersection_models_4090_20261009')
EXPECTED = {
    PKG / 'continuous_targets.py': 'ffaff995d2bbd9a6bce1e395aa4483996d7b434ea5e390e1ed32bcb821699c02',
    PKG / 'env.py': '79b13df5470b24112da5d3d1549ba3df96c98fa098c3a98f99371fd2d5ce562b',
    MODELS / 'scripts/run_tuned_intersection_4090.sh': '58784e92093a0dca11d7aaae2620f053a34ae152c91228ace73f85d27e1b2012',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace(path, data, mode):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temp, mode)
    os.replace(temp, path)


def main():
    selection=json.loads((TASK / 'final_parameter_choice.json').read_text())
    assert (selection['selected_default_arm_gain'], selection['selected_default_jaw_gain']) == (8,4), 'Mixed gain not accepted as default'
    lock = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy/console.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output([
        'nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not (TASK / 'mixed_response_gain_deployment.json').exists()
    targets = {}
    for path, expected in EXPECTED.items():
        old = path.read_bytes()
        assert sha(old) == expected, f'Concurrent source update: {path}'
        new = (TASK / 'mixed_production_candidate' / path.name).read_bytes()
        if path.suffix == '.py':
            compile(new, str(path), 'exec')
        else:
            subprocess.run(['bash', '-n', str(TASK / 'mixed_production_candidate' / path.name)], check=True)
        targets[path] = [old, new, path.stat().st_mode & 0o777]
    launcher = MODELS / 'scripts/run_tuned_intersection_4090.sh'
    launcher_hash = sha(targets[launcher][1])
    for family in ('pi05', 'openwam'):
        path = CONSOLE / f'deployment_validation/intersection_{family}_server.json'
        if not path.exists():
            # An unregistered model stays unregistered; create no server pass.
            assert family == 'openwam'
            continue
        old = path.read_bytes()
        value = json.loads(old)
        assert value['status'] == 'ok'
        assert value['simulation_launcher_sha256'] == EXPECTED[launcher]
        value['simulation_launcher_sha256'] = launcher_hash
        value['simulation_launcher_update_20261010'] = dict(
            scope='PI0.5 arm governor gain8 and jaw gain4; other models retain gain4; server/checkpoint unchanged',
            evidence=str(TASK / 'mixed_response_gain_deployment.json'))
        targets[path] = [old, (json.dumps(value, indent=2) + '\n').encode(), path.stat().st_mode & 0o777]
    backup = TASK / 'production_before_mixed_deploy'
    backup.mkdir(mode=0o700)
    for path, (old, _, mode) in targets.items():
        out = backup / path.name
        out.write_bytes(old)
        os.chmod(out, mode)
    changed = []
    try:
        for path, (old, new, mode) in targets.items():
            assert path.read_bytes() == old, f'Concurrent source update: {path}'
            replace(path, new, mode)
            changed.append(path)
    except BaseException:
        for path in reversed(changed):
            old, new, mode = targets[path]
            assert path.read_bytes() == new, f'Refusing to overwrite concurrent source change: {path}'
            replace(path, old, mode)
        raise
    report = dict(status='deployed', response_gain_pi05_intersection=8, jaw_response_gain_pi05_intersection=4,
                  response_gain_other_policies=4, physics_hz=240, solver_iterations=128,
                  velocity_rad_s=.8, acceleration_rad_s2=1.5,
                  backup=str(backup), server_and_checkpoint_changed=False,
                  sources=[dict(path=str(p), before_sha256=sha(old), after_sha256=sha(new))
                           for p, (old, new, _) in targets.items()])
    (TASK / 'mixed_response_gain_deployment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    lock.close()


if __name__ == '__main__':
    main()
