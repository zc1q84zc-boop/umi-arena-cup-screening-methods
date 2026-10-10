#!/usr/bin/env python3
"""Loopback-only console for Isaac Sim runs on the RTX 5090 host.

This is deliberately separate from the read-only practice replay server. Only
policies in sim_policy_registry.json can be launched; checkpoint entries without
validated online adapters are visible but cannot be executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "sim_runs"
REGISTRY = ROOT / "sim_policy_registry.json"
REMOTE_HOST = os.environ.get("UMI_SIM_HOST", "squirrel_5090")
REMOTE_ROOT = Path(os.environ.get("UMI_SIM_REMOTE_ROOT", "/home/lrl/dual-franka-yubi-isaac-sim-deploy"))
TUNED_ROOT = Path('/home/lrl/dual-franka-yubi-isaac-sim-console-tuned-v1')
TUNED_REPLAY_ID = 'tuned-paired-replay-259632-259633'
GPU_INDEX = 0
OFFICIAL_CUP_EPISODES = frozenset((61164, 61165, 136238, 136239, 231149,
                                  231150, 259632, 259633, 262232, 262233))
_PVC_REGISTRY = runpy.run_path(str(ROOT/'simulator_profiles/tuned_v1/yubi_isaac_sim_env/pvc_stiffness.py'))
_PVC_NUMERICS = runpy.run_path(str(ROOT/'simulator_profiles/tuned_v1/yubi_isaac_sim_env/pvc_numerics.py'))
SHELL_PROFILE_IDS = (*_PVC_REGISTRY['SHELL_PROFILE_IDS'], _PVC_NUMERICS['PRECISION_ID'])
DEFAULT_CONTACT_PROFILE = 'baseline'
DEFAULT_POLICY_ID = None
DEFAULT_RUN_MODE = 'fixed'


def _require_verified_pvc_probe(profile_id='pvc_elastic_shell_v1') -> dict:
    """Fail closed while the physical shell test is pending or code changed."""
    try:
        assert profile_id in SHELL_PROFILE_IDS
        precision = profile_id == _PVC_NUMERICS['PRECISION_ID']
        sweep = profile_id != 'pvc_elastic_shell_v1'
        marker = (ROOT/'sim_validation/pvc_stiffness'/profile_id/'verified_probe.json'
                  if sweep else ROOT/'sim_validation/pvc_shell_verified_probe.json')
        report = json.loads(marker.read_text())
        package = ROOT/'simulator_profiles/tuned_v1/yubi_isaac_sim_env'
        assert report['status'] == 'completed'
        assert report['purpose'] == 'physical_platen_compression_not_model_grasp'
        assert report['prescribed_vertex_animation'] is False
        assert report['profile']['id'] == profile_id and report['profile']['measured'] is False
        assert all(report[key] is True for key in ('unloaded_shape_preserved',
                   'physical_deformation_observed', 'recovered_after_release',
                   'platen_motion_verified'))
        assert report['sample_count'] == (1920 if precision else 960) and report['video_frames'] == 240
        assert report['shell_sha256'] == hashlib.sha256((package/'pvc_shell.py').read_bytes()).hexdigest()
        probe = 'pvc_precision_probe.py' if precision else ('pvc_stiffness_probe.py' if sweep else 'pvc_shell_probe.py')
        assert report['source_sha256'] == hashlib.sha256((package/probe).read_bytes()).hexdigest()
        if sweep:
            assert report['registry_sha256'] == hashlib.sha256((package/'pvc_stiffness.py').read_bytes()).hexdigest()
            base = runpy.run_path(str(package/'pvc_shell.py'))['PROFILE']
            expected = (_PVC_NUMERICS['precision_profile'](_PVC_REGISTRY['profile_for']('pvc_shell_e3000mpa_v1', base))
                        if precision else _PVC_REGISTRY['profile_for'](profile_id, base))
            assert report['profile'] == expected
            if precision:
                assert report['physics_hz'] == 240
                for field, filename in (('numerics_sha256','pvc_numerics.py'), ('readback_source_sha256','pvc_response_probe.py')):
                    assert report[field] == hashlib.sha256((package/filename).read_bytes()).hexdigest()
                for field in ('material_before_reset', 'material_after_reset'):
                    readback = report[field]
                    assert readback['verified_composed_usd'] is True
                    assert readback['solver_position_iterations'] == 128
                    assert readback['values']['omniphysics:youngsModulus'] == 3e9
                    assert math.isclose(readback['values']['omniphysics:surfaceThickness'], .001, rel_tol=2e-6)
        return report
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        raise ValueError('PVC shell physical validation pending; keep baseline or official friction trial') from exc


def _tuned_visual_profile() -> dict:
    path = ROOT / 'simulator_profiles/tuned_v1/active_visual_profile.json'
    if path.is_file():
        return json.loads(path.read_text())
    return {'reference_light_intensity': 220,
            'finger_visual_material': 'tuned_textured_asset',
            'measured_hand_eye_calibration': False}


def _stiffness_catalog() -> list[dict]:
    profiles = []
    for key, modulus in (*_PVC_REGISTRY['STIFFNESS_SPECS'], (_PVC_NUMERICS['PRECISION_ID'], 3e9)):
        try:
            _require_verified_pvc_probe(key)
            ready = True
        except ValueError:
            ready = False
        label = ('3 GPa 高精度弹性杯 · 128迭代 / 240Hz（未实测）'
                 if key == _PVC_NUMERICS['PRECISION_ID'] else f'弹性薄壳 {modulus/1e9:g} GPa（未实测）')
        profiles.append(dict(id=key, youngs_modulus_Pa=modulus, ready=ready,
                             measured=False, label=label))
    return profiles


def _stable_lift_streak(rows: list[dict], initial_z: float) -> int:
    """Consecutive observed upright holds, not peak height during a lift."""
    longest = current = 0
    for row in rows:
        quaternion = row.get("cup_quaternion_wxyz")
        linear = row.get("cup_linear_velocity_m_s")
        angular = row.get("cup_angular_velocity_rad_s")
        if quaternion is None or linear is None or angular is None:
            current = 0  # Old traces lack the evidence needed for this gate.
            continue
        _, x, y, _ = quaternion
        upright = 1 - 2 * (x*x + y*y) >= math.cos(math.radians(15))
        held = (row.get("phase") == "hold"
                and row["cup_position_m"][2] - initial_z >= 0.05
                and upright
                and math.sqrt(sum(v*v for v in linear)) <= 0.05
                and math.sqrt(sum(v*v for v in angular)) <= 0.3
                and 0.1 < row["driven_jaw_rad"] < 0.55)
        current = current + 1 if held else 0
        longest = max(longest, current)
    return longest


INFERENCE_BACKENDS = {
    "a100": {
        "label": "A100 · 现有远端推理", "host": "openwam-a100", "gpu": 6,
        "unit": "pi05-online-a100.service", "port": 18781,
        "checkpoint": "/mnt/data/benyun/workspace/pi05_cup_clean_20260923/checkpoints/pi05_cup_clean_success/pi05_cup_clean_v1/30000",
        "adapter_url": "http://127.0.0.1:18782/infer",
        "bridge_port": 18782,
        "bridge_units": ("pi05-a100-forward.service", "pi05-squirrel-reverse.service"),
    },
    "pi05_10000_a100": {
        "label": "A100 · π0.5 10k 在线推理", "host": "openwam-a100", "gpu": 6,
        "unit": "pi05-sim-10000.service", "port": 18790,
        "checkpoint": "/mnt/data/benyun/workspace/pi05_cup_clean_20260923/checkpoints/pi05_cup_clean_success/pi05_cup_clean_v1/10000",
        "adapter_url": "http://127.0.0.1:18794/infer",
        "process_signature": "pi05_sim_server.py",
        "model_id": "pi05-cup-clean-10000",
        "bridge_port": 18794,
        "bridge_units": ("pi05-10000-a100-forward.service", "pi05-10000-squirrel-reverse.service"),
    },
    "pi05_20000_a100": {
        "label": "A100 · π0.5 20k 在线推理", "host": "openwam-a100", "gpu": 6,
        "unit": "pi05-sim-20000.service", "port": 18791,
        "checkpoint": "/mnt/data/benyun/workspace/pi05_cup_clean_20260923/checkpoints/pi05_cup_clean_success/pi05_cup_clean_v1/20000",
        "adapter_url": "http://127.0.0.1:18795/infer",
        "process_signature": "pi05_sim_server.py",
        "model_id": "pi05-cup-clean-20000",
        "bridge_port": 18795,
        "bridge_units": ("pi05-20000-a100-forward.service", "pi05-20000-squirrel-reverse.service"),
    },
    "rtx5090": {
        "label": "squirrel RTX 5090 · π0.5 30k", "host": REMOTE_HOST, "gpu": 0,
        "unit": "umi-squirrel-pi05-30000-console.service", "port": 18783,
        "checkpoint": "/home/lrl/workspace/umi_cup_models_5090_20260928/pi05/30000/inference_export",
        "adapter_url": "http://127.0.0.1:18783/infer",
        "launcher_script": "run_pi05_30000.sh", "model_id": "pi05-cup-clean-30000",
    },
    "pi05_10000_rtx5090": {
        "label": "squirrel RTX 5090 · π0.5 10k", "host": REMOTE_HOST, "gpu": 0,
        "unit": "umi-squirrel-pi05-10000-console.service", "port": 18784,
        "checkpoint": "/home/lrl/workspace/umi_cup_models_5090_20260928/pi05/10000/inference_export",
        "adapter_url": "http://127.0.0.1:18784/infer",
        "launcher_script": "run_pi05_10000.sh", "model_id": "pi05-cup-clean-10000",
    },
    "pi05_20000_rtx5090": {
        "label": "squirrel RTX 5090 · π0.5 20k", "host": REMOTE_HOST, "gpu": 0,
        "unit": "umi-squirrel-pi05-20000-console.service", "port": 18785,
        "checkpoint": "/home/lrl/workspace/umi_cup_models_5090_20260928/pi05/20000/inference_export",
        "adapter_url": "http://127.0.0.1:18785/infer",
        "launcher_script": "run_pi05_20000.sh", "model_id": "pi05-cup-clean-20000",
    },
    "lingbot_a100": {
        "label": "A100 · LingBot VLA2 5k 在线推理", "host": "openwam-a100", "gpu": 7,
        "unit": "lingbot-sim-5000.service", "port": 18784,
        "checkpoint": "/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/lingbot/checkpoints/checkpoints/global_step_5000/hf_ckpt",
        "adapter_url": "http://127.0.0.1:18786/infer",
        "process_signature": "lingbot_sim_server.py",
        "model_id": "lingbot-cup-clean-5000",
        "bridge_port": 18786,
        "bridge_units": ("lingbot-a100-forward.service", "lingbot-squirrel-reverse.service"),
    },
    "lingbot_5000_rtx5090": {
        "label": "squirrel RTX 5090 · LingBot VLA2 5k", "host": REMOTE_HOST, "gpu": 0,
        "unit": "lingbot-5000-squirrel.service", "port": 18810,
        "checkpoint": "/home/lrl/workspace/umi_cup_models_5090_20260928/lingbot/5000/hf_ckpt",
        "adapter_url": "http://127.0.0.1:18810/infer",
        "launcher_script": "run_lingbot_5000.sh", "process_signature": "lingbot_sim_server.py",
        "model_id": "lingbot-cup-clean-5000",
    },
    "lingbot_10000_a100": {
        "label": "A100 · LingBot VLA2 10k 在线推理", "host": "openwam-a100", "gpu": 7,
        "unit": "lingbot-sim-10000.service", "port": 18802,
        "checkpoint": "/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/lingbot/checkpoints/checkpoints/global_step_10000/hf_ckpt",
        "adapter_url": "http://127.0.0.1:18804/infer",
        "process_signature": "lingbot_sim_server.py",
        "model_id": "lingbot-cup-clean-10000",
        "bridge_port": 18804,
        "bridge_units": ("lingbot-10000-a100-forward.service", "lingbot-10000-squirrel-reverse.service"),
    },
    "lingbot_10000_rtx5090": {
        "label": "squirrel RTX 5090 · LingBot VLA2 10k", "host": REMOTE_HOST, "gpu": 0,
        "unit": "lingbot-10000-squirrel.service", "port": 18811,
        "checkpoint": "/home/lrl/workspace/umi_cup_models_5090_20260928/lingbot/10000/hf_ckpt",
        "adapter_url": "http://127.0.0.1:18811/infer",
        "launcher_script": "run_lingbot_10000.sh", "process_signature": "lingbot_sim_server.py",
        "model_id": "lingbot-cup-clean-10000",
    },
    "lingbot_official_rtx5090": {
        "label": "squirrel RTX 5090 · LingBot VLA2 官方预训练版", "host": REMOTE_HOST, "gpu": 0,
        "unit": "lingbot-official-squirrel.service", "port": 18814,
        "checkpoint": "/home/lrl/workspace/umi_cup_models_5090_20260928/lingbot/official/hf_ckpt",
        "adapter_url": "http://127.0.0.1:18814/infer",
        "launcher_script": "run_lingbot_official.sh", "process_signature": "lingbot_sim_server.py",
        "model_id": "lingbot-vla2-official-pretrained",
    },
    "openwam_a100": {
        "label": "A100 · OpenWAM Alpha 10k 在线推理", "host": "openwam-a100", "gpu": 5,
        "unit": "openwam-sim-10000.service", "port": 18787,
        "checkpoint": "/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/openwam/checkpoints/2026-09-25_18-55-32",
        "adapter_url": "http://127.0.0.1:18789/infer",
        "process_signature": "openwam_sim_server.py",
        "checkpoint_name": "checkpoint_step_10000.safetensors",
        "model_id": "openwam-cup-clean-10000",
        "bridge_port": 18789,
        "bridge_units": ("openwam-a100-forward.service", "openwam-squirrel-reverse.service"),
    },
    "openwam_5000_a100": {
        "label": "A100 · OpenWAM Alpha 5k 在线推理", "host": "openwam-a100", "gpu": 5,
        "unit": "openwam-sim-5000.service", "port": 18796,
        "checkpoint": "/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/openwam/checkpoints/2026-09-25_18-55-32",
        "adapter_url": "http://127.0.0.1:18798/infer",
        "process_signature": "openwam_sim_server.py",
        "checkpoint_name": "checkpoint_step_5000.safetensors",
        "model_id": "openwam-cup-clean-5000",
        "bridge_port": 18798,
        "bridge_units": ("openwam-5000-a100-forward.service", "openwam-5000-squirrel-reverse.service"),
    },
    "openwam_5000_rtx5090": {
        "label": "squirrel RTX 5090 · OpenWAM Alpha 5k", "host": REMOTE_HOST, "gpu": 0,
        "unit": "openwam-5000-squirrel.service", "port": 18812,
        "checkpoint": "/home/lrl/workspace/umi_cup_models_5090_20260928/openwam/5000",
        "checkpoint_name": "checkpoint_step_5000.safetensors",
        "adapter_url": "http://127.0.0.1:18812/infer",
        "launcher_script": "run_openwam_5000.sh", "process_signature": "openwam_sim_server.py",
        "model_id": "openwam-cup-clean-5000",
    },
    "openwam_10000_rtx5090": {
        "label": "squirrel RTX 5090 · OpenWAM Alpha 10k", "host": REMOTE_HOST, "gpu": 0,
        "unit": "openwam-10000-squirrel.service", "port": 18813,
        "checkpoint": "/home/lrl/workspace/umi_cup_models_5090_20260928/openwam/10000",
        "checkpoint_name": "checkpoint_step_10000.safetensors",
        "adapter_url": "http://127.0.0.1:18813/infer",
        "launcher_script": "run_openwam_10000.sh", "process_signature": "openwam_sim_server.py",
        "model_id": "openwam-cup-clean-10000",
    },
}
INFERENCE_BACKENDS.update(json.loads(
    (ROOT / "deployments/intersection_20261009/backends.json").read_text()))

RUN_ID = re.compile(r"^[0-9a-f]{12}$")
CAMERAS = frozenset(("head", "overview", "left_wrist", "right_wrist"))
# A private, short-lived transport avoids seven independent handshakes over
# a high-latency Tailscale relay. No login/configuration or remote permissions
# change; the master expires 30 seconds after its last channel closes.
SSH_CONTROL_PATH = f"/run/user/{os.getuid()}/umi-simulation-console-%C"
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
               "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=1",
               "-o", "ControlMaster=auto", "-o", "ControlPersist=30",
               "-o", f"ControlPath={SSH_CONTROL_PATH}")
SSH = ("ssh", *SSH_OPTIONS)
SCP = ("scp", "-B", *SSH_OPTIONS)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _int_field(value, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


class SimulationRunner:
    def __init__(self, runs_dir: Path = RUNS):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.active_id: str | None = None
        self._catalog_lock = threading.Lock()
        self._catalog_cache: dict | None = None
        self._catalog_at = 0.0

    def catalog(self) -> dict:
        """Coalesce browser polling so it cannot flood the SSH jump host."""
        if self._catalog_cache is not None and time.monotonic() - self._catalog_at < 90:
            return self._catalog_cache
        if not self._catalog_lock.acquire(blocking=False):
            if self._catalog_cache is not None:
                return self._catalog_cache
            raise ValueError("catalog refresh in progress")
        try:
            if self._catalog_cache is not None and time.monotonic() - self._catalog_at < 90:
                return self._catalog_cache
            snapshot = {
                "repository": "StevenLiudw/dual-franka-yubi-isaac-sim",
                "host": "squirrel_5090", "gpu": GPU_INDEX,
                "gpu_status": self.gpu_status(),
                "inference_backends": self.inference_backends(),
                "policies": self.policies(),
                "stiffness_profiles": _stiffness_catalog(),
                "default_contact_profile": DEFAULT_CONTACT_PROFILE,
                "default_policy_id": DEFAULT_POLICY_ID,
                "default_run_mode": DEFAULT_RUN_MODE,
                "continuous_contact_profiles": ['baseline', _PVC_NUMERICS['PRECISION_ID']],
            }
            self._catalog_cache = snapshot
            self._catalog_at = time.monotonic()
            return snapshot
        finally:
            self._catalog_lock.release()

    def _runtime_snapshot(self, run_id: str) -> dict | None:
        """Read one identity-checked durable runtime, never launch/stop it."""
        if not RUN_ID.fullmatch(run_id):
            raise ValueError('invalid recovery run ID')
        unit = f'umi-tuned-runtime-{run_id}.service'
        command = ('systemctl --user show ' + shlex.quote(unit) +
                   ' -p ActiveState -p MainPID -p ExecMainStatus -p ExecMainCode -p Result; '
                   'task_pid=$(systemctl --user show ' + shlex.quote(unit) + ' -p MainPID --value); '
                   'if [[ $task_pid =~ ^[1-9][0-9]*$ ]]; then '
                   'printf "CommandLine="; tr "\\0" " " < "/proc/$task_pid/cmdline"; printf "\\n"; fi')
        result = self._remote(REMOTE_HOST, command)
        if result.returncode:
            return None
        state = dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        if int(state.get('MainPID','0')):
            cmd = state.get('CommandLine','')
            expected = str(REMOTE_ROOT/'runs'/f'console_{run_id}')
            if '-m yubi_isaac_sim_env.run' not in cmd or '--record-run '+expected not in cmd:
                raise ValueError('durable runtime PID identity mismatch; not adopted')
        return state

    def recover_active_run(self) -> str | None:
        """Reattach to a pre-existing online run after console restart."""
        for path in sorted(self.runs_dir.glob('*/metadata.json'), reverse=True):
            metadata = json.loads(path.read_text())
            if (metadata.get('status') not in ('starting','running')
                    or metadata.get('simulator_profile') != 'tuned_online_v1'):
                continue
            run_id = metadata['id']
            if metadata.get('remote_dir') != str(REMOTE_ROOT/'runs'/f'console_{run_id}'):
                continue
            state = self._runtime_snapshot(run_id)
            if not state or state.get('ActiveState') != 'active' or not int(state.get('MainPID','0')):
                continue
            policy = next((p for p in self.policies() if p['id']==metadata['policy']), None)
            if not policy or not policy.get('online_inference'):
                continue
            with self.lock:
                if self.active_id is not None:
                    return self.active_id
                self.active_id = run_id
                metadata['console_tracking_recovered'] = True
                atomic_json(path, metadata)
            threading.Thread(target=self._execute, args=(metadata,policy,True),
                             name=f'recovered-isaac-{run_id}', daemon=True).start()
            return run_id
        return None

    def _wait_existing_runtime(self, run_id: str) -> subprocess.CompletedProcess:
        while True:
            state = self._runtime_snapshot(run_id)
            if state and not int(state.get('MainPID','0')) and state.get('ActiveState') in ('inactive','failed'):
                clean = (state.get('ExecMainStatus')=='0' and
                         ((state.get('ExecMainCode')=='1' and
                           (state['ActiveState']=='inactive' or state.get('Result')=='timeout'))
                          # The launcher resets a finished unit's failure
                          # state. This clears ExecMainCode but preserves a
                          # successful, inactive unit. The report and all
                          # artifacts are still checked after this wait.
                          or (state.get('ExecMainCode')=='0' and state['ActiveState']=='inactive'
                              and state.get('Result')=='success')))
                return subprocess.CompletedProcess(['recovered-runtime',run_id], 0 if clean else 1)
            # Ordinary ongoing execution or temporarily unavailable SSH is not
            # a reason to kill/restart the simulator or its inference server.
            time.sleep(10)

    def policies(self) -> list[dict]:
        entries = json.loads(REGISTRY.read_text())["policies"]
        registered = {entry["id"] for entry in entries}
        for checkpoint in json.loads((ROOT / "checkpoints.json").read_text())["checkpoints"]:
            if checkpoint["id"] not in registered:
                absolute_model = checkpoint["model"].lower() in ("openwam", "lingbot")
                entries.append({
                    "id": checkpoint["id"], "label": checkpoint["label"],
                    "kind": "checkpoint", "model": checkpoint["model"],
                    "ready": False,
                    "reason": ("需标定训练 VR 手根坐标到仿真 world/YUBI 工具端的变换，再验证在线相机及动作适配；目前只有离线 replay。"
                               if absolute_model else
                               "需要验证该 checkpoint 的在线观测、坐标变换与动作适配器；离线 replay 不能直接用于闭环仿真。"),
                })
        if len({entry["id"] for entry in entries}) != len(entries):
            raise ValueError("duplicate simulation policy ID")
        return entries

    def gpu_status(self) -> dict:
        result = subprocess.run(
            [*SSH, REMOTE_HOST,
             f"nvidia-smi -i {GPU_INDEX} --query-compute-apps=pid --format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode:
            return {"available": False, "reason": "RTX 5090 host is unreachable"}
        if result.stdout.strip():
            return {"available": False, "reason": "squirrel GPU 0 有其他计算进程"}
        return {"available": True, "reason": "squirrel GPU 0 空闲"}

    @staticmethod
    def _remote(host: str, command: str, timeout: int = 15) -> subprocess.CompletedProcess:
        return subprocess.run([*SSH, host, command], capture_output=True,
                              text=True, timeout=timeout, check=False)

    def _service_state(self, backend_id: str) -> dict:
        backend = INFERENCE_BACKENDS[backend_id]
        prefix = backend.get("systemctl_prefix", "")
        result = self._remote(backend["host"],
                              f"{prefix} systemctl --user show {backend['unit']} --property=ActiveState --property=MainPID --property=ExecStart --no-pager")
        if result.returncode:
            raise ValueError(f"{backend['label']} service status unavailable")
        state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        signature = backend.get("launcher_script", backend.get("process_signature", "pi05_online_server.py"))
        if (signature not in state.get("ExecStart", "")
                or (not backend.get("launcher_script") and backend["checkpoint"] not in state.get("ExecStart", ""))
                or (not backend.get("launcher_script") and backend.get("checkpoint_name")
                    and backend["checkpoint_name"] not in state.get("ExecStart", ""))
                or (not backend.get("launcher_script") and f"{backend['port']}" not in state.get("ExecStart", ""))):
            raise ValueError(f"{backend['label']} unit does not match the expected private checkpoint")
        return {"active": state.get("ActiveState") == "active", "pid": int(state.get("MainPID", "0"))}

    def backend_status(self, backend_id: str) -> dict:
        if backend_id == "orin":
            return {"label": "本机 Jetson Orin 推理", "active": False, "ready": False,
                    "deploy_ready": False,
                    "reason": "本机没有这份 30k 权重和经过验证的 Orin 推理环境；不会把 A100 代称为本机。"}
        backend = INFERENCE_BACKENDS[backend_id]
        try:
            state = self._service_state(backend_id)
            if not state["active"]:
                return {"label": backend["label"], "active": False, "ready": False,
                        "deploy_ready": True,
                        "reason": "推理服务已退出；启动仿真时将自动加载。"}
            health = self._remote(backend["host"],
                                  f"curl -fsS --max-time 4 http://127.0.0.1:{backend['port']}/health", timeout=10)
            ready = health.returncode == 0 and json.loads(health.stdout).get("model") == backend.get("model_id", "pi05-cup-clean-30000")
            if ready and backend.get("bridge_port"):
                bridge = self._remote(REMOTE_HOST,
                                      f"curl -fsS --max-time 4 http://127.0.0.1:{backend['bridge_port']}/health", timeout=10)
                ready = bridge.returncode == 0 and json.loads(bridge.stdout).get("model") == backend.get("model_id", "pi05-cup-clean-30000")
            return {"label": backend["label"], "active": True, "ready": bool(ready),
                    "deploy_ready": True,
                    "reason": "就绪" if ready else "服务正在加载或到仿真机的私有通道尚未就绪。"}
        except (ValueError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            return {"label": backend["label"], "active": False, "ready": False,
                    "deploy_ready": False,
                    "reason": str(exc)}

    def inference_backends(self) -> dict:
        # User requested inference and simulation exclusively on the LAN 5090.
        # Retain historical backend definitions for old reports, not launching.
        return {key: self.backend_status(key) for key, backend in INFERENCE_BACKENDS.items()
                if backend["host"] == REMOTE_HOST}

    @staticmethod
    def _manage_bridges(backend: dict, action: str) -> None:
        units = backend.get("bridge_units", ())
        if not units:
            return
        if action == "stop":
            units = tuple(reversed(units))
        result = subprocess.run(["systemctl", "--user", action, *units],
                                capture_output=True, text=True, timeout=15,
                                check=False)
        if result.returncode:
            raise ValueError(f"private inference bridge {action} failed: {result.stderr.strip()[:300]}")

    def _gpu_processes(self, backend: dict) -> set[int]:
        result = self._remote(backend["host"],
                              f"nvidia-smi -i {backend['gpu']} --query-compute-apps=pid --format=csv,noheader")
        if result.returncode:
            raise ValueError(f"{backend['label']} GPU {backend['gpu']} status unavailable")
        try:
            return {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
        except ValueError as exc:
            raise ValueError(f"{backend['label']} GPU process list is invalid") from exc

    def _verify_managed_pid(self, backend: dict, pid: int) -> None:
        if pid <= 0:
            raise ValueError("service PID missing; refusing to stop an unidentified process")
        command = self._remote(backend["host"], f"tr '\\0' ' ' </proc/{pid}/cmdline")
        if (command.returncode or backend.get("process_signature", "pi05_online_server.py") not in command.stdout
                or backend["checkpoint"] not in command.stdout
                or (backend.get("checkpoint_name") and backend["checkpoint_name"] not in command.stdout)):
            raise ValueError("service process identity mismatch; refusing to stop")

    def _stop_backend_locked(self, backend_id: str, state: dict) -> None:
        backend = INFERENCE_BACKENDS[backend_id]
        if state["active"]:
            self._verify_managed_pid(backend, state["pid"])
        prefix = backend.get("systemctl_prefix", "")
        result = self._remote(backend["host"],
                              f"{prefix} systemctl --user stop {backend['unit']}", timeout=20)
        if result.returncode:
            raise ValueError(f"unable to stop {backend['label']}: {result.stderr.strip()[:300]}")
        self._manage_bridges(backend, "stop")

    def _start_backend_locked(self, backend_id: str) -> None:
        backend = INFERENCE_BACKENDS[backend_id]
        if backend["host"] != REMOTE_HOST:
            raise ValueError("Only squirrel_5090 inference is enabled; A100 and other hosts are disabled")
        target = self._service_state(backend_id)
        siblings = {
            other_id: self._service_state(other_id)
            for other_id, other in INFERENCE_BACKENDS.items()
            if other_id != backend_id and other["host"] == backend["host"]
            and other["gpu"] == backend["gpu"]
        }
        active_siblings = {key: state for key, state in siblings.items() if state["active"]}
        if target["active"]:
            self._verify_managed_pid(backend, target["pid"])
        for key, state in active_siblings.items():
            self._verify_managed_pid(INFERENCE_BACKENDS[key], state["pid"])
        allowed_pids = {state["pid"] for state in active_siblings.values()}
        if target["active"]:
            allowed_pids.add(target["pid"])
        if self._gpu_processes(backend) - allowed_pids:
            raise ValueError(f"{backend['label']} GPU {backend['gpu']} has an unmanaged compute process")
        for key, state in active_siblings.items():
            self._stop_backend_locked(key, state)
        deadline = time.monotonic() + 15
        while self._gpu_processes(backend) - ({target["pid"]} if target["active"] else set()):
            if time.monotonic() >= deadline:
                raise ValueError(f"{backend['label']} GPU {backend['gpu']} did not become free")
            time.sleep(1)
        prefix = backend.get("systemctl_prefix", "")
        if not target["active"]:
            result = self._remote(backend["host"],
                                  f"{prefix} systemctl --user start {backend['unit']}", timeout=20)
            if result.returncode:
                raise ValueError(f"unable to start {backend['label']}: {result.stderr.strip()[:300]}")
        try:
            self._manage_bridges(backend, "start")
        except ValueError:
            if not target["active"]:
                self._remote(backend["host"],
                             f"{prefix} systemctl --user stop {backend['unit']}", timeout=20)
            raise
        deadline = time.monotonic() + backend.get("ready_timeout_seconds", 120)
        while True:
            status = self.backend_status(backend_id)
            if status["ready"]:
                return
            if time.monotonic() >= deadline:
                raise ValueError(f"{backend['label']}: {status['reason']} (等待就绪超时)")
            time.sleep(2)

    def manage_inference(self, backend_id: str, action: str) -> dict:
        if backend_id not in INFERENCE_BACKENDS or action not in ("start", "stop"):
            raise ValueError("unsupported managed inference backend")
        with self.lock:
            if self.active_id is not None:
                raise ValueError(f"cannot {action} inference while simulation {self.active_id} is active")
            if action == "start":
                self._start_backend_locked(backend_id)
            else:
                self._stop_backend_locked(backend_id, self._service_state(backend_id))
        self._catalog_at = 0.0
        return self.backend_status(backend_id)

    def _metadata(self, run_id: str) -> dict:
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("invalid run ID")
        path = self.runs_dir / run_id / "metadata.json"
        if not path.is_file():
            raise ValueError("unknown run ID")
        return json.loads(path.read_text())

    def run(self, run_id: str) -> dict:
        metadata = self._metadata(run_id)
        if metadata["status"] in ("starting", "running") and self.active_id != run_id:
            metadata["status"] = "unverified"
            metadata["error"] = "console service restarted during this run; inspect the remote report before relaunching"
        report_path = self.runs_dir / run_id / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            metadata["result"] = {
                "status": report.get("status"),
                "error": report.get("error"),
                "task_objective": report.get("task_objective", "plate"),
                "episodes": [
                    {"scenario_id": item.get("scenario_id"), "success": item.get("success"),
                     "stop_reason": item.get("stop_reason"),
                     "plate_placed": item.get("plate_placed"),
                     "full_task_success": item.get("full_task_success"),
                     "left_return_diagnostic": item.get("left_return_diagnostic"),
                     "ever_success": item.get('ever_success'),
                     "first_success_policy_step": item.get('first_success_policy_step'),
                     "task_stage": (item.get("transitions") or [{}])[-1].get("task_stage"),
                     "policy_steps": item.get("policy_steps"),
                     "video_frames": item.get("video_frames"),
                     "joint_samples": item.get("joint_samples")}
                    for item in report.get("episodes", [])
                ],
            }
        metadata["video_available"] = (self.runs_dir / run_id / "video.mp4").is_file()
        metadata['three_views_available'] = all((self.runs_dir / run_id / f'video_{s}_wrist.mp4').is_file() for s in ('left','right'))
        preview = self.runs_dir / run_id / "preview.jpg"
        metadata["preview_available"] = preview.is_file()
        metadata["preview_version"] = preview.stat().st_mtime_ns if preview.is_file() else None
        metadata["joints_available"] = (self.runs_dir / run_id / "joints.csv").is_file()
        audit_path = self.runs_dir / run_id / "online_audit_summary.json"
        if audit_path.is_file():
            metadata["online_audit"] = json.loads(audit_path.read_text())
        for summary_name in ('sweep_20261008.json', 'precision_20261008.json'):
            sweep_path = ROOT/'sim_validation/pvc_stiffness'/summary_name
            if not sweep_path.is_file() or not report_path.is_file():
                continue
            trials = json.loads(sweep_path.read_text()).get('trials', [])
            trial = next((item for item in trials if item.get('run_id') == run_id), {})
            if trial.get('report_sha256') == hashlib.sha256(report_path.read_bytes()).hexdigest():
                metadata['cup_lift_summary'] = trial.get('result')
        metadata.pop("remote_dir", None)
        return metadata

    @staticmethod
    def _copy_live_preview(remote_dir: str, directory: Path, stop: threading.Event) -> None:
        preview = directory / "preview.jpg"
        temporary = directory / "preview.jpg.part"
        while not stop.is_set():
            try:
                result = subprocess.run(
                    [*SCP, f"{REMOTE_HOST}:{remote_dir}/preview.jpg", str(temporary)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=15, check=False,
                )
                if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size:
                    os.replace(temporary, preview)
            except (OSError, subprocess.TimeoutExpired):
                pass
            stop.wait(2)

    def recent_runs(self) -> list[dict]:
        directories = sorted(
            (p for p in self.runs_dir.iterdir()
             if p.is_dir() and RUN_ID.fullmatch(p.name) and (p / "metadata.json").is_file()),
            key=lambda path: (path / "metadata.json").stat().st_mtime,
            reverse=True,
        )
        return [self.run(path.name) for path in directories[:20]]

    def start(self, request: dict) -> dict:
        expected_fields = {"policy", "setup_index", "seed", "camera", "inference_backend"}
        if (not isinstance(request, dict) or not expected_fields.issubset(request)
                or set(request) - expected_fields - {"task_objective", "steps", "run_until_success", "left_return_diagnostic", "left_extra_closure_fraction", "contact_profile"}):
            raise ValueError("expected policy, setup_index, seed, camera, inference_backend "
                             "and optional steps, run_until_success, task_objective")
        policy_id = request["policy"]
        if not isinstance(policy_id, str):
            raise ValueError("invalid policy")
        policy = next((p for p in self.policies() if p["id"] == policy_id), None)
        if policy is None or not policy.get("ready"):
            raise ValueError("policy has no validated simulator adapter")
        setup_index = _int_field(request["setup_index"], "setup_index", 0, 10000)
        seed = _int_field(request["seed"], "seed", 0, 2**31 - 1)
        run_until_success = request.get("run_until_success", DEFAULT_RUN_MODE == 'until_success'
                                        and bool(policy.get('online_inference')) and 'steps' not in request)
        if type(run_until_success) is not bool:
            raise ValueError("run_until_success must be boolean")
        if run_until_success and not policy.get("online_inference"):
            raise ValueError("run-until-success requires an online model")
        steps = None if run_until_success else _int_field(request.get("steps"), "steps", 1, 10000)
        task_objective = request.get("task_objective", "plate")
        if task_objective not in ("plate", "plate_return"):
            raise ValueError("unsupported task objective")
        left_diagnostic = request.get('left_return_diagnostic', False)
        default_contact = (DEFAULT_CONTACT_PROFILE
                           if policy.get('simulator_profile') == 'tuned_online_v1' and not left_diagnostic
                           else 'baseline')
        contact_profile = request.get('contact_profile', default_contact)
        if contact_profile not in ('baseline', 'official_fingertip_friction', *SHELL_PROFILE_IDS):
            raise ValueError('unsupported contact profile')
        if contact_profile != 'baseline':
            if policy.get('simulator_profile') != 'tuned_online_v1' or left_diagnostic:
                raise ValueError('contact trial requires tuned online model and no action assistance')
            if contact_profile != _PVC_NUMERICS['PRECISION_ID'] and (run_until_success or steps > 600):
                raise ValueError('contact trial requires <=600 requests; verified high-precision cup supports continuous runs')
        if contact_profile in SHELL_PROFILE_IDS:
            _require_verified_pvc_probe(contact_profile)
        if type(left_diagnostic) is not bool:
            raise ValueError('left_return_diagnostic must be boolean')
        extra_closure = request.get('left_extra_closure_fraction', 0.)
        if (type(extra_closure) not in (int, float) or not math.isfinite(extra_closure)
                or not 0 <= extra_closure <= .05 or (extra_closure and not left_diagnostic)):
            raise ValueError('extra closure requires left diagnostic and a finite fraction between 0 and .05')
        if left_diagnostic and (policy_id != 'pi05-cup-clean-30000' or request['inference_backend'] != 'rtx5090'
                or task_objective != 'plate_return' or run_until_success or steps > 600):
            raise ValueError('left diagnostic requires squirrel pi05 30k, plate_return and <=600 requests')
        camera = request["camera"]
        if camera not in CAMERAS:
            raise ValueError("unsupported camera")
        if policy["kind"] not in ("builtin", "joint_script", "trajectory_script"):
            raise ValueError("unsupported policy kind")
        if policy["kind"] != "builtin":
            script = Path(policy["remote_script"])
            trusted_tuned = (policy_id == TUNED_REPLAY_ID and script == TUNED_ROOT / 'yubi_isaac_sim_env/policies/umi_left_second_height_replay.py')
            if not trusted_tuned and not script.is_relative_to(REMOTE_ROOT / "repo") and not script.is_relative_to(REMOTE_ROOT / "adapters"):
                raise ValueError("policy script is outside trusted simulator directories")
        if policy_id == TUNED_REPLAY_ID:
            if (setup_index, seed, steps, camera) != (0, 42, 215, 'overview'):
                raise ValueError('tuned paired replay requires scene 0, seed 42, 215 steps and overview; both wrists are recorded automatically')
        inference_backend = request["inference_backend"]
        if inference_backend not in (*INFERENCE_BACKENDS, "orin"):
            raise ValueError("unsupported inference backend")
        if policy.get("online_inference"):
            if inference_backend not in policy.get("supported_backends", ("a100", "rtx5090", "orin")):
                raise ValueError("selected inference backend is not validated for this policy")
        with self.lock:
            if self.active_id is not None:
                raise ValueError(f"simulation run {self.active_id} is already active")
            if not (policy.get("online_inference") and INFERENCE_BACKENDS[inference_backend]["host"] == REMOTE_HOST):
                status = self.gpu_status()
                if not status["available"]:
                    raise ValueError(status["reason"])
            if policy.get("online_inference"):
                self._start_backend_locked(inference_backend)
            run_id = uuid.uuid4().hex[:12]
            directory = self.runs_dir / run_id
            directory.mkdir(mode=0o700)
            remote_dir = str(REMOTE_ROOT / "runs" / f"console_{run_id}")
            if policy_id == TUNED_REPLAY_ID:
                remote_dir = f'/home/lrl/umi-tuned-replay-private/runs/console_{run_id}'
            metadata = {
                "id": run_id, "status": "starting",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "policy": policy_id, "policy_label": policy["label"],
                "setup_index": setup_index, "seed": seed, "steps": steps,
                "run_until_success": run_until_success,
                "task_objective": task_objective,
                "left_return_diagnostic": left_diagnostic,
                "contact_profile": contact_profile,
                "left_extra_closure_fraction": extra_closure,
                "camera": camera, "gpu": GPU_INDEX, "remote_dir": remote_dir,
                "canonical_hand_sides": True,
                "right_wrist_extrinsic_status": "unmeasured; image alignment under audit",
                "inference_backend": inference_backend if policy.get("online_inference") else None,
            }
            atomic_json(directory / "metadata.json", metadata)
            self.active_id = run_id
            thread = threading.Thread(target=self._execute, args=(metadata, policy),
                                      name=f"isaac-{run_id}", daemon=True)
            thread.start()
        self._catalog_at = 0.0
        return self.run(run_id)

    def stop_run(self, run_id: str) -> dict:
        with self.lock:
            metadata = self._metadata(run_id)
            if metadata["status"] not in ("starting", "running"):
                raise ValueError("run is not active")
            stop_file = REMOTE_ROOT / "runs" / f".stop_{run_id}"
            result = self._remote(REMOTE_HOST, shlex.join(["touch", "--", str(stop_file)]))
            if result.returncode:
                raise ValueError(f"unable to request simulation stop: {result.stderr.strip()[:300]}")
            metadata["stop_requested"] = True
            atomic_json(self.runs_dir / run_id / "metadata.json", metadata)
        return self.run(run_id)

    def _execute(self, metadata: dict, policy: dict, resume_existing: bool = False) -> None:
        run_id = metadata["id"]
        directory = self.runs_dir / run_id
        command = [
            str(REMOTE_ROOT / "run_headless.sh"), str(GPU_INDEX),
            "--setup", f"random:{metadata['setup_index']}",
            "--seed", str(metadata["seed"]), "--camera", metadata["camera"],
            "--task-objective", metadata["task_objective"],
            "--record-run", metadata["remote_dir"],
        ]
        if metadata.get("run_until_success"):
            command.append("--until-success")
        else:
            command.extend(("--steps", str(metadata["steps"])))
        command.extend(("--stop-file", str(REMOTE_ROOT / "runs" / f".stop_{run_id}")))
        if policy["kind"] == "builtin":
            command.extend(("--policy", policy["value"]))
        else:
            option = ("--trajectory-policy-script" if policy["kind"] == "trajectory_script"
                      else "--policy-script")
            command.extend((option, policy["remote_script"]))
            if policy.get("images"):
                command.extend(("--policy-images", "all"))
        # A separate remote lock prevents overlapping console launches even if
        # the local HTTP service is restarted while a simulation is running.
        environment = ["YUBI_CANONICAL_HAND_SIDES=1",
                       f"SIM_ADAPTER_AUDIT_DIR={metadata['remote_dir']}"]
        if policy.get("online_inference"):
            variable = policy.get("inference_url_env", "PI05_ONLINE_URL")
            command[0] = str(REMOTE_ROOT / 'run_reference_online.sh')
            command[command.index('--camera')+1] = 'head'
            command.append('--online-chunk-30hz')
            metadata['camera'] = 'head'
            metadata['recorded_views'] = ['head','left_wrist','right_wrist']
            environment.append('UMI_RECORD_THREE_VIEWS=1')
            command[command.index('--setup')+1] = str(REMOTE_ROOT / 'adapters/reference_259632.json')
            environment.extend(("UMI_ONLINE_SMOOTH=1", "UMI_ONLINE_CALIBRATION=reference_259632_v1",
                                "UMI_TRAJECTORY_PREVIEW=1",
                                "UMI_EXECUTE_30HZ=1",
                                "UMI_REFERENCE_FLANGE_PLUS90=1",
                                f"UMI_REFERENCE_DIAGNOSTIC={REMOTE_ROOT}/adapters/reference_259632.json"))
            if (policy["id"].startswith("pi05-")
                    and INFERENCE_BACKENDS[metadata["inference_backend"]]["host"] == REMOTE_HOST):
                environment.append("UMI_REQUIRE_CAUSAL_PI05=1")
                # Matched-frame camera probes on reference 259632 showed that
                # the default inspection light underexposes the tabletop and
                # cup. Two independent 40-step causal rollouts at intensity
                # 500 approached the cup; the 220 controls diverged upward.
                # This is an approximate visual prior, not camera calibration.
                environment.append("UMI_REFERENCE_LIGHT_INTENSITY=500")
                environment.append("UMI_REFERENCE_DARK_FINGERS=1")
                metadata["visual_profile"] = {
                    "reference_light_intensity": 500,
                    "dark_finger_visual_material": True,
                    "basis": "matched-frame real/Isaac exposure and bounded causal rollouts",
                    "measured_hand_eye_calibration": False,
                }
                if policy["id"] == "pi05-cup-clean-30000-gain135":
                    environment.append("UMI_PI05_RIGHT_TRANSLATION_GAIN=1.35")
                    metadata["evaluation_class"] = "provisional_action_scale_diagnostic_not_baseline_model"
                    metadata["right_translation_gain"] = 1.35
                if policy["id"] in ("pi05-cup-clean-30000-assisted",
                                    "pi05-cup-clean-30000-assisted-extra30",
                                    "pi05-cup-clean-30000-assisted-extra30-heldwrist",
                                    "pi05-cup-clean-30000-assisted-extra30-centered",
                                    "pi05-cup-clean-30000-assisted-extra30-gapgate",
                                    "pi05-cup-clean-30000-assisted-extra30-earlylevel",
                                    "pi05-cup-clean-30000-assisted-extra30-earlylevel-zfollow",
                                    "pi05-cup-clean-30000-assisted-physics",
                                    "pi05-cup-clean-30000-assisted-friction",
                                    "pi05-cup-clean-30000-assisted-friction-only"):
                    environment.extend((
                        "UMI_PREGRASP_APPROACH_M=0.030",
                        "UMI_PREGRASP_LATERAL_CENTER_M=0.040",
                        "UMI_PREGRASP_WRIST_LEVEL_MAX_DEG=15",
                        "UMI_PREGRASP_CONTACT_PRELOAD_FRACTION=0.10",
                        f"UMI_FINGER_MESH_DIR={REMOTE_ROOT}/repo/yubi_isaac_sim_env/assets/yubi/meshes",
                        "YUBI_DIAGNOSTIC_CUP_CONTACTS=1",
                    ))
                    metadata["evaluation_class"] = "assisted_grasp_diagnostic_not_pure_model"
                    if policy["id"] in ("pi05-cup-clean-30000-assisted-extra30",
                                        "pi05-cup-clean-30000-assisted-extra30-heldwrist",
                                        "pi05-cup-clean-30000-assisted-extra30-centered",
                                        "pi05-cup-clean-30000-assisted-extra30-gapgate",
                                        "pi05-cup-clean-30000-assisted-extra30-earlylevel",
                                        "pi05-cup-clean-30000-assisted-extra30-earlylevel-zfollow"):
                        environment.append("UMI_PREGRASP_ADDITIONAL_M=0.030")
                        metadata["preclosure_approach"] = {
                            "base_m": 0.030, "additional_m": 0.030,
                            "total_from_model_waypoint_m": 0.060,
                            "requires_measured_contact_and_two_sided_pad_gate": True,
                        }
                        if policy["id"] in ("pi05-cup-clean-30000-assisted-extra30-heldwrist",
                                            "pi05-cup-clean-30000-assisted-extra30-centered",
                                            "pi05-cup-clean-30000-assisted-extra30-gapgate",
                                            "pi05-cup-clean-30000-assisted-extra30-earlylevel",
                                            "pi05-cup-clean-30000-assisted-extra30-earlylevel-zfollow"):
                            environment.append("UMI_PREGRASP_HOLD_TRIGGER_ORIENTATION=1")
                            metadata["preclosure_approach"]["hold_trigger_orientation"] = True
                        if policy["id"] in ("pi05-cup-clean-30000-assisted-extra30-centered",
                                            "pi05-cup-clean-30000-assisted-extra30-gapgate",
                                            "pi05-cup-clean-30000-assisted-extra30-earlylevel",
                                            "pi05-cup-clean-30000-assisted-extra30-earlylevel-zfollow"):
                            environment.append("UMI_PREGRASP_EARLY_LATERAL_CENTER=1")
                            metadata["preclosure_approach"]["early_lateral_centering"] = True
                        if policy["id"] in ("pi05-cup-clean-30000-assisted-extra30-gapgate",
                                             "pi05-cup-clean-30000-assisted-extra30-earlylevel",
                                             "pi05-cup-clean-30000-assisted-extra30-earlylevel-zfollow"):
                            environment.append("UMI_PREGRASP_BALANCE_PAD_GAPS=1")
                            metadata["preclosure_approach"]["balance_pad_gaps_before_forward"] = True
                        if policy["id"] in ("pi05-cup-clean-30000-assisted-extra30-earlylevel",
                                             "pi05-cup-clean-30000-assisted-extra30-earlylevel-zfollow"):
                            environment.append("UMI_PREGRASP_EARLY_LEVEL=1")
                            metadata["preclosure_approach"]["level_before_approach"] = True
            metadata['effective_setup'] = 'reference_259632_fixed_cup_plate'
            scene = (REMOTE_ROOT / "variants" / "online_smooth_grip_diagnostic_20260930.usda"
                     if policy["id"] == "pi05-cup-clean-30000-assisted-physics"
                     else REMOTE_ROOT / "variants" / "online_smooth_grip_force_friction_20260930.usda"
                     if policy["id"] == "pi05-cup-clean-30000-assisted-friction"
                     else REMOTE_ROOT / "variants" / "online_smooth_grip_friction_only_20260930.usda"
                     if policy["id"] == "pi05-cup-clean-30000-assisted-friction-only"
                     else REMOTE_ROOT / "repo" / "yubi_isaac_sim_env" / "scenes"
                          / "dual_franka_yubi_online_smooth.usda")
            command.extend(("--scene", str(scene)))
            if policy["id"] == "pi05-cup-clean-30000-assisted-physics":
                metadata["evaluation_class"] = "assisted_grasp_physics_diagnostic_not_pure_model"
                metadata["scene_hypothesis"] = {
                    "jaw_max_force_Nm": 2, "jaw_static_friction": 2,
                    "jaw_dynamic_friction": 1.5,
                    "jaw_collision_approximation": "convexDecomposition",
                    "measured_hardware_values": False,
                }
            if policy["id"] == "pi05-cup-clean-30000-assisted-friction":
                metadata["evaluation_class"] = "assisted_grasp_force_friction_diagnostic_not_pure_model"
                metadata["scene_hypothesis"] = {
                    "jaw_max_force_Nm": 2, "jaw_static_friction": 2,
                    "jaw_dynamic_friction": 1.5,
                    "jaw_collision_approximation": "original",
                    "measured_hardware_values": False,
                }
            if policy["id"] == "pi05-cup-clean-30000-assisted-friction-only":
                metadata["evaluation_class"] = "assisted_grasp_friction_only_diagnostic_not_pure_model"
                metadata["scene_hypothesis"] = {
                    "jaw_max_force_Nm": "original", "jaw_static_friction": 2,
                    "jaw_dynamic_friction": 1.5,
                    "jaw_collision_approximation": "original",
                    "measured_hardware_values": False,
                }
            metadata["initial_state"] = {
                "profile": "reference_259632_v1", "source_episode": 259632,
                "source_frame": 0, "simulation_fit": True, "physical_calibration_measured": False,
                "reset_gate": "position <= 20 mm, orientation <= 10 deg; otherwise refuse model start"}
            metadata["online_control"] = {"calibration": "reference_259632_v1", "measured": False,
                                          "model_request_hz": 10, "action_execution_hz": 30,
                                          "continuous_targets": True, "velocity_rad_s": .8,
                                          "acceleration_rad_s2": 1.5,
                                          "chunk_preview": "natural_cubic_0.7_feedback_0.3_feedforward",
                                          "panda_drive": {"stiffness": 2500, "damping": 70.710678,
                                                          "scope": "isolated_simulation_scene"}}
            environment.append(f"{variable}={INFERENCE_BACKENDS[metadata['inference_backend']]['adapter_url']}")
        remote_command = shlex.join(
            ["flock", "-n", str(REMOTE_ROOT / "console.lock"), "env", *environment, *command]
        )
        if policy['id'] == TUNED_REPLAY_ID:
            # Explicit opt-in successful replay profile; never use its fixed
            # demonstration corrections as an unannounced online-model assist.
            remote_command = shlex.join(['flock', '-n', str(REMOTE_ROOT / 'console.lock'),
                                        'bash', str(TUNED_ROOT / 'run_tuned_branch.sh'), metadata['remote_dir'],
                                        str(REMOTE_ROOT / 'runs' / f'.stop_{run_id}')])
            metadata.update(simulator_profile='tuned_v1', recorded_views=['overview','left_wrist','right_wrist'],
                            anatomical_mounts={'left':'LeftMount','right':'RightMount'},
                            effective_setup='replay_259632_259633_tuned',
                            evaluation_class='recorded_demonstration_replay_not_online_model')
        elif policy.get('simulator_profile') == 'tuned_online_v1':
            if not policy.get('online_inference') or '-assisted' in policy['id']:
                raise ValueError('tuned online profile is restricted to baseline trained policies')
            backend = INFERENCE_BACKENDS[metadata['inference_backend']]
            if backend['host'] != REMOTE_HOST:
                raise ValueError('tuned online profile only runs on squirrel_5090')
            if policy.get('model_provenance'):
                metadata['model_provenance'] = policy['model_provenance']
            adapter = Path(policy['remote_script']).name
            variable = policy.get('inference_url_env', 'PI05_ONLINE_URL')
            remote_command = shlex.join(['flock', '-n', str(REMOTE_ROOT / 'console.lock'),
                'env', f"UMI_MODEL_UNIT={backend['unit']}", f"{variable}={backend['adapter_url']}",
                'bash', policy.get('remote_launcher', str(TUNED_ROOT/'run_tuned_online.sh')), metadata['remote_dir'], adapter,
                str(REMOTE_ROOT/'runs'/f'.stop_{run_id}'),
                'until-success' if metadata.get('run_until_success') else str(metadata['steps']),
                metadata['task_objective']])
            metadata.update(simulator_profile='tuned_online_v1', camera='head',
                recorded_views=['head','left_wrist','right_wrist'],
                anatomical_mounts={'left':'LeftMount','right':'RightMount'},
                effective_setup='online_tuned_v1_fixed_initial_state',
                evaluation_class='pure_model_online_with_provisional_CAD_simulation_calibration',
                initial_state={'profile':'online_tuned_v1', 'source_frame':0,
                               'simulation_fit':True, 'physical_calibration_measured':False},
                visual_profile=_tuned_visual_profile(),
                online_control={'calibration':'tuned_online_v1', 'measured':False,
                                'model_request_hz':10, 'action_execution_hz':30,
                                'continuous_targets':True, 'velocity_rad_s':.8, 'acceleration_rad_s2':1.5,
                                'chunk_preview':'natural_cubic_0.7_feedback_0.3_feedforward',
                                'jaw_mapping':'fixed_CAD_distal_gap_bidirectional', 'two_mirrored_jaw_drives':True,
                                'replay_specific_offsets':False, 'oracle_action_feedback':False})
            if (adapter in ('pi05_isaac_online_adapter.py', 'pi05_intersection_isaac_online_adapter.py', 'pi05_intersection_10000_isaac_online_adapter.py', 'pi05_intersection_20000_isaac_online_adapter.py')
                    and metadata.get('contact_profile') == 'pvc_shell_e3000mpa_i128_h240_v2'
                    and metadata['task_objective'] == 'plate_return'
                    and not metadata.get('left_return_diagnostic')):
                metadata['online_control'].update(motion_profile='natural_fast_v1',
                    response_gain=6, jaw_response_gain=4, velocity_rad_s=1.2, acceleration_rad_s2=2.4)
                metadata['evaluation_class'] = 'live_model_manipulation_with_explicit_sequential_homing'
                metadata['sequential_homing'] = {'right_home_before_left': True,
                    'left_extra_closure_rad': .005, 'right_retreat_vertical_m': .12}
            if adapter in ('lingbot_isaac_online_adapter.py', 'lingbot_official_isaac_online_adapter.py'):
                from deploy_servers.official_cup_prompts import PROMPT_PROTOCOL, PROMPT_SOURCE
                metadata['language_instructions'] = {'protocol': PROMPT_PROTOCOL, 'source': PROMPT_SOURCE,
                    'selection': 'right_place_then_left_return_after_released_placement',
                    'inference_mode': 'native_lingbot_vla_v2_chunk', 'custom_combined_prompt': False,
                    'extra_closure_fraction': 0., 'cup_position_action_assistance': False}
            if adapter in ('pi05_intersection_isaac_online_adapter.py', 'openwam_intersection_isaac_online_adapter.py'):
                from deploy_servers.official_cup_prompts import PROMPT_PROTOCOL, PROMPT_SOURCE
                metadata['language_instructions'] = {'protocol': PROMPT_PROTOCOL, 'source': PROMPT_SOURCE,
                    'selection': 'evaluator stage: right place, then left return',
                    'inference_mode': 'future_aligned_10hz_row0', 'custom_combined_prompt': False}
                metadata['online_control'].update(target_update_hz=10, action_hz=10,
                    rows_consumed=[0], servo_hz=30, future_row_shift=False,
                    chunk_preview='single 100ms endpoint held for three servo ticks')
            if metadata.get('left_return_diagnostic'):
                remote_command += ' left-return'
                extra_closure = metadata.get('left_extra_closure_fraction', 0.)
                if extra_closure:
                    remote_command += ' ' + shlex.quote(f'{extra_closure:.6f}')
                metadata.update(
                    policy_label='π0.5 30k · 杯在盘上 / 左臂分阶段诊断（右臂保持）',
                    effective_setup='left_return_diagnostic',
                    evaluation_class='left_phase_reset_intervention_not_full_task_evaluation',
                    diagnostic_interventions={'cup_on_plate_at_reset_only': True, 'right_arm_commands_held': True,
                        'original_table_return_origin_m': [-.01647554668358119,-.022639707627907582,.75],
                        'model_prompt_changed': False, 'oracle_action_feedback': False})
                if extra_closure:
                    metadata['policy_label'] += ' · 额外闭合对照'
                    metadata['diagnostic_interventions']['left_extra_closure'] = {
                        'max_fraction_of_stroke': extra_closure, 'fade_start_fraction': .55,
                        'unchanged_open_release_at_or_above_fraction': .65,
                        'force_limit_changed': False, 'friction_changed': False,
                        'physical_aperture_measured': False, 'pure_model_action': False}
            contact_profile = metadata.get('contact_profile', 'baseline')
            if contact_profile != 'baseline':
                # Slots 6/7 stay explicit; this must not be interpreted as
                # the unrelated left-return intervention.
                remote_command += ' baseline 0 ' + shlex.quote(contact_profile)
                metadata.update(evaluation_class='model_in_unmeasured_physics_contact_trial',
                    physics_contact_trial={'id': contact_profile, 'measured': False,
                        'finger_static_friction': .8, 'finger_dynamic_friction': .8,
                        'friction_combine_mode': 'max', 'cup_material_changed': False,
                        'cup_deformable': False, 'drive_force_changed': False,
                        'model_actions_changed': False})
                if contact_profile in SHELL_PROFILE_IDS:
                    metadata['physics_contact_trial'].update(cup_material_changed=True,
                        cup_deformable=True, physics_hz=_PVC_NUMERICS['physics_hz_for'](contact_profile),
                        material_status='Unmeasured PVC-like elastic shell; not calibrated PVC; no plastic yield',
                        rigid_cup_colliders_disabled=True, force_telemetry_available=False)
        if metadata.get('sequential_homing'):
            metadata['evaluation_class'] = 'live_model_manipulation_with_explicit_sequential_homing'
            metadata.setdefault('physics_contact_trial', {})['model_actions_changed'] = True
        with self.lock:
            metadata["status"] = "running"
            atomic_json(directory / "metadata.json", metadata)
        preview_stop = threading.Event()
        preview_thread = threading.Thread(
            target=self._copy_live_preview,
            args=(metadata["remote_dir"], directory, preview_stop),
            name=f"preview-{run_id}", daemon=True,
        )
        # User requested result-only output: no preview transfer thread.
        try:
            if resume_existing:
                with (directory / 'run.log').open('a') as log:
                    log.write('\nReattached to the identity-checked existing durable runtime; no relaunch.\n')
                result = self._wait_existing_runtime(run_id)
            else:
                with (directory / "run.log").open("w") as log:
                    result = subprocess.run(
                        [*SSH, REMOTE_HOST, remote_command],
                        stdout=log, stderr=subprocess.STDOUT,
                        timeout=None if metadata.get("run_until_success") else 45 * 60,
                        check=False,
                    )
                if result.returncode == 255 and policy.get('simulator_profile') == 'tuned_online_v1':
                    # A broken SSH transport does not terminate systemd's
                    # durable simulator. Never stop its inference server
                    # while that identity-checked runtime is still active.
                    metadata['ssh_transport_recovered'] = True
                    with (directory/'run.log').open('a') as log:
                        log.write('\nSSH transport lost; tracking existing runtime without relaunch.\n')
                    result = self._wait_existing_runtime(run_id)
            for name in ("report.json", "manifest.json", "video.mp4", "video_left_wrist.mp4", "video_right_wrist.mp4", "joints.csv", *(['wrist_camera_poses.jsonl'] if policy['id'] == TUNED_REPLAY_ID or policy.get('simulator_profile') == 'tuned_online_v1' else []), *(['gripper_contact_audit.jsonl'] if policy.get('simulator_profile') == 'tuned_online_v1' and metadata.get('contact_profile') not in SHELL_PROFILE_IDS else []), *(['cup_deformation.jsonl'] if metadata.get('contact_profile') in SHELL_PROFILE_IDS else [])):
                subprocess.run(
                    [*SCP, f"{REMOTE_HOST}:{metadata['remote_dir']}/{name}", str(directory / name)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=None if metadata.get("run_until_success") else 120, check=False,
                )
            if policy["id"] == "grasp-contact-probe":
                name = "grasp_calibration.jsonl"
                subprocess.run(
                    [*SCP, f"{REMOTE_HOST}:{metadata['remote_dir']}/{name}", str(directory / name)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=120, check=False,
                )
            if policy.get("online_inference"):
                for name in ("online_adapter.jsonl", "continuous_targets.jsonl", *(['left_return_diagnostic.jsonl'] if metadata.get('left_return_diagnostic') else []), *(f"input_{side}_{step:04d}.jpg"
                                                      for step in range(3) for side in ("head", "left", "right"))):
                    subprocess.run([*SCP, f"{REMOTE_HOST}:{metadata['remote_dir']}/{name}", str(directory / name)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=None if metadata.get("run_until_success") else 120, check=False)
            report_path = directory / "report.json"
            report = json.loads(report_path.read_text()) if report_path.is_file() else {}
            if report.get('cup_physics_profile'):
                metadata['cup_physics_profile'] = report['cup_physics_profile']
            if report.get('visual_profile'):
                metadata['visual_profile'] = report['visual_profile']
            complete = (result.returncode == 0 and report.get("status") in ("completed", "stopped")
                        and (directory / "video.mp4").is_file()
                        and (directory / "joints.csv").is_file())
            if complete and policy['id'] == TUNED_REPLAY_ID and report.get('status') == 'completed':
                from dataset_replay.publish_tuned_branch_run import verify
                metadata['tuned_replay_validation'] = verify(directory, require_wrists=True)
            if complete and policy["id"] == "grasp-contact-probe":
                trace = directory / "grasp_calibration.jsonl"
                episode = report.get("episodes", [{}])[0]
                rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.is_file() else []
                reset = rows[0] if rows else {}
                steps = rows[1:] if rows else []
                valid = (reset.get("event") == "reset"
                         and len(steps) == episode.get("policy_steps")
                         and [row.get("step") for row in steps] == list(range(len(steps)))
                         and episode.get("video_frames") == episode.get("joint_samples"))
                complete = complete and valid
                if valid:
                    initial_z = float(reset["cup_initial_position_m"][2])
                    cup = episode["final_observation"]["objects"]["cup"]
                    final_z = float(cup["position_m"][2])
                    _, x, y, _ = cup["quaternion_wxyz"]
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x*x + y*y)))))
                    max_lift = max(row["cup_position_m"][2] for row in steps) - initial_z
                    reached_close = any(row["phase"] == "close" for row in steps)
                    stable_steps = _stable_lift_streak(steps, initial_z)
                    stable_lift_candidate = stable_steps >= 10
                    summary = {
                        "classification": ("stable_lift_candidate" if stable_lift_candidate
                                           else "contact_or_transient_lift_only"),
                        "training_model_used": False,
                        "scene_tuning": "原始场景；摩擦与夹爪驱动仍为默认估计",
                        "initial_cup_world_m": reset["cup_initial_position_m"],
                        "initial_tool_world_pose": reset["tool_initial_pose"],
                        "max_cup_lift_m": max_lift,
                        "close_phase_reached": reached_close,
                        "final_cup_lift_m": final_z - initial_z,
                        "final_cup_tilt_deg": tilt,
                        "stable_lift_hold_steps": stable_steps,
                        "stable_lift_candidate": stable_lift_candidate,
                        "stable_grasp_verified": False,
                        "success_flag": episode.get("success"),
                        "caveat": "YUBI contact and UMI hand-root transforms are not physically measured.",
                    }
                    atomic_json(directory / "calibration_summary.json", summary)
                    metadata["calibration_result"] = {
                        "label": ("稳定提起候选，仍需双侧接触核验" if stable_lift_candidate else
                                  "接触后滑脱，稳定抓取未通过" if reached_close else
                                  "尚未闭爪接触；请检查步数或对准"),
                        "max_cup_lift_mm": round(1000 * max_lift, 1),
                        "final_cup_tilt_deg": round(tilt, 1),
                        "scene_tuning": summary["scene_tuning"],
                    }
            if complete and policy.get("online_inference") and report.get("episodes", [{}])[0].get("policy_steps", 0):
                audit_path = directory / "online_adapter.jsonl"
                rows = [json.loads(line) for line in audit_path.read_text().splitlines()] if audit_path.is_file() else []
                expected = report.get("episodes", [{}])[0].get("policy_steps")
                future_aligned = policy['id'] in ('pi05-cup-intersection-10000', 'pi05-cup-intersection-20000', 'pi05-cup-intersection-30000',
                                                   'openwam-cup-intersection-fullpass-5069')
                pose_rows, grip_rows = ([0], [0]) if future_aligned else ([1, 2, 3], [0, 1, 2])
                valid = (len(rows) == expected and [row["step"] for row in rows] == list(range(expected))
                         and all(row.get("observation_origin") == "current_simulator_render_and_robot_state" for row in rows)
                         and all(row.get("image_shapes") == {"left": [480, 640, 3], "right": [480, 640, 3]} for row in rows)
                         and all(row.get('execution_hz') == 30 and len(row.get('substep_waypoints') or []) == 3
                                 for row in rows)
                         and report.get('action_execution_hz') == 30
                         and report.get('model_request_hz') == 10
                         and (not policy["id"].startswith("pi05-")
                              or INFERENCE_BACKENDS[metadata["inference_backend"]]["host"] != REMOTE_HOST
                              or all((row.get("action_timing") or {}).get("pose_rows") == pose_rows
                                     and (row.get("action_timing") or {}).get("gripper_rows") == grip_rows
                                     and (policy.get('simulator_profile') == 'tuned_online_v1'
                                          or (row.get("reference_light_intensity") == 500
                                              and row.get("reference_dark_fingers") is True))
                                     for row in rows))
                         and (policy["id"] not in ("pi05-cup-clean-30000-assisted",
                                                   "pi05-cup-clean-30000-assisted-extra30",
                                                   "pi05-cup-clean-30000-assisted-extra30-heldwrist",
                                                   "pi05-cup-clean-30000-assisted-extra30-centered",
                                                   "pi05-cup-clean-30000-assisted-extra30-gapgate",
                                                   "pi05-cup-clean-30000-assisted-extra30-earlylevel",
                                                   "pi05-cup-clean-30000-assisted-extra30-earlylevel-zfollow",
                                                   "pi05-cup-clean-30000-assisted-physics",
                                                   "pi05-cup-clean-30000-assisted-friction",
                                                   "pi05-cup-clean-30000-assisted-friction-only")
                              or all((row.get("pregrasp_approach") or {}).get("mode")
                                     == "assisted_preclosure_diagnostic" for row in rows))
                         and 1+3*(expected-1)+1 <= report.get('episodes',[{}])[0].get('video_frames', 0)
                         <= 1+3*expected)
                motion = {}
                gripper_feedback = {side: {"driven": [], "mimic": []} for side in ("left", "right")}
                with (directory / "joints.csv").open(newline="") as source:
                    for row in csv.DictReader(source):
                        if row["joint_kind"] == "arm":
                            key = f"{row['side']}.{row['joint_name']}"
                            motion.setdefault(key, []).append(float(row["position"]))
                        elif row["joint_name"] == "yubi_finger_joint":
                            gripper_feedback[row["side"]]["driven"].append(float(row["position"]))
                        elif row["joint_name"] == "yubi_finger_mimic_joint":
                            gripper_feedback[row["side"]]["mimic"].append(float(row["position"]))
                max_joint_motion = max((max(values)-min(values) for values in motion.values()), default=0.0)
                measured_grippers = {}
                for side, samples in gripper_feedback.items():
                    driven, mimic = samples["driven"], samples["mimic"]
                    if driven and len(driven) == len(mimic):
                        measured_grippers[side] = {
                            "start_rad": driven[0], "min_rad": min(driven),
                            "max_rad": max(driven), "end_rad": driven[-1],
                            "max_mimic_opposition_error_rad": max(abs(a+b) for a,b in zip(driven,mimic)),
                        }
                summary = {
                    "validated": bool(valid), "online_observation_steps": len(rows),
                    "expected_policy_steps": expected,
                    "model_request_hz": 10, "action_execution_hz": 30,
                    "action_timing": rows[0].get("action_timing") if rows else None,
                    "visual_profile": metadata.get("visual_profile"),
                    "evaluation_class": metadata.get("evaluation_class", "pure_model_online"),
                    "live_wrist_images": bool(valid and all((directory / f"input_{side}_0000.jpg").is_file()
                                                           for side in ("left", "right"))),
                    "live_head_image": (directory / "input_head_0000.jpg").is_file(),
                    "model_input_views": rows[0].get("model_input_views", ["left_wrist", "right_wrist"]) if rows else [],
                    "world_pose_transform": rows[0].get("endpoint_assumption") if rows else None,
                    "calibration": rows[0].get("calibration") if rows else None,
                    "gripper_calibration": rows[0].get("gripper_calibration") if rows else None,
                    "max_arm_joint_motion_rad": max_joint_motion,
                    "measured_gripper_joint_rad": measured_grippers,
                    "mean_model_latency_ms": sum(row["model_latency_ms"] for row in rows)/len(rows) if rows else None,
                }
                atomic_json(directory / "online_audit_summary.json", summary)
                if policy.get('simulator_profile') == 'tuned_online_v1':
                    from verify_tuned_online import verify
                    try:
                        summary['tuned_profile_audit'] = verify(directory)
                    except (AssertionError, KeyError) as exc:
                        raise ValueError(f'tuned online audit failed: {exc or "artifact contract mismatch"}') from exc
                    atomic_json(directory / 'online_audit_summary.json', summary)
                complete = complete and valid and summary["live_wrist_images"]
            metadata["status"] = ("stopped" if report.get("status") == "stopped" else "completed") if complete else "failed"
            if not complete:
                trace = directory / "online_adapter.jsonl"
                if trace.is_file():
                    with trace.open() as source:
                        metadata["observed_policy_steps"] = sum(1 for line in source if line.strip())
                metadata["error"] = (
                    f"simulator exit={result.returncode}; report={report.get('status', 'missing')}; "
                    f"{report.get('error', 'inspect run.log and report before retrying')}"
                )
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            preview_stop.set()
            if preview_thread.is_alive():
                preview_thread.join(timeout=20)
            with self.lock:
                backend_id = metadata.get("inference_backend")
                if backend_id in INFERENCE_BACKENDS:
                    try:
                        state = self._service_state(backend_id)
                        self._stop_backend_locked(backend_id, state)
                        metadata["inference_cleanup"] = "stopped"
                    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                        metadata["inference_cleanup"] = f"requires_review: {exc}"
                current = self._metadata(run_id)
                metadata["stop_requested"] = current.get("stop_requested", False)
                metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
                atomic_json(directory / "metadata.json", metadata)
                if self.active_id == run_id:
                    self.active_id = None
            # A finished run releases its dedicated inference service and GPU.
            # Do not keep serving the "GPU busy" catalog snapshot for 90 s.
            self._catalog_at = 0.0


class SimulationServer(ThreadingHTTPServer):
    def __init__(self, address, runner: SimulationRunner):
        super().__init__(address, SimulationHandler)
        self.runner = runner


class SimulationHandler(BaseHTTPRequestHandler):
    server: SimulationServer

    def _data(self, payload: dict | list, status=200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _file(self, path: Path, content_type: str, include_body=True) -> None:
        if not path.is_file():
            return self.send_error(404)
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get("Range", "")
        if requested:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
            if not match:
                return self.send_error(416)
            start = int(match.group(1))
            end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
            if start >= size or end < start:
                return self.send_error(416)
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if include_body:
            with path.open("rb") as source:
                source.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = source.read(min(256 * 1024, remaining))
                    if not block:
                        break
                    try:
                        self.wfile.write(block)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(block)

    def _get(self, include_body=True) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/simulation.html"):
            return self._file(ROOT / "simulation.html", "text/html; charset=utf-8", include_body)
        if path == "/dataset-replay":
            return self._file(ROOT / "dataset_replay" / "review.html", "text/html; charset=utf-8", include_body)
        if path == "/dataset-replay/tuned/status.json":
            return self._file(ROOT / "dataset_replay" / "tuned_branch_status.json",
                              "application/json", include_body)
        if path == "/dataset-replay/tuned/video.mp4":
            return self._file(ROOT / "dataset_replay" / "tuned_branch_video.mp4",
                              "video/mp4", include_body)
        tuned_video = re.fullmatch(r'/dataset-replay/tuned/artifacts/([A-Za-z0-9_-]+)/(video(?:_left_wrist|_right_wrist)?\.mp4)', path)
        if tuned_video:
            return self._file(ROOT / 'dataset_replay' / 'tuned_published' / tuned_video.group(1) / tuned_video.group(2),
                              'video/mp4', include_body)
        if path == "/dataset-replay/official/manifest.json":
            return self._file(ROOT / "dataset_replay" / "official_cup_5" / "manifest.json",
                              "application/json", include_body)
        if path == "/dataset-replay/official/player.js":
            return self._file(ROOT / "dataset_replay" / "official_replay.js",
                              "text/javascript; charset=utf-8", include_body)
        official_episode = re.fullmatch(r"/dataset-replay/official/episode-(\d+)\.json", path)
        if official_episode and int(official_episode.group(1)) in OFFICIAL_CUP_EPISODES:
            return self._file(ROOT / "dataset_replay" / "official_cup_5" /
                              f"episode-{official_episode.group(1)}.json",
                              "application/json", include_body)
        official_video = re.fullmatch(r"/dataset-replay/official/episode-(\d+)-(center|left|right)\.mp4", path)
        if official_video and int(official_video.group(1)) in OFFICIAL_CUP_EPISODES:
            return self._file(ROOT / "videos" /
                              f"episode-{official_video.group(1)}-{official_video.group(2)}.mp4",
                              "video/mp4", include_body)
        replay_asset = re.fullmatch(
            r"/dataset-replay/assets/episode_1271_(?:center|left|right|sim_full_(?:rear|left_wrist|right_wrist)_(?:normalfriction|highfriction))\.mp4", path)
        if replay_asset:
            return self._file(ROOT / "dataset_replay" / "review_assets" / path.rsplit("/", 1)[-1],
                              "video/mp4", include_body)
        corrected_replay = re.fullmatch(
            r"/dataset-replay/assets/episode_1271_tool_roll180_(?:overview|left_wrist|camera_upright)_20260929\.mp4", path)
        if corrected_replay:
            return self._file(ROOT / "dataset_replay" / "review_assets" / path.rsplit("/", 1)[-1],
                              "video/mp4", include_body)
        fixed_y_replay = re.fullmatch(
            r"/dataset-replay/assets/episode_1271_fixedy22_padcorridor_(?:overview|left_wrist|right_wrist_candidate)_20260929\.mp4", path)
        if fixed_y_replay:
            return self._file(ROOT / "dataset_replay" / "review_assets" / path.rsplit("/", 1)[-1],
                              "video/mp4", include_body)
        if path == "/dataset-replay/assets/episode_1271_fixedy22_padcorridor_summary_20260929.json":
            return self._file(ROOT / "dataset_replay" / "review_assets" /
                              "episode_1271_fixedy22_padcorridor_summary_20260929.json",
                              "application/json", include_body)
        if path == "/dataset-replay/assets/episode_1271_tool_roll180_camera_upright_summary_20260929.json":
            return self._file(ROOT / "dataset_replay" / "review_assets" /
                              "episode_1271_tool_roll180_camera_upright_summary_20260929.json",
                              "application/json", include_body)
        if path == "/dataset-replay/assets/right_wrist_landmarks_tool_roll180_20260929.json":
            return self._file(ROOT / "dataset_replay" / "right_wrist_landmarks_tool_roll180_20260929.json",
                              "application/json", include_body)
        if path == "/dataset-replay/assets/episode_1271_sim_full_right_wrist_follow_verified_20260929.mp4":
            return self._file(ROOT / "dataset_replay" / "review_assets" /
                              "episode_1271_sim_full_right_wrist_follow_verified_20260929.mp4",
                              "video/mp4", include_body)
        if path == "/dataset-replay/assets/episode_1271_flange180_right_wrist_full_20260929.mp4":
            return self._file(ROOT / "dataset_replay" / "review_assets" /
                              "episode_1271_flange180_right_wrist_full_20260929.mp4",
                              "video/mp4", include_body)
        if path == "/dataset-replay/assets/episode_1271_flange180_validation_20260929.json":
            return self._file(ROOT / "dataset_replay" /
                              "episode_1271_flange180_validation_20260929.json",
                              "application/json", include_body)
        if path == "/dataset-replay/assets/right_wrist_online_input_random000_20260929.png":
            return self._file(ROOT / "dataset_replay" / "review_assets" /
                              "right_wrist_online_input_random000_20260929.png",
                              "image/png", include_body)
        replay_summary = re.fullmatch(r"/dataset-replay/assets/episode_1271_full_replay_20260929\.json", path)
        if replay_summary:
            return self._file(ROOT / "dataset_replay" / "review_assets" / path.rsplit("/", 1)[-1],
                              "application/json", include_body)
        if path == "/dataset-replay/assets/right_wrist_landmarks_audit_20260929.json":
            return self._file(ROOT / "dataset_replay" / "right_wrist_landmarks_audit_20260929.json",
                              "application/json", include_body)
        if path == "/dataset-replay/assets/right_wrist_landmarks_follow_verified_20260929.json":
            return self._file(ROOT / "dataset_replay" / "right_wrist_landmarks_follow_verified_20260929.json",
                              "application/json", include_body)
        if path == "/dataset-replay/assets/camera_follow_validation_20260929.json":
            return self._file(ROOT / "dataset_replay" / "camera_follow_validation_20260929.json",
                              "application/json", include_body)
        if path in ("/simulation.js", "/simulation.css", "/style.css"):
            return self._file(ROOT / path[1:], mimetypes.guess_type(path)[0] or "text/plain", include_body)
        if path == "/api/catalog":
            try:
                return self._data(self.server.runner.catalog())
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                return self._data({"error": str(exc)}, 503)
        if path == "/api/runs":
            return self._data(self.server.runner.recent_runs())
        match = re.fullmatch(r"/api/runs/([0-9a-f]{12})", path)
        if match:
            try:
                return self._data(self.server.runner.run(match.group(1)))
            except ValueError as exc:
                return self._data({"error": str(exc)}, 404)
        match = re.fullmatch(r"/runs/([0-9a-f]{12})/(video(?:_left_wrist|_right_wrist)?\.mp4|preview\.jpg|joints\.csv|online_adapter\.jsonl|online_audit_summary\.json|grasp_calibration\.jsonl|calibration_summary\.json|gripper_contact_audit\.jsonl|cup_deformation\.jsonl|input_(?:head|left|right)_0000\.jpg)", path)
        if match:
            run_id, filename = match.groups()
            content_type = ("video/mp4" if filename.endswith(".mp4") else
                            "text/csv; charset=utf-8" if filename == "joints.csv" else
                            "image/jpeg" if filename.endswith(".jpg") else "application/json")
            return self._file(self.server.runner.runs_dir / run_id / filename, content_type, include_body)
        return self.send_error(404)

    def do_GET(self) -> None:
        self._get()

    def do_HEAD(self) -> None:
        self._get(include_body=False)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        stop_match = re.fullmatch(r"/api/runs/([0-9a-f]{12})/stop", path)
        if path not in ("/api/runs", "/api/inference/start", "/api/inference/stop") and not stop_match:
            return self.send_error(404)
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        console_port = globals().get('CONSOLE_PORT', 8772)
        if host not in (f"127.0.0.1:{console_port}", f"localhost:{console_port}") or origin not in (
                f"http://127.0.0.1:{console_port}", f"http://localhost:{console_port}"):
            return self._data({"error": "origin not allowed"}, 403)
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._data({"error": "JSON required"}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            if path == "/api/runs":
                return self._data(self.server.runner.start(request), 202)
            if stop_match:
                if request != {}:
                    raise ValueError("expected empty stop request")
                return self._data(self.server.runner.stop_run(stop_match.group(1)), 200)
            if not isinstance(request, dict) or set(request) != {"backend"}:
                raise ValueError("expected backend")
            return self._data(self.server.runner.manage_inference(
                request["backend"], path.rsplit("/", 1)[-1]), 200)
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            return self._data({"error": str(exc)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    console_port = globals().get('CONSOLE_PORT', 8772)
    parser.add_argument("--port", type=int, default=console_port)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost") or args.port != console_port:
        parser.error(f"simulation console must remain on loopback port {console_port}")
    server = SimulationServer((args.host, args.port), SimulationRunner())
    recovered = server.runner.recover_active_run()
    if recovered:
        print(f'Recovered existing simulation {recovered}; no relaunch', flush=True)
    print(f"Isaac simulation console: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
