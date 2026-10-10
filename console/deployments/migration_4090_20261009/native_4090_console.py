#!/usr/bin/env python3
"""Private native console using existing identity-checked 4090 model units.

Legacy policy IDs stay stable; checkpoints/actions/coordinates are not changed.
Only this process's console transport and deployment paths are relocated.
"""
import json
import hashlib
import fcntl
import socket
import sys
from pathlib import Path

import sim_console as app

BASE = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi')
SHARED_HOME = Path('/home/claude')
HOST = 'squirrel_4090_2'
RUNTIME = BASE / 'dual-franka-yubi-isaac-sim-deploy'
MODELS = SHARED_HOME / 'workspace/umi_cup_models_4090_20261009'
INTERSECTION = SHARED_HOME / 'workspace/umi_cup_intersection_models_4090_20261009'
TUNED = app.ROOT / 'simulator_profiles/tuned_v1'
OLD_RUNTIME = str(app.REMOTE_ROOT)
OLD_TUNED = str(app.TUNED_ROOT)
OLD_MODELS = '/home/lrl/workspace/umi_cup_models_5090_20260928'
OLD_INTERSECTION = '/home/lrl/workspace/umi_cup_intersection_models_20261009'
INTERSECTION_KEYS = {'pi05_intersection_rtx5090':'pi05',
                     'openwam_intersection_rtx5090':'openwam'}


def relocate(value):
    if isinstance(value, str):
        return value.replace(OLD_INTERSECTION, str(INTERSECTION)).replace(OLD_RUNTIME, str(RUNTIME)).replace(
            OLD_TUNED, str(TUNED)).replace(OLD_MODELS, str(MODELS)).replace(
            'squirrel RTX 5090', '双 RTX 4090').replace('squirrel_5090', HOST)
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if isinstance(value, dict):
        return {key: relocate(item) for key, item in value.items()}
    return value


def validated_lingbot(backend_id):
    checkpoints = {'lingbot_5000_rtx5090': '5000', 'lingbot_10000_rtx5090': '10000',
                   'lingbot_official_rtx5090': 'official'}
    checkpoint = checkpoints.get(backend_id)
    if not checkpoint:
        return False
    try:
        record = json.loads((app.ROOT / f'deployment_validation/lingbot_{checkpoint}_server.json').read_text())
        paths = {
            'server_sha256': MODELS / 'lingbot/scripts/lingbot_sim_server.py',
            'adapter_sha256': TUNED / 'adapters/lingbot_isaac_online_adapter.py',
            'launcher_sha256': MODELS / f'lingbot/run_lingbot_{checkpoint}.sh',
            'normalization_sha256': MODELS / 'lingbot/norm_stats.json',
        }
        model_id = f'lingbot-cup-clean-{checkpoint}'
        if checkpoint == 'official':
            model_id = 'lingbot-vla2-official-pretrained'
            paths.update({
                'official_adapter_sha256': TUNED / 'adapters/lingbot_official_isaac_online_adapter.py',
                'official_loader_sha256': MODELS / 'lingbot/scripts/official_lingbot_weights.py',
                'official_manifest_sha256': MODELS / 'lingbot/official/hf_ckpt/official_manifest.json',
                'robot_config_sha256': MODELS / 'configs/robot_configs/umi_cup_clean.yaml',
            })
            provenance = record['model_provenance']
            if (provenance.get('repo_id') != 'robbyant/lingbot-vla-v2-6b'
                    or provenance.get('revision') != '11c703bf6a5c1f45b3b69168482da11fdbba53d7'
                    or provenance.get('fine_tuned') is not False
                    or provenance.get('all_files_sha256_verified') is not True):
                return False
        return (record['status'] == 'causal_server_probe_passed'
                and record['hostname'] == 'benyun-workstation' and record['checkpoint'] == checkpoint
                and record['model_id'] == model_id
                and record['model_service_cleanup'] == 'stopped'
                and [step['step'] for step in record['saved_sim_steps']] == [0, 1, 2]
                and all(step['action_rows'] == 3 for step in record['saved_sim_steps'])
                and all(record[key] == hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()))
    except (OSError, KeyError, TypeError, ValueError):
        return False


STEP_KEYS = {'pi05_intersection_10000_rtx5090':'10000', 'pi05_intersection_20000_rtx5090':'20000'}
INTERSECTION_KEYS.update({key:'pi05' for key in STEP_KEYS})
STEP_MANIFEST_SHA = '5f6df930632dc11c139f3287814a535cf1ebd97d9d50e2ecd7ed123283dc9a1b'


def validated_pi05_step(backend_id):
    step=STEP_KEYS[backend_id];model=f'pi05-cup-intersection-{step}'
    try:
        record=json.loads((app.ROOT/f'deployment_validation/intersection_pi05_{step}_server.json').read_text())
        integrity=json.loads((INTERSECTION/f'validation/pi05_{step}_checkpoint_integrity.json').read_text())
        paths={'server_sha256':INTERSECTION/'scripts/intersection_step_server.py',
            'adapter_sha256':TUNED/'adapters/intersection_adapter.py',
            'entry_sha256':TUNED/f'adapters/pi05_intersection_{step}_isaac_online_adapter.py',
            'launcher_sha256':INTERSECTION/f'scripts/run_pi05_intersection_{step}.sh',
            'simulation_launcher_sha256':INTERSECTION/'scripts/run_tuned_intersection_4090.sh'}
        return (record['status']=='ok' and record['model']==model and record['hardware'] in ('RTX4090 GPU0','RTX4090 GPU1')
            and record['verified_requests']==2 and record['model_service_cleanup']=='stopped'
            and record['health']['checkpoint']==str(INTERSECTION/f'pi05/{step}')
            and integrity['model']==model and integrity['all_sha256_verified'] is True
            and integrity['manifest_sha256']==STEP_MANIFEST_SHA
            and all(record[k]==hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()))
    except (OSError,KeyError,TypeError,ValueError):return False


def validated_intersection(backend_id):
    if backend_id in STEP_KEYS:return validated_pi05_step(backend_id)
    family = INTERSECTION_KEYS.get(backend_id)
    if not family:
        return False
    try:
        record = json.loads((app.ROOT/f'deployment_validation/intersection_{family}_server.json').read_text())
        paths = {
            'server_sha256':INTERSECTION/'scripts/intersection_server.py',
            'adapter_sha256':TUNED/'adapters/intersection_adapter.py',
            'entry_sha256':TUNED/f'adapters/{family}_intersection_isaac_online_adapter.py',
            'launcher_sha256':INTERSECTION/f'scripts/run_{family}_intersection_{"30000" if family=="pi05" else "5069"}.sh',
            'simulation_launcher_sha256':INTERSECTION/'scripts/run_tuned_intersection_4090.sh',
        }
        if family == 'openwam':
            paths.update({
                'cpu_offload_wrapper_sha256':INTERSECTION/'scripts/intersection_4090_cpu_text_server.py',
                'cpu_text_helper_sha256':MODELS/'openwam/scripts/openwam_cpu_text_encoder_4090.py',
            })
        model = {'pi05':'pi05-cup-intersection-30000','openwam':'openwam-cup-intersection-fullpass-5069'}[family]
        integrity=json.loads((INTERSECTION/f'validation/{family}_checkpoint_integrity.json').read_text())
        return (record['status']=='ok' and record['model']==model
            and record['hardware']=='RTX4090 GPU1' and record['verified_requests']==2
            and record['model_service_cleanup']=='stopped'
            and integrity['model']==model and integrity['all_sha256_verified'] is True
            and integrity['manifest_sha256']=='f4b34ab3fa3236a5a1b0cec8b42005b8faa9e902721ff17a718051790a4e0662'
            and all(record[key]==hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()))
    except (OSError,KeyError,TypeError,ValueError):
        return False


def deployment_backend_status(backend_id, status):
    if backend_id in INTERSECTION_KEYS and not validated_intersection(backend_id):
        return {**status,'deploy_ready':False,'ready':False,
                'reason':'交集权重/4090 推理接口尚待验证，保持停用'}
    if 'lingbot' not in backend_id or validated_lingbot(backend_id):
        return status
    return {**status, 'deploy_ready': False, 'weights_installed': True,
            'migration_validation': 'adapter_pending',
            'reason': '4090 LingBot 训练权重已校验；运行适配器尚待验证，保持停用'}


def check_service_ownership(runner, selected_backend=None):
    owned = getattr(runner, '_4090_owned_model_pids', {})
    for key in app.INFERENCE_BACKENDS:
        state = runner._service_state(key)
        if state['active'] and owned.get(key) != state['pid']:
            raise ValueError('4090 有其他控制台或任务启动的模型服务；不会接管或停止它')


def configure():
    if socket.gethostname() != 'benyun-workstation' or app.ROOT != BASE / 'umi-track1-console-4090-20261009':
        raise RuntimeError('4090 native console must run in its verified dedicated directory')
    app.REMOTE_HOST = HOST
    app.REMOTE_ROOT = RUNTIME
    app.TUNED_ROOT = TUNED
    app.GPU_INDEX = 0
    app.CONSOLE_PORT = 8774
    app.DEFAULT_CONTACT_PROFILE = app._PVC_NUMERICS['PRECISION_ID']
    app.DEFAULT_POLICY_ID = 'pi05-cup-intersection-30000'
    app.DEFAULT_RUN_MODE = 'until_success'
    transport = str(app.ROOT / 'native_4090_transport.py')
    app.SSH = (sys.executable, transport, 'command')
    app.SCP = (sys.executable, transport, 'copy')
    for key,step in STEP_KEYS.items():
        entry=dict(app.INFERENCE_BACKENDS['pi05_intersection_rtx5090'])
        port=18864 if step=='10000' else 18865
        entry.update(label=f'π0.5 交集版 {int(step)//1000}k', checkpoint=OLD_INTERSECTION+f'/pi05/{step}',
            model_id=f'pi05-cup-intersection-{step}',port=port,adapter_url=f'http://127.0.0.1:{port}/infer',
            launcher_script=f'run_pi05_intersection_{step}.sh')
        app.INFERENCE_BACKENDS[key]=entry
    app.INFERENCE_BACKENDS = {
        key: {**relocate(backend), 'gpu': 1}
        for key, backend in app.INFERENCE_BACKENDS.items()
        if backend['host'] == 'squirrel_5090'
    }
    for key,family in INTERSECTION_KEYS.items():
        backend=app.INFERENCE_BACKENDS[key]
        backend['unit']=f'umi-intersection-{family}-{"30000" if family=="pi05" else "5069"}-4090-console.service'
        backend.pop('checkpoint_name',None)
        backend['process_signature']=('intersection_server.py' if family=='pi05'
                                      else 'intersection_4090_cpu_text_server.py')
    for key,step in STEP_KEYS.items():
        app.INFERENCE_BACKENDS[key]['unit']=f'umi-intersection-pi05-{step}-4090-console.service'
        app.INFERENCE_BACKENDS[key]['process_signature']='intersection_step_server.py'
    original_policies = app.SimulationRunner.policies

    def policies(self):
        entries = relocate(original_policies(self))
        for entry in entries:
            if entry['id'] in ('pi05-cup-intersection-10000','pi05-cup-intersection-20000','pi05-cup-intersection-30000','openwam-cup-intersection-fullpass-5069'):
                entry['ready']=any(validated_intersection(key) for key in entry['supported_backends'])
                entry['remote_launcher']=str(INTERSECTION/'scripts/run_tuned_intersection_4090.sh')
                entry['reason']='' if entry['ready'] else '新交集版权重及目标 GPU 接口验证未完成'
            # Every LingBot variant needs its own target-host serving audit.
            if entry.get('online_inference') and (
                    not set(entry.get('supported_backends', [])) & app.INFERENCE_BACKENDS.keys()
                    or ('lingbot' in entry['id'] and not any(validated_lingbot(key)
                        for key in entry.get('supported_backends', [])))):
                entry['ready'] = False
                entry['reason'] = '4090 权重/运行适配器尚待验证；不能视为在线可用'
            if entry.get('ready') and entry.get('online_inference') and entry.get('simulator_profile') != 'tuned_online_v1':
                entry['ready'] = False
                entry['reason'] = '历史诊断保留；4090 本次仅启用隔离 tuned_online_v1 基线'
            if entry.get('remote_script') and entry.get('simulator_profile') == 'tuned_online_v1':
                if 'intersection' not in entry['id']:
                    entry['remote_launcher'] = str(TUNED / 'run_tuned_online.sh')
            if entry.get('online_inference') and entry.get('ready'):
                weights = ('已校验的官方原始预训练权重' if entry['id'] == 'lingbot-vla2-official-pretrained'
                           else '已安装校验的训练 checkpoint')
                entry['description'] = ('双 RTX 4090 · ' + weights + '；GPU0 仿真、GPU1 推理。'
                    'tuned_online_v1 保留随动双腕图像、CAD 开度、双指驱动及 30Hz 连续限速。'
                    '目标机各版本的短测记录见运行列表；短测不代表抓取或完整任务成功，标定仍未实测。')
            elif entry.get('online_inference') and entry.get('reason'):
                entry['description'] = '本次迁移未启用：' + entry['reason']
            entry['deployment_host'] = HOST
        return entries

    app.SimulationRunner.policies = policies
    original_backend_status = app.SimulationRunner.backend_status

    def backend_status(self, backend_id):
        return deployment_backend_status(backend_id, original_backend_status(self, backend_id))

    app.SimulationRunner.backend_status = backend_status
    original_manage = app.SimulationRunner.manage_inference

    def manage(self, backend_id, action):
        if action=='start' and backend_id in INTERSECTION_KEYS and not validated_intersection(backend_id):
            raise ValueError('交集版 4090 校验尚未通过，保持停用')
        if action == 'start' and 'lingbot' in backend_id and not validated_lingbot(backend_id):
            raise ValueError('LingBot 4090 适配器尚未验证，保持停用')
        # A second console may be using this host. Never alter inference while
        # any simulator is active on GPU0, including another dedicated unit.
        status = self.gpu_status()
        if not status.get('available'):
            raise ValueError('4090 GPU0 有正在运行的仿真；不会改变任何模型服务')
        check_service_ownership(self, backend_id)
        return original_manage(self, backend_id, action)

    app.SimulationRunner.manage_inference = manage
    original_gpu_status = app.SimulationRunner.gpu_status

    def gpu_status(self):
        result = original_gpu_status(self)
        result['reason'] = ('4090 GPU0 空闲' if result['available']
                            else '4090 GPU0 被占用或不可达；不会抢占')
        return result

    app.SimulationRunner.gpu_status = gpu_status
    original_start_backend = app.SimulationRunner._start_backend_locked

    def start_backend(self, backend_id):
        if backend_id in INTERSECTION_KEYS and not validated_intersection(backend_id):
            raise ValueError('交集版 4090 校验尚未通过，保持停用')
        if 'lingbot' in backend_id and not validated_lingbot(backend_id):
            raise ValueError('LingBot 4090 适配器尚未验证，保持停用')
        if not self.gpu_status().get('available'):
            raise ValueError('4090 GPU0 有其他仿真；不会启动或停止模型服务')
        with (RUNTIME / 'console.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('4090 公共运行锁被占用；不会改变任何模型服务')
            check_service_ownership(self, backend_id)
            result = original_start_backend(self, backend_id)
            owned = getattr(self, '_4090_owned_model_pids', {})
            owned[backend_id] = self._service_state(backend_id)['pid']
            self._4090_owned_model_pids = owned
            return result

    app.SimulationRunner._start_backend_locked = start_backend
    original_catalog = app.SimulationRunner.catalog

    def catalog(self):
        result = original_catalog(self)
        return {**result, 'host': HOST, 'gpu': 0, 'inference_gpu': 1,
                'console_execution': 'native_on_4090',
                'deployment_scope': 'seven trained checkpoints and official LingBot; readiness requires target-host validation'}

    app.SimulationRunner.catalog = catalog


if __name__ == '__main__':
    configure()
    app.main()
