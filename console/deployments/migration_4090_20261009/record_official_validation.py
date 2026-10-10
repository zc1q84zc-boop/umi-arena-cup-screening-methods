"""Record a completed native official serving probe after its service stops."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess

assert socket.gethostname() == 'benyun-workstation'
console = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009')
models = Path('/home/claude/workspace/umi_cup_models_4090_20261009')
audit = console / 'deployment_validation/official_20261009'
record = json.loads((audit / 'server_probe.json').read_text())
assert record['status'] == 'causal_server_probe_passed'
assert record['expected_model'] == 'lingbot-vla2-official-pretrained'
assert record['model_provenance']['fine_tuned'] is False
assert record['model_provenance']['all_files_sha256_verified'] is True
state = subprocess.run(['systemctl', '--user', 'show', 'lingbot-official-squirrel.service',
                        '-p', 'ActiveState', '-p', 'MainPID'], capture_output=True, text=True, check=True).stdout
assert 'ActiveState=inactive' in state and 'MainPID=0' in state
load = json.loads((models / 'lingbot/official/official_load_audit.json').read_text())
assert load['inference_parameters_loaded'] > 0 and load['inference_parameters_missing'] == 0
assert load['all_inference_parameters_from_official_snapshot'] is True
assert load['checkpoint_files_modified'] is False
paths = {
    'server_sha256': models / 'lingbot/scripts/lingbot_sim_server.py',
    'adapter_sha256': console / 'simulator_profiles/tuned_v1/adapters/lingbot_isaac_online_adapter.py',
    'launcher_sha256': models / 'lingbot/run_lingbot_official.sh',
    'normalization_sha256': models / 'lingbot/norm_stats.json',
    'official_adapter_sha256': console / 'simulator_profiles/tuned_v1/adapters/lingbot_official_isaac_online_adapter.py',
    'official_loader_sha256': models / 'lingbot/scripts/official_lingbot_weights.py',
    'official_manifest_sha256': models / 'lingbot/official/hf_ckpt/official_manifest.json',
    'robot_config_sha256': models / 'configs/robot_configs/umi_cup_clean.yaml',
}
record.update(hostname=socket.gethostname(), checkpoint='official',
              model_id='lingbot-vla2-official-pretrained', model_service_cleanup='stopped',
              source_run='364e3bbf1886', official_load_audit=load,
              **{key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()})
(console / 'deployment_validation/lingbot_official_server.json').write_text(json.dumps(record, indent=2)+'\n')
print('OFFICIAL_TARGET_VALIDATION_RECORDED')
