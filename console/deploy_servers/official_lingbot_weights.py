"""Verify and load unchanged Robbyant pretrained weights for inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OFFICIAL_REPO = 'robbyant/lingbot-vla-v2-6b'
OFFICIAL_REVISION = '11c703bf6a5c1f45b3b69168482da11fdbba53d7'
AUXILIARY_PREFIXES = (
    'model.depth_align_head.', 'model.future_depth_align_head.',
    'model.current_video_align_head.', 'model.future_video_align_head.',
    'model.future_shared_task_proj.', 'model.current_shared_task_proj.',
)
AUXILIARY_TENSORS = frozenset((
    'model.depth_align_embs', 'model.future_depth_align_embs',
    'model.future_video_align_embs', 'model.current_video_align_embs',
))


def verify_official_checkpoint(checkpoint: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('repo_id') != OFFICIAL_REPO
            or manifest.get('revision') != OFFICIAL_REVISION
            or manifest.get('fine_tuned') is not False):
        raise ValueError('Unrecognized official pretrained provenance')
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid official manifest path')
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != expected['size']:
            raise ValueError('Official file missing or wrong size: ' + name)
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected['sha256']:
            raise ValueError('Official file SHA-256 mismatch: ' + name)
    return {'weights_origin': 'official_pretrained_unmodified',
            'repo_id': OFFICIAL_REPO, 'revision': OFFICIAL_REVISION,
            'fine_tuned': False, 'all_files_sha256_verified': True,
            'runtime_dtype': 'bfloat16',
            'interface_profile': 'umi_cup_clean_existing_coordinate_and_normalization_adapter'}


def official_server_class(native_class):
    class OfficialPretrainedServer(native_class):
        def load_model_weights(self, path_to_pi_model, strict=True):
            import torch
            from safetensors import safe_open

            root = Path(path_to_pi_model)
            index = json.loads((root / 'model.safetensors.index.json').read_text())
            weight_map = index['weight_map']
            expected = set(self.vla.state_dict())
            missing = expected - set(weight_map)
            extra = set(weight_map) - expected
            if missing or any(key not in AUXILIARY_TENSORS and not key.startswith(AUXILIARY_PREFIXES)
                              for key in extra):
                raise ValueError(f'Official inference architecture mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
            # The upstream inference architecture omits loss-only projection
            # heads. Every parameter it DOES instantiate must come from the
            # official snapshot. Shard loading keeps CPU memory bounded.
            self.vla.to(dtype=torch.bfloat16)
            loaded = set()
            for name in sorted(set(weight_map.values())):
                with safe_open(str(root / name), framework='pt', device='cpu') as f:
                    part = {key: f.get_tensor(key) for key in f.keys() if key in expected}
                    result = self.vla.load_state_dict(part, strict=False)
                    if result.unexpected_keys or loaded.intersection(part):
                        raise ValueError('Unexpected or duplicate official inference parameter')
                    loaded.update(part)
                del part
            if loaded != expected:
                raise ValueError('Official inference parameter coverage incomplete')
            self.official_load_audit = {
                'inference_parameters_loaded': len(loaded),
                'inference_parameters_missing': 0,
                'unused_training_projection_keys': sorted(extra),
                'all_inference_parameters_from_official_snapshot': True,
                'checkpoint_files_modified': False,
            }
            (root.parent / 'official_load_audit.json').write_text(
                json.dumps(self.official_load_audit, indent=2) + '\n')
            print('OFFICIAL_INFERENCE_WEIGHTS_LOADED', len(loaded),
                  'unused_loss_only_keys', len(extra), flush=True)

    return OfficialPretrainedServer
