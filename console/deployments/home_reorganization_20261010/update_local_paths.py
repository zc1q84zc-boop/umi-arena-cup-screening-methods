"""Keep current local 4090 deployment scripts aligned with their new workspace."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--apply', action='store_true'); args = parser.parse_args()
    own = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('umi_path_relocation', own/'reorganize_4090.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = own.parents[1]
    modified = []
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink() or path.suffix not in {'.py','.sh','.service'}:
            continue
        relative = path.relative_to(root)
        if any(p in {'before','sim_validation','maintenance','__pycache__','home_reorganization_20261010'} or p.endswith('_before') for p in relative.parts) or '_before_' in path.name:
            continue
        old = path.read_bytes(); new = module.rewrite(old)
        if relative == Path('deployments/migration_4090_20261009/native_4090_console.py'):
            expected = b"BASE = Path('/home/claude')"
            if expected in new:
                new = new.replace(expected, ("BASE = Path('"+str(module.DEST)+"')\nSHARED_HOME = Path('/home/claude')").encode())
                new = new.replace(b"MODELS = BASE / 'workspace/", b"MODELS = SHARED_HOME / 'workspace/")
                new = new.replace(b"INTERSECTION = BASE / 'workspace/", b"INTERSECTION = SHARED_HOME / 'workspace/")
        if new != old:
            if args.apply:
                backup = own/'local_before'/relative
                backup.parent.mkdir(parents=True, exist_ok=True); backup.write_bytes(old)
                path.write_bytes(new)
            modified.append(str(relative))
    (own/'local_path_updates.json').write_text(json.dumps({'applied':args.apply,'files':modified},indent=2)+'\n')
    print(json.dumps({'applied':args.apply,'file_count':len(modified),'files':modified},indent=2))


if __name__ == '__main__':main()
