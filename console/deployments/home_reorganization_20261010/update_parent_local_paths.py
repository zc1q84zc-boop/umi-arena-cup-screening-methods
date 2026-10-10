"""Sync active local deployment scripts with the final Corl_Track_1 parent."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    own = Path(__file__).resolve().parent
    root = own.parents[1]
    audit = own / 'parent_relocation_20261010'
    pattern = re.compile(rb'/home/claude/umi_workspace_zhangchi(?=$|[^A-Za-z0-9_.-])')
    modified = []
    counts = {'python_syntax': 0, 'shell_syntax': 0, 'service_files': 0}
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink() or path.suffix not in {'.py', '.sh', '.service'}:
            continue
        relative = path.relative_to(root)
        if any(p in {'before', 'sim_validation', 'maintenance', '__pycache__',
                     'home_reorganization_20261010'} or p.endswith('_before')
               for p in relative.parts) or '_before_' in path.name:
            continue
        old = path.read_bytes()
        new = pattern.sub(b'/home/claude/Corl_Track_1/umi_workspace_zhangchi', old)
        if old == new:
            continue
        if path.suffix == '.py':
            ast.parse(new.decode(), filename=str(path))
            counts['python_syntax'] += 1
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n'], input=new, check=True, capture_output=True)
            counts['shell_syntax'] += 1
        else:
            counts['service_files'] += 1
        if args.apply:
            backup = audit / 'local_before' / relative
            assert not backup.exists(), f'Backup already exists: {backup}'
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(old)
            path.write_bytes(new)
        modified.append(str(relative))
    result = {'applied': args.apply, 'files': modified, 'verification': counts}
    if args.apply:
        audit.mkdir(parents=True, exist_ok=True)
        (audit / 'local_path_updates.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'applied': args.apply, 'file_count': len(modified), 'verification': counts}, indent=2))


if __name__ == '__main__':
    main()
