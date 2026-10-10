#!/usr/bin/env python3
"""Native-only transport for the isolated 4090 console; no SSH credentials."""
import shutil
import subprocess
import sys
from pathlib import Path

HOST = 'squirrel_4090_2'
RUNTIME = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/dual-franka-yubi-isaac-sim-deploy')
CONSOLE = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009')


def execute(arguments):
    if len(arguments) != 3:
        return 2
    mode, source, destination = arguments
    if mode == 'command':
        if source != HOST:
            return 2
        return subprocess.call(['bash', '-lc', destination])
    if mode != 'copy' or not source.startswith(HOST + ':'):
        return 2
    original = Path(source.split(':', 1)[1])
    target = Path(destination)
    if ('..' in original.parts or '..' in target.parts
            or not original.is_relative_to(RUNTIME / 'runs')
            or not target.is_relative_to(CONSOLE / 'sim_runs')):
        return 2
    if not original.is_file():
        return 1
    shutil.copy2(original, target)
    return 0


if __name__ == '__main__':
    raise SystemExit(execute(sys.argv[1:]))
