#!/usr/bin/env python3
"""Apply the reviewed source overlay to a clean upstream simulator checkout."""

from pathlib import Path
import shutil
import subprocess
import sys


BASE_COMMIT = "609e6da"
SOURCE = Path(__file__).resolve().parent / "files"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 apply_overlay.py /path/to/clean/simulator-checkout")
    target = Path(sys.argv[1]).resolve()
    if not (target / "yubi_isaac_sim_env" / "run.py").is_file():
        raise SystemExit("target is not the expected simulator checkout")
    if git(target, "rev-parse", "--short=7", "HEAD") != BASE_COMMIT:
        raise SystemExit(f"checkout commit {BASE_COMMIT} before applying this overlay")
    if git(target, "status", "--porcelain"):
        raise SystemExit("target checkout has changes; use a clean checkout")
    files = sorted(path for path in SOURCE.rglob("*")
                   if path.is_file() and "__pycache__" not in path.parts
                   and path.suffix != ".pyc")
    for source in files:
        relative = source.relative_to(SOURCE)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"Applied {len(files)} source files to {target}")
    print("Provision licensed Panda assets, then rebuild and validate the USD.")


if __name__ == "__main__":
    main()
