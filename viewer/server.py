#!/usr/bin/env python3
"""Local, SSH-backed review viewer. Only selected episodes are cached locally."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = "/mnt/data/benyun/workspace/umi_cup_review_20260923/clean_cup_v1/extracted_bad_episodes"
DEFAULT_PYTHON = "/home/benyun/.venvs/umi_arena_pi05/bin/python"
VIEWS = {"left", "right", "center"}
EPISODE_RE = re.compile(r"[0-9]+")
VIDEO_RE = re.compile(r"/video/([0-9]+)/(left|right|center)\.mp4")

CATALOG_CODE = r"""
import csv,json
items=[]
with (ROOT.parent/'episode_quality_labels.csv').open(newline='') as stream:
    for row in csv.DictReader(stream):
        if row['decision']!='quarantine':
            continue
        episode=int(row['episode_index']);frames=int(row['frames'])
        if not (ROOT/f'episode_{episode:06d}'/'extraction.json').is_file():
            raise ValueError('Missing extracted episode')
        items.append({'id':episode,'frames':frames,
                      'flags':list(filter(None,row['machine_flags'].split(';'))),
                      'candidate_frames':len(list(filter(None,row['candidate_frames'].split(';')))),
                      'human_status':'confirmed_anomaly' if row['human_confirmed_jump']=='True' else 'not_reviewed'})
items.sort(key=lambda item:item['id'])
if len(items)!=552 or sum(item['frames'] for item in items)!=80409:
    raise ValueError('Expected 552 episodes and 80,409 frames')
print(json.dumps({'episodes':items,'total_frames':80409,'fps':30},allow_nan=False))
"""

EPISODE_CODE = r"""
import json
from pathlib import Path
import pyarrow.parquet as pq
folder=ROOT/f'episode_{EPISODE:06d}'
annotation=json.loads((folder/'annotation.json').read_text())
extraction=json.loads((folder/'extraction.json').read_text())
columns=['frame_index','timestamp','observation.joint_states','action.joint_states',
         'observation.pose.left_hand_root.absolute','observation.pose.right_hand_root.absolute',
         'action.pose.left_hand_root.relative','action.pose.right_hand_root.relative']
rows=pq.read_table(folder/'motion.parquet',columns=columns).to_pylist()
rows.sort(key=lambda row:row['frame_index'])
frames=int(extraction['motion_rows'])
if len(rows)!=frames or [row['frame_index'] for row in rows]!=list(range(frames)):
    raise ValueError('Missing or duplicate motion rows')
scan=annotation.get('machine_scan') or {}
human=annotation.get('human_review') or {}
print(json.dumps({
  'id':EPISODE,'fps':30,'frames':frames,
  'times':[row['timestamp'] for row in rows],
  'poses':[[row['observation.pose.left_hand_root.absolute'],
            row['observation.pose.right_hand_root.absolute']] for row in rows],
  'grippers':[row['observation.joint_states'] for row in rows],
  'action_grippers':[row['action.joint_states'] for row in rows],
  'action_poses':[[row['action.pose.left_hand_root.relative'],
                   row['action.pose.right_hand_root.relative']] for row in rows],
  'flags':scan.get('flags_for_review_only',[]),
  'events':scan.get('candidate_events',[]),
  'human_status':human.get('review_status','not_reviewed'),
  'human_label':human.get('episode_review_label',''),
  'human_frames':human.get('confirmed_frames',[])},allow_nan=False,separators=(',',':')))
"""


class Source:
    def __init__(self, host: str, root: str, remote_python: str, cache: Path):
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host):
            raise ValueError("SSH host must be an alias, hostname, or user@hostname")
        self.host = host
        self.root = root
        self.remote_python = remote_python
        self.cache = cache.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.catalog = self._remote_json(CATALOG_CODE)
        self.index = {int(item["id"]): item for item in self.catalog["episodes"]}

    def _remote_json(self, script: str, episode: int | None = None) -> dict:
        prefix = "from pathlib import Path\nROOT=Path(" + repr(self.root) + ")\n"
        if episode is not None:
            prefix += f"EPISODE={episode}\n"
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   self.host, shlex.quote(self.remote_python) + " -"]
        result = subprocess.run(command, input=prefix + script, text=True,
                                capture_output=True, timeout=120, check=False)
        if result.returncode:
            raise RuntimeError("A100 read failed: " + result.stderr[-500:])
        return json.loads(result.stdout)

    def _folder(self, episode: int) -> Path:
        if episode not in self.index:
            raise ValueError("Unknown episode")
        folder = self.cache / f"episode_{episode:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def episode(self, episode: int) -> dict:
        folder = self._folder(episode)
        target = folder / "trajectory.json"
        with self.lock:
            if target.is_file():
                data = json.loads(target.read_text())
            else:
                data = self._remote_json(EPISODE_CODE, episode)
                if data["id"] != episode or data["frames"] != self.index[episode]["frames"]:
                    raise ValueError("Remote episode mismatch")
                temporary = folder / "trajectory.json.part"
                try:
                    temporary.write_text(json.dumps(data, separators=(",", ":")))
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        return data

    def video(self, episode: int, view: str) -> Path:
        if view not in VIEWS:
            raise ValueError("Unknown camera")
        folder = self._folder(episode)
        target = folder / f"video_{view}.mp4"
        with self.lock:
            if not target.is_file():
                temporary = folder / f"video_{view}.mp4.part"
                remote = f"{self.host}:{self.root}/episode_{episode:06d}/video_{view}.mp4"
                try:
                    result = subprocess.run(["scp", "-q", "-o", "BatchMode=yes",
                                             "-o", "ConnectTimeout=10", remote, str(temporary)],
                                            capture_output=True, timeout=240, check=False)
                    if result.returncode or not temporary.is_file() or not temporary.stat().st_size:
                        raise RuntimeError("A100 video transfer failed")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        return target


class Handler(BaseHTTPRequestHandler):
    source: Source

    def headers_common(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")

    def send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.headers_common(content_type, len(data))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: dict) -> None:
        self.send_bytes(json.dumps(data, separators=(",", ":"), allow_nan=False).encode(),
                        "application/json; charset=utf-8")

    def send_video(self, file: Path) -> None:
        size = file.stat().st_size
        header = self.headers.get("Range")
        start, end = 0, size - 1
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            lo, hi = match.groups()
            start = int(lo) if lo else max(0, size - int(hi))
            end = min(int(hi), size - 1) if lo and hi else size - 1
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if header else 200)
        self.headers_common("video/mp4", end - start + 1)
        self.send_header("Accept-Ranges", "bytes")
        if header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with file.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        path = urlsplit(self.path)
        try:
            if path.path == "/api/catalog":
                self.send_json(self.source.catalog)
            elif path.path == "/api/episode":
                values = parse_qs(path.query).get("id", [])
                if len(values) != 1 or not EPISODE_RE.fullmatch(values[0]):
                    raise ValueError("Invalid episode id")
                self.send_json(self.source.episode(int(values[0])))
            elif match := VIDEO_RE.fullmatch(path.path):
                self.send_video(self.source.video(int(match[1]), match[2]))
            elif path.path in ("/", "/app.js", "/style.css"):
                name = "index.html" if path.path == "/" else path.path[1:]
                kind = "text/html; charset=utf-8" if name.endswith(".html") else (
                    "text/javascript; charset=utf-8" if name.endswith(".js") else "text/css; charset=utf-8")
                self.send_bytes((HERE / name).read_bytes(), kind)
            else:
                self.send_error(404)
        except ValueError:
            self.send_error(400, "Invalid request")
        except Exception as error:
            print(f"Viewer error: {error}", file=sys.stderr)
            self.send_error(500, "Data unavailable")

    def log_message(self, _format: str, *_args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="openwam-a100")
    parser.add_argument("--remote-root", default=DEFAULT_ROOT)
    parser.add_argument("--remote-python", default=DEFAULT_PYTHON)
    parser.add_argument("--cache-dir", type=Path, default=HERE.parent / ".cache" / "review_viewer")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    source = Source(args.ssh_host, args.remote_root, args.remote_python, args.cache_dir)
    Handler.source = source
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"Review viewer: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    print(f"Remote catalog: {len(source.index)} episodes. Cache: {source.cache}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
