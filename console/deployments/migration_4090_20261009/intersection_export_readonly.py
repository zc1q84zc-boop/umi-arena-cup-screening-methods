"""Bounded, authenticated, manifest-only A100 -> 4090 file transfer.

Reads an existing immutable export; never imports a model or rewrites weights.
Only this session's private credentials are created, then removed on shutdown.
"""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import threading
import time
from urllib.parse import unquote, urlsplit
import secrets

EXPORT = Path('/mnt/data/benyun/workspace/umi_cup_intersection_export_20261009')
MANIFEST_SHA = 'f4b34ab3fa3236a5a1b0cec8b42005b8faa9e902721ff17a718051790a4e0662'
MAPPING_SHA = 'b98a7acb146ed3ea7abba10d86e37089c9178ea24da27310851248798dfb8910'
ALLOWED_ROOTS = (
    Path('/mnt/data/benyun/workspace/pi05_cup_intersection_20261008'),
    Path('/mnt/data/benyun/workspace/openwam_cup_fullpass_20261009'),
    Path('/mnt/data/benyun/workspace/openwam_charger_full_20260922/OpenWAM'),
)


def checked_manifest(raw, expected_sha=MANIFEST_SHA):
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Frozen manifest checksum mismatch')
    value = json.loads(raw)
    names = set()
    for row in value['files']:
        path = PurePosixPath(row['path'])
        if (path.is_absolute() or '..' in path.parts or str(path) != row['path']
                or not path.parts or path.parts[0] not in ('pi05', 'openwam', 'source')
                or row['path'] in names):
            raise ValueError('Invalid or duplicate manifest path')
        if type(row['bytes']) is not int or row['bytes'] < 0:
            raise ValueError('Invalid size')
        if not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid digest')
        names.add(row['path'])
    if sum(row['bytes'] for row in value['files']) != value['total_bytes']:
        raise ValueError('Manifest total mismatch')
    return value


def frozen_sources():
    raw = (EXPORT / 'manifest.json').read_bytes()
    manifest = checked_manifest(raw)
    mapping_raw = (EXPORT / 'source_files.json').read_bytes()
    if hashlib.sha256(mapping_raw).hexdigest() != MAPPING_SHA:
        raise ValueError('Frozen source map checksum mismatch')
    rows = json.loads(mapping_raw)
    if [{k: v for k, v in row.items() if k != 'source'} for row in rows] != manifest['files']:
        raise ValueError('Source map and manifest disagree')
    mapping = {}
    for row in rows:
        path = Path(row['source']).resolve(strict=True)
        if (path != EXPORT / 'inference_export.json'
                and not any(path.is_relative_to(root) for root in ALLOWED_ROOTS)):
            raise ValueError('Source outside authorized roots')
        if not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError('Source missing or changed size')
        mapping[row['path']] = (path, row['bytes'])
    return raw, manifest, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session-dir', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'kemove-SYS-420GP-TNAR':
        raise RuntimeError('Not the verified A100 source')
    session = args.session_dir.resolve(strict=True)
    if session.parent != Path('/tmp') or not session.name.startswith('umi-intersection-to-4090.'):
        raise ValueError('Not a dedicated transfer session')
    if session.stat().st_uid != os.getuid() or session.stat().st_mode & 0o077:
        raise ValueError('Transfer session must be owned and private')
    raw, manifest, mapping = frozen_sources()
    token = secrets.token_urlsafe(32)
    gate = threading.BoundedSemaphore(1)
    rate_bytes = 64 * 1024 * 1024

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.send_error(403)
                return
            name = unquote(urlsplit(self.path).path).lstrip('/')
            if name == 'manifest.json':
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if name not in mapping:
                self.send_error(404)
                return
            if not gate.acquire(blocking=False):
                self.send_error(503)
                return
            try:
                path, expected_size = mapping[name]
                with path.open('rb') as stream:
                    if os.fstat(stream.fileno()).st_size != expected_size:
                        self.send_error(409)
                        return
                    offset = 0
                    header = self.headers.get('Range')
                    if header:
                        match = re.fullmatch(r'bytes=(\d+)-', header)
                        if not match or not 0 <= int(match[1]) < expected_size:
                            self.send_error(416)
                            return
                        offset = int(match[1])
                    self.send_response(206 if header else 200)
                    self.send_header('Content-Length', str(expected_size - offset))
                    if header:
                        self.send_header('Content-Range', f'bytes {offset}-{expected_size - 1}/{expected_size}')
                    self.end_headers()
                    stream.seek(offset)
                    sent = 0
                    started = time.monotonic()
                    while block := stream.read(min(1024 * 1024, expected_size - offset - sent)):
                        self.wfile.write(block)
                        sent += len(block)
                        delay = sent / rate_bytes - (time.monotonic() - started)
                        if delay > 0:
                            time.sleep(delay)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                gate.release()

    server = ThreadingHTTPServer(('192.168.110.11', 0), Handler)
    server.daemon_threads = True
    access_path = session / 'access.json'
    access = {'url': f'http://192.168.110.11:{server.server_port}', 'token': token,
              'pid': os.getpid(), 'manifest_sha256': MANIFEST_SHA, 'expires_unix': time.time() + 7200}
    with os.fdopen(os.open(access_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        json.dump(access, stream)
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    timer = threading.Timer(7200, stop)
    timer.daemon = True
    timer.start()
    print(json.dumps({'status': 'serving_readonly', 'pid': os.getpid(),
                      'files': len(manifest['files']), 'bytes': manifest['total_bytes'],
                      'manifest_sha256': MANIFEST_SHA, 'max_MiB_s': 64}), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        timer.cancel()
        server.server_close()
        access_path.unlink(missing_ok=True)
        print(json.dumps({'status': 'stopped', 'credentials_removed': True}), flush=True)


if __name__ == '__main__':
    main()
