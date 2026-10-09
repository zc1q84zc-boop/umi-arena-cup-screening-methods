"""Temporary authenticated LAN source for the public official model files."""
import hmac
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

config = json.loads(Path(__file__).with_name('transfer_config.json').read_text())
root = Path(config['root'])
allowed = set(config['manifest']['files'])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if not hmac.compare_digest(self.headers.get('Authorization', ''),
                                   'Bearer ' + config['token']):
            self.send_error(403)
            return
        name = self.path.removeprefix('/')
        if name not in allowed:
            self.send_error(404)
            return
        path = root / name
        size = path.stat().st_size
        offset = 0
        requested_range = self.headers.get('Range')
        if requested_range:
            match = re.fullmatch(r'bytes=(\d+)-', requested_range)
            if not match or int(match.group(1)) >= size:
                self.send_error(416)
                return
            offset = int(match.group(1))
        self.send_response(206 if requested_range else 200)
        self.send_header('Content-Length', str(size - offset))
        self.send_header('Accept-Ranges', 'bytes')
        if requested_range:
            self.send_header('Content-Range', f'bytes {offset}-{size-1}/{size}')
        self.end_headers()
        try:
            with path.open('rb') as f:
                f.seek(offset)
                while block := f.read(8 * 1024 * 1024):
                    self.wfile.write(block)
        except (BrokenPipeError, ConnectionResetError):
            pass


server = ThreadingHTTPServer(('192.168.110.11', 18919), Handler)
Path(__file__).with_name('transfer_source.pid').write_text(str(os.getpid()))
print('OFFICIAL_TRANSFER_SOURCE_READY', os.getpid(), flush=True)
server.timeout = 5
deadline = time.monotonic() + 7200
while time.monotonic() < deadline:
    server.handle_request()
