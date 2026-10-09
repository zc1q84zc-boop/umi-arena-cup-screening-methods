import hashlib
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock


class ResumeTest(unittest.TestCase):
    def execute_copy(self, status=206, corrupt=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'download_official.py'
            script.write_text(Path(__file__).with_name(script.name).read_text())
            data = b'official unchanged bytes'
            manifest = {'revision': 'test', 'files': {'model': {
                'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}
            (root/'transfer_config.json').write_text(json.dumps({
                'manifest': manifest, 'target': str(root/'target'),
                'source': 'http://localhost', 'token': 'test'}))
            (root/'target').mkdir()
            prefix = 8
            (root/'target/model.partial').write_bytes(data[:prefix])
            response = io.BytesIO(b'x' * (len(data)-prefix) if corrupt else data[prefix:])
            response.status = status
            response.headers = {'Content-Range': f'bytes {prefix}-{len(data)-1}/{len(data)}'}
            response.fp = SimpleNamespace(raw=SimpleNamespace(_sock=Mock()))
            with patch('urllib.request.urlopen', return_value=response) as request:
                runpy.run_path(str(script))
                self.assertEqual(request.call_args.args[0].get_header('Range'), 'bytes=8-')
            self.assertEqual((root/'target/model').read_bytes(), data)
            self.assertTrue(json.loads((root/'target/official_manifest.json').read_text())['verified'])

    def test_resume_preserves_existing_prefix(self):
        self.execute_copy()

    def test_ignored_range_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'byte range'):
            self.execute_copy(status=200)

    def test_bad_hash_cannot_mark_complete(self):
        with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
            self.execute_copy(corrupt=True)


if __name__ == '__main__':
    unittest.main()
