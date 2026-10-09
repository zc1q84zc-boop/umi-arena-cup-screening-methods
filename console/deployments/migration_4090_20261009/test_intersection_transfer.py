import hashlib
import io
import json
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import intersection_export_readonly as export
import intersection_download_4090 as download


class ManifestTests(unittest.TestCase):
    def manifest(self, name='pi05/30000/params/test', size=1):
        value = {'total_bytes': size, 'files': [
            {'path': name, 'bytes': size, 'sha256': 'a' * 64}]}
        raw = json.dumps(value).encode()
        return raw, hashlib.sha256(raw).hexdigest()

    def test_valid_manifest(self):
        raw, sha = self.manifest()
        self.assertEqual(export.checked_manifest(raw, sha)['total_bytes'], 1)

    def test_changed_manifest(self):
        raw, _ = self.manifest()
        with self.assertRaises(ValueError):
            export.checked_manifest(raw, '0' * 64)

    def test_paths_cannot_escape_or_export_secrets(self):
        for name in ('/pi05/a', 'pi05/../../etc/passwd', 'access.json', 'pi05//a', 'pi05/./a'):
            raw, sha = self.manifest(name)
            with self.subTest(name=name), self.assertRaises(ValueError):
                export.checked_manifest(raw, sha)

    def test_negative_size_rejected(self):
        raw, sha = self.manifest(size=-1)
        with self.assertRaises(ValueError):
            export.checked_manifest(raw, sha)

    def test_mismatched_total_rejected(self):
        raw, _ = self.manifest()
        value = json.loads(raw)
        value['total_bytes'] = 2
        raw = json.dumps(value).encode()
        with self.assertRaises(ValueError):
            export.checked_manifest(raw, hashlib.sha256(raw).hexdigest())

    def test_reuse_excludes_mutable_metadata(self):
        self.assertIsNone(download.reuse_candidate(Path('pi05/30000/inference_export.json')))
        self.assertIsNone(download.reuse_candidate(Path('pi05/30000/provenance/status.json')))
        self.assertIsNone(download.reuse_candidate(Path('openwam/5069/config.yaml')))

    def test_reuse_accepts_regular_weight_but_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(download, 'PI_REUSE', Path(folder)):
            path = Path(folder) / 'params/test'
            path.parent.mkdir()
            path.write_bytes(b'test')
            self.assertEqual(download.reuse_candidate(Path('pi05/30000/params/test')), path)
            link = path.with_name('link')
            link.symlink_to(path)
            self.assertIsNone(download.reuse_candidate(Path('pi05/30000/params/link')))

    def fake_export(self, rows, responder):
        raw = json.dumps({'files': rows, 'total_bytes': sum(row['bytes'] for row in rows)}).encode()
        sha = hashlib.sha256(raw).hexdigest()
        class Response(io.BytesIO):
            def __init__(self, content, status=200):
                super().__init__(content)
                self.status = status
                self.headers = {'Content-Length': str(len(content))}
        def request(req, **_):
            if req.full_url.endswith('/manifest.json'):
                return Response(raw)
            content, status = responder(req)
            return Response(content, status)
        return (mock.patch.object(download, 'urlopen', side_effect=request),
                mock.patch.object(download, 'checked_manifest',
                    side_effect=lambda received: export.checked_manifest(received, sha)))

    def access(self):
        import time
        return {'url': 'http://192.168.110.11:12345', 'token': 'test-only',
                'manifest_sha256': download.MANIFEST_SHA, 'expires_unix': time.time() + 60}

    def test_resume_and_empty_files_hash_before_publish(self):
        content = b'hello'
        rows = [{'path': 'openwam/5069/test', 'bytes': 5, 'sha256': hashlib.sha256(content).hexdigest()},
                {'path': 'source/empty.py', 'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}]
        def remaining(req):
            self.assertEqual(req.get_header('Range'), 'bytes=2-')
            return b'llo', 206
        http, validate = self.fake_export(rows, remaining)
        with tempfile.TemporaryDirectory() as folder, http, validate, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            partial = root / 'openwam/5069/test.part'
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b'he')
            result = download.transfer(self.access(), root)
            self.assertTrue(result['all_sha256_verified'])
            self.assertEqual((root / 'openwam/5069/test').read_bytes(), content)
            self.assertEqual((root / 'source/empty.py').read_bytes(), b'')
            self.assertFalse(partial.exists())

    def test_wrong_checksum_not_published_and_partial_preserved(self):
        rows = [{'path': 'openwam/5069/test', 'bytes': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()}]
        http, validate = self.fake_export(rows, lambda req: (b'bad', 200))
        with tempfile.TemporaryDirectory() as folder, http, validate, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                download.transfer(self.access(), root)
            self.assertFalse((root / 'openwam/5069/test').exists())
            self.assertEqual((root / 'openwam/5069/test.part').read_bytes(), b'bad')

    def test_existing_mismatch_never_overwritten(self):
        rows = [{'path': 'openwam/5069/test', 'bytes': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()}]
        http, validate = self.fake_export(rows, lambda req: self.fail('Should not download'))
        with tempfile.TemporaryDirectory() as folder, http, validate, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            path = root / 'openwam/5069/test'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'old')
            with self.assertRaisesRegex(ValueError, 'will not overwrite'):
                download.transfer(self.access(), root)
            self.assertEqual(path.read_bytes(), b'old')

    def test_range_not_honored_preserves_original_partial(self):
        rows = [{'path': 'openwam/5069/test', 'bytes': 5, 'sha256': hashlib.sha256(b'hello').hexdigest()}]
        http, validate = self.fake_export(rows, lambda req: (b'hello', 200))
        with tempfile.TemporaryDirectory() as folder, http, validate, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            partial = root / 'openwam/5069/test.part'
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b'he')
            with self.assertRaisesRegex(ValueError, 'byte range'):
                download.transfer(self.access(), root)
            self.assertEqual(partial.read_bytes(), b'he')


if __name__ == '__main__':
    unittest.main()
