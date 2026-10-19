"""Boundary tests for the public artifact checksum verifier."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from verify_public import verify

class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        (self.root / 'payload.txt').write_bytes(b'public payload\n')
        self.entry = {'path': 'payload.txt', 'bytes': 15,
                      'sha256': hashlib.sha256(b'public payload\n').hexdigest()}
        self.manifest([self.entry])
    def manifest(self, entries):
        (self.root / 'PUBLIC_MANIFEST.json').write_text(json.dumps({'schema_version': 1, 'files': entries}))
    def test_valid_payload(self):
        self.assertEqual(verify(self.root), [])
    def test_detects_same_length_tamper(self):
        (self.root / 'payload.txt').write_bytes(b'PUBLIC payload\n')
        self.assertTrue(any('hash mismatch' in x for x in verify(self.root)))
    def test_detects_missing_and_extra(self):
        (self.root / 'payload.txt').unlink()
        (self.root / 'extra.txt').write_text('unexpected')
        errors = verify(self.root)
        self.assertTrue(any('missing' in x for x in errors))
        self.assertTrue(any('unexpected' in x for x in errors))
    def test_rejects_parent_escape(self):
        self.manifest([{**self.entry, 'path': '../payload.txt'}])
        self.assertTrue(any('unsafe path' in x for x in verify(self.root)))
    def test_rejects_symlink(self):
        (self.root / 'payload.txt').unlink()
        (self.root / 'payload.txt').symlink_to('PUBLIC_MANIFEST.json')
        self.assertTrue(any('symlink' in x for x in verify(self.root)))
    def test_rejects_duplicate_manifest_entry(self):
        self.manifest([self.entry, self.entry])
        self.assertTrue(any('duplicate' in x for x in verify(self.root)))
    def test_allows_local_official_inputs(self):
        inputs = self.root / 'code' / '附件'
        inputs.mkdir(parents=True)
        (inputs / '附件1.xlsx').write_bytes(b'local only')
        self.assertEqual(verify(self.root), [])

if __name__ == '__main__':
    unittest.main(verbosity=2)
