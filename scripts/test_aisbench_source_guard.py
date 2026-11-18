#!/usr/bin/env python3
"""Regression tests for the AISBench source structure guard; no imports or network."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from aisbench_source_guard import SourceStructureError, validate_source

ROOT = Path(__file__).resolve().parents[1]
VALID = '''\
class BaseAPIModel:
    async def generate(self):
        for _ in range(self.retry):
            try:
                if self.stream:
                    await self.stream_infer({}, None)
                else:
                    await self.text_infer({}, None)
                break
            except Exception:
                continue
'''


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.base-api-guard-test-', dir=ROOT)
        self.dir = Path(self.temp.name).resolve()
        assert self.dir.parent == ROOT.resolve() and self.dir.name.startswith('.base-api-guard-test-')
        self.path = self.dir / 'fixture.py'

    def tearDown(self):
        self.temp.cleanup()

    def check_rejected(self, code):
        self.path.write_text(code, encoding='utf-8')
        with self.assertRaises(SourceStructureError):
            validate_source(self.path)

    def test_json_file_is_parseable_python_but_rejected(self):
        self.path.write_text(json.dumps([{'artifact': 'records/example.json', 'sha256': '0' * 64}]), encoding='utf-8')
        ast.parse(self.path.read_text(encoding='utf-8'))
        with self.assertRaises(SourceStructureError):
            validate_source(self.path)

    def test_archived_true_source_hash_and_structure(self):
        source = ROOT / 'code/aisbench/base_api.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         '203a416444b972193274a13e1593328a288b6879a3d62e4f201bbb9adfa81755')
        self.assertEqual(validate_source(source)['retry_iterator'], 'range(self.retry)')

    def test_valid_synthetic_structure(self):
        self.path.write_text(VALID, encoding='utf-8')
        self.assertEqual(validate_source(self.path)['class'], 'BaseAPIModel')

    def test_wrong_structure_is_rejected(self):
        for text in (VALID.replace('BaseAPIModel', 'OtherModel'),
                     VALID.replace('range(self.retry)', 'range(3)'),
                     VALID.replace('async def generate', 'async def get_ppl'),
                     VALID.replace('await self.text_infer', 'await self.other_infer'),
                     VALID.replace('except Exception:', 'except ValueError:'),
                     'class BaseAPIModel:\n    async def generate(self):\n        pass\n    async def other(self):\n        for _ in range(self.retry):\n            pass\n',
                     'class BaseAPIModel:\n    async def generate(self):\n        async def nested():\n            for _ in range(self.retry):\n                pass\n'):
            with self.subTest(text=text):
                self.check_rejected(text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
