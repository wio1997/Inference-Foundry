#!/usr/bin/env python3
"""Synthetic-stream and existing-archive regression tests, no trace download."""
import collections
import hashlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

import analyze_trace_030 as subject

ROOT = Path(__file__).resolve().parents[1]


def stream(events):
    raw = json.dumps(events, ensure_ascii=False, separators=(',', ':')).encode()
    return raw, hashlib.sha256(raw).hexdigest()


def sample():
    # Metadata follows events intentionally; request-like text must never be emitted.
    return [
        {'ph': 'X', 'pid': 7, 'tid': 11, 'name': subject.GMM, 'ts': 10, 'dur': 10},
        {'ph': 'X', 'pid': 7, 'tid': 11, 'name': subject.GMM, 'ts': 25, 'dur': 20},
        {'ph': 'X', 'pid': 8, 'tid': 12, 'cat': 'cpu_op', 'name': 'step[VERIFY private text]', 'ts': 0, 'dur': 50},
        {'ph': 'X', 'pid': 8, 'tid': 12, 'cat': 'cpu_op', 'name': subject.GMM, 'ts': 5, 'dur': 99},
        {'ph': 'X', 'pid': 7, 'tid': 11, 'name': 'sensitive request text', 'ts': 1, 'dur': 2},
        {'ph': 'M', 'pid': 7, 'name': 'process_name', 'args': {'name': 'Ascend Hardware'}},
        {'ph': 'M', 'pid': 7, 'name': 'process_labels', 'args': {'labels': 'NPU 0'}}]


class TraceTests(unittest.TestCase):
    def test_streamed_aggregates_are_sanitized(self):
        raw, digest = stream(sample())
        with mock.patch.object(subject, 'CHUNK', 7):
            out = subject.analyze_stream(io.BytesIO(raw), digest, 0, 'sg')
        self.assertEqual(out['trace_sha256'], digest)
        self.assertEqual(out['cpu_step_window_count'], 1)
        self.assertEqual(len(out['groups']), 1)
        group = out['groups'][0]
        self.assertEqual(group['count'], 2)
        self.assertEqual(group['median_us'], 15)
        self.assertEqual(group['event_union_us'], 30)
        self.assertEqual(group['event_starts_in_cpu_step_window'], 2)
        self.assertNotIn('sensitive', json.dumps(out))
        self.assertNotIn('private text', json.dumps(out))
        self.assertNotIn('ts', group)

    def test_hash_rank_format_and_bounds_reject(self):
        raw, digest = stream(sample())
        with self.assertRaises(ValueError):
            subject.analyze_stream(io.BytesIO(raw), '0' * 64, 0, 'sg')
        with self.assertRaises(ValueError):
            subject.analyze_stream(io.BytesIO(raw), digest, 1, 'sg')
        with self.assertRaises(ValueError):
            subject.analyze_stream(io.BytesIO(raw), digest, 0, 'sg', max_bytes=len(raw))
        for bad in (b'{}', b'[{},]', b'[{},{] ', b'[]junk', b'['):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                subject.analyze_stream(io.BytesIO(bad), hashlib.sha256(bad).hexdigest(), 0, 'sg')

    def test_missing_events_not_counted_as_zero(self):
        raw, digest = stream(sample())
        out = subject.analyze_stream(io.BytesIO(raw), digest, 0, 'sg')
        self.assertEqual([x['name'] for x in out['groups']], [subject.GMM])
        self.assertEqual(subject.quantile([1, 2, 3, 4], .9), 3.7)
        self.assertEqual(subject.union([(0, 10), (5, 15), (20, 21)]), 16)

    def test_published_summaries_match_compact_identity(self):
        for engine in ('sg', 'vllm'):
            for rank in (0, 1):
                with self.subTest(engine=engine, rank=rank):
                    d = json.loads((ROOT / f'evidence/trace_030/{engine}_rank{rank}.json').read_text())
                    base = ('evidence/matched_SG_G181/profile_G240/CPU_d/review_g252' if engine == 'sg'
                            else 'evidence/G4_profile_G138/CPU_d/review')
                    compact = json.loads((ROOT / f'{base}/rank_{rank}.json').read_text())
                    self.assertEqual((d['engine'], d['rank'], d['trace_sha256']),
                                     (engine, rank, compact['trace_sha256']))
                    ops = ({x['name']: (x['count'], x['sum_us']) for x in compact['top50_hardware']}
                           if engine == 'sg' else
                           {x['key']: (x['events'], x['sum_duration_us']) for x in compact['analysis']['operators']})
                    for group in d['groups']:
                        self.assertIn(group['name'], subject.ALLOWLIST)
                        count, total = ops[group['name']]
                        self.assertEqual(group['count'], count)
                        self.assertAlmostEqual(group['sum_us_overlapping_not_wallclock'], total, places=6)
                    if engine == 'vllm':
                        old = compact['raw_layout_proof']['raw_files']
                        self.assertEqual(len(d['raw_frozen_checks']), 13 if rank == 0 else 12)
                        self.assertEqual(collections.Counter((x['sha256'], x['bytes']) for x in old),
                                         collections.Counter((x['sha256'], x['bytes']) for x in d['raw_frozen_checks']))
                    else:
                        self.assertEqual(d['raw_frozen_checks'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
