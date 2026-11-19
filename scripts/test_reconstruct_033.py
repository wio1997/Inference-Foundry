#!/usr/bin/env python3
"""Synthetic and archived-data regression tests; no trace or service access."""
import copy
import json
import unittest
from pathlib import Path

import reconstruct_033 as subject

ROOT = Path(__file__).resolve().parents[1]


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads(subject.DEFAULT.read_text())

    def test_two_recorded_windows(self):
        result = subject.reconstruct(self.doc)
        a = result['windows']['partial']['d_conditional_bootstrap_minus_handshake']
        b = result['windows']['to_completion']['d_conditional_bootstrap_minus_handshake']
        self.assertEqual((a['whole_count'], a['handshake_count']), (4, 4))
        self.assertEqual((b['whole_count'], b['handshake_count']), (8, 8))
        self.assertAlmostEqual(a['conditional_mean_seconds'], 1102.5117529698182)
        self.assertAlmostEqual(b['conditional_mean_seconds'], 1173.2982267262414)
        self.assertIn('do not prove remote network handshake', result['legacy_field_note'])
        self.assertFalse(a['same_sample_identity_verified'])
        self.assertFalse(b['same_sample_identity_verified'])
        self.assertEqual(result, subject.reconstruct(self.doc))

    def test_mismatched_count_is_not_paired(self):
        d = copy.deepcopy(self.doc)
        d['snapshots']['d'][1]['series']['kv_transfer_bootstrap_ms']['count'] += 1
        pair = subject.reconstruct(d)['windows']['partial']['d_conditional_bootstrap_minus_handshake']
        self.assertFalse(pair['same_count'])
        self.assertIsNone(pair['conditional_mean_seconds'])
        self.assertFalse(pair['same_sample_identity_verified'])

    def test_invalid_series_are_rejected(self):
        for kind in ('reset', 'rank', 'foreign_field', 'zero_count_sum_change', 'nonfinite'):
            d = copy.deepcopy(self.doc)
            snap = d['snapshots']['d'][1]
            if kind == 'reset':
                snap['series']['decode_bootstrap']['count'] = 0
            elif kind == 'rank':
                snap['rank'] = 1
            elif kind == 'foreign_field':
                snap['host'] = 'not-allowed'
            elif kind == 'zero_count_sum_change':
                snap['series']['decode_bootstrap']['count'] = d['snapshots']['d'][0]['series']['decode_bootstrap']['count']
            else:
                snap['series']['decode_bootstrap']['sum'] = float('nan')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                subject.reconstruct(d)

    def test_terminal_scope_and_health(self):
        summary = json.loads((ROOT / 'evidence/diagnostic_033/terminal_summary.json').read_text())
        self.assertEqual((summary['workload']['request_count'], summary['workload']['success_count'],
                          summary['workload']['input_tokens_per_request'], summary['workload']['output_tokens_per_request']),
                         (20, 20, 262144, 32))
        self.assertEqual((summary['health']['samples'], summary['health']['p_timeout']), (114, 1))
        self.assertEqual(summary['router']['observed_http_503'], 0)
        self.assertFalse(summary['usage_source']['independent_raw_sse_usage_frames_archived'])
        self.assertFalse(summary['separately_verified_inner_exit_code'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
