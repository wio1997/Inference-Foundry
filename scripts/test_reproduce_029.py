#!/usr/bin/env python3
"""Offline regression checks for document 029; no file or device writes."""
import json
import unittest

import reproduce_029 as subject


class ReproductionTests(unittest.TestCase):
    def test_archive_reproduction_and_no_request_payload(self):
        out = subject.analyze()
        self.assertEqual(out['diagnostic_rows'], 16)
        self.assertEqual(out['diagnostic_decompressed_sha256'], '6e1da60b597d8fb978cc9c7549c720a9de2a8f0b9af956fc73bd9e110f4eaae8')
        for item in out['formal'].values():
            self.assertEqual(item['matched'], 32)
            self.assertEqual(item['ttft_ms_sg_minus_vllm']['positive'], 32)
        self.assertAlmostEqual(out['formal']['32768_cold']['ttft_ms_sg_minus_vllm']['median'], 26938.187920255587)
        self.assertEqual(out['profile_32k_rank_events']['sg_p']['kda']['counts'], [272])
        self.assertEqual(out['profile_32k_rank_events']['vllm_d']['indexer']['counts'], [96])
        # No request data, per-request hash, or individual timestamps are published.
        encoded = json.dumps(out)
        for field in ('input_sha256', 'request_witness', 'time_points', 'prediction'):
            self.assertNotIn(field, encoded)
        self.assertEqual(out, subject.analyze())

    def test_identity_and_input_mismatch_are_rejected(self):
        a = [{'id': 1, 'input_sha256': 'one', 'ttft_ms': 3, 'post_first_ms': 2, 'e2e_ms': 5}]
        b = [{'id': 1, 'input_sha256': 'one', 'ttft_ms': 1, 'post_first_ms': 1, 'e2e_ms': 2}]
        self.assertEqual(subject.compare_rows(a, b)['ttft_ms_sg_minus_vllm']['median'], 2)
        with self.assertRaises(ValueError):
            subject.compare_rows(a, [dict(b[0], input_sha256='different')])
        with self.assertRaises(ValueError):
            subject.compare_rows(a + a, b)
        with self.assertRaises(ValueError):
            subject.compare_rows(a, [dict(b[0], id=2)])

    def test_missing_compact_is_not_zero(self):
        self.assertIsNone(subject.rank_event({'top50_hardware': []}, 'sg', 'missing'))
        self.assertIsNone(subject.rank_event({'analysis': {'operators': []}}, 'vllm', 'missing'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
