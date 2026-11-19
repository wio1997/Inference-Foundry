#!/usr/bin/env python3
"""Archive and synthetic bounded-sample tests; never reads original device traces."""
import copy
import json
import unittest
from pathlib import Path

import timeline_036 as subject


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.sg = json.loads((subject.DIR / 'sglang_rank0.json').read_text())
        self.v = json.loads((subject.DIR / 'vllm_rank0.json').read_text())

    def test_counts_rank_and_quantized_zero(self):
        a, b = subject.validate(self.sg, 'sglang'), subject.validate(self.v, 'vllm')
        self.assertEqual((a['events'], a['anchors'], a['rounded_zero_durations']), (366, 276, 35))
        self.assertEqual((b['events'], b['anchors'], b['rounded_zero_durations']), (48, 48, 0))
        self.assertTrue(self.sg['source_positive_duration_filter'])
        self.assertEqual(self.sg['timing_precision_ms'], .001)
        self.assertEqual(sum(row[6] for row in self.sg['events']), 2)
        self.assertEqual(sum(row[7] for row in self.sg['events']), 1)

    def test_invalid_sample_or_private_fields_rejected(self):
        for case in ('rank', 'extra', 'category', 'stream', 'bounds', 'nonfinite', 'whitelist', 'limit', 'precision', 'trace'):
            d = copy.deepcopy(self.sg)
            if case == 'rank': d['rank'] = 1
            elif case == 'extra': d['device_pid'] = 1
            elif case == 'category': d['events'][0][0] = 'not-allowed'
            elif case == 'stream': d['events'][0][1] = 'not-numeric'
            elif case == 'bounds': d['events'][0][4] = 300
            elif case == 'nonfinite': d['events'][0][3] = float('nan')
            elif case == 'whitelist': d['event_names']['sdma'] = 'not-allowed'
            elif case == 'limit': d['window']['limit'] = 9999
            elif case == 'precision': d['source_positive_duration_filter'] = False
            else: d['trace_sha256'] = '0' * 64
            with self.subTest(case=case), self.assertRaises(ValueError):
                subject.validate(d, 'sglang')

    def test_static_html_has_two_independent_axes_and_safe_fields(self):
        subject.validate(self.sg, 'sglang'); subject.validate(self.v, 'vllm')
        page = subject.render([self.sg, self.v])
        self.assertEqual(page.count('<svg'), 2)
        self.assertEqual(page.count('<title>'), 1 + 366 + 48)
        self.assertIn('各框架独立归零', page)
        self.assertIn('不能累加为墙钟或根据数量排名', page)
        self.assertIn('source duration &gt; 0', page)
        for forbidden in ('/' + 'home' + '/', '/' + 'data' + '/', '/' + 'task' + '/', 'device_pid', 'request_text', 'prediction'):
            self.assertNotIn(forbidden, page)
        self.assertEqual(page, subject.render([self.sg, self.v]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
