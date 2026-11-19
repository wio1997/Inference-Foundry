#!/usr/bin/env python3
"""D rank0 archive tests; no raw trace, device or network access."""
import copy
import json
import unittest
from html.parser import HTMLParser

import timeline_037 as subject


class Counts(HTMLParser):
    def __init__(self):
        super().__init__(); self.tags = {}

    def handle_starttag(self, tag, attrs):
        self.tags[tag] = self.tags.get(tag, 0) + 1


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.sg = json.loads((subject.DIR / 'sglang_rank0.json').read_text())
        self.v = json.loads((subject.DIR / 'vllm_rank0.json').read_text())

    def test_source_scopes_axis_and_rounding(self):
        a, b = subject.validate(self.sg, 'sglang'), subject.validate(self.v, 'vllm')
        self.assertEqual((a['events'], a['bin_ms'], a['axis_end_ms']), (599, 100, 100.645))
        self.assertEqual((b['events'], b['bin_ms'], b['axis_end_ms']), (569, 100, 100.0))
        self.assertEqual((a['rounded_zero_durations'], b['rounded_zero_durations']), (0, 0))
        self.assertFalse(self.sg['shape_available']); self.assertFalse(self.v['shape_available'])
        self.assertEqual(len(self.v['frozen_component_checks']), 13)
        self.assertFalse(self.v['source_checks']['analysis_copy_sdk_conversion_replayed'])
        self.assertEqual(self.sg['timing_precision_ms'], .001)

    def test_bounds_privacy_and_identity_are_enforced(self):
        for case in ('rank', 'pid', 'category', 'stream', 'start', 'duration', 'nonfinite', 'axis', 'bin', 'downsample', 'trace', 'rounding', 'shape', 'component'):
            d = copy.deepcopy(self.v if case == 'component' else self.sg)
            fw = 'vllm' if case == 'component' else 'sglang'
            if case == 'rank': d['rank'] = 1
            elif case == 'pid': d['device_pid'] = 1
            elif case == 'category': d['events'][0][0] = 'not-allowlisted'
            elif case == 'stream': d['events'][0][1] = 'not-numeric'
            elif case == 'start': d['events'][0][2] = 101
            elif case == 'duration': d['events'][0][3] = 1000
            elif case == 'nonfinite': d['events'][0][3] = float('nan')
            elif case == 'axis': d['window']['axis_end_ms'] = 100
            elif case == 'bin': d['window']['bin_ms'] = 250
            elif case == 'downsample': d['window']['uniformly_downsampled_to_1000'] = True
            elif case == 'trace': d['trace_sha256'] = '0' * 64
            elif case == 'rounding': d['source_positive_duration_filter'] = False
            elif case == 'shape': d['shape_available'] = True
            else: d['frozen_component_checks'][0]['sha256'] = '0' * 64
            with self.subTest(case=case), self.assertRaises(ValueError):
                subject.validate(d, fw)

    def test_two_axes_22_ticks_1168_bars_and_no_private_data(self):
        subject.validate(self.sg, 'sglang'); subject.validate(self.v, 'vllm')
        page = subject.render([self.sg, self.v]); parser = Counts(); parser.feed(page)
        self.assertEqual(parser.tags.get('svg'), 2)
        self.assertEqual(parser.tags.get('rect'), 1168)
        self.assertEqual(parser.tags.get('line'), 22)
        self.assertEqual(parser.tags.get('title'), 1169)
        self.assertIn('100.645 ms', page); self.assertIn('100.000 ms', page)
        self.assertIn('独立归零', page); self.assertIn('不能累加为墙钟或根据数量排名', page)
        for forbidden in ('/' + 'home' + '/', '/' + 'data' + '/', '/' + 'task' + '/', 'device_pid', 'request_text', 'prediction'):
            self.assertNotIn(forbidden, page)
        self.assertEqual(page, subject.render([self.sg, self.v]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
