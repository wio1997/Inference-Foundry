#!/usr/bin/env python3
"""Check conditional block arithmetic and sampled-count logic without a live service."""
import copy
import json
import unittest

import reconstruct_034 as subject


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads(subject.DEFAULT.read_text())

    def test_reported_device_bound_arithmetic(self):
        r = subject.reconstruct(self.doc)
        self.assertEqual(r['arithmetic']['device_bound_denominator_unique'], 428)
        self.assertEqual(2060 * 263296 // 428, 1267265)
        self.assertFalse(r['arithmetic']['device_bound_assumption_proven'])
        self.assertFalse(r['arithmetic']['actual_per_request_blocks_verified'])
        self.assertEqual(r['arithmetic']['illustrative_2060_times_640_not_group_aware'], 1318400)
        self.assertEqual((r['observed']['cold']['sample_count'], r['observed']['hot']['sample_count']), (119, 115))
        self.assertEqual((r['observed']['cold']['sampled_running_max'], r['observed']['hot']['sampled_running_max']), (1, 1))

    def test_integer_interval_matches_exhaustive_small_cases(self):
        for blocks in range(1, 12):
            for length in range(1, 12):
                for reported in range(1, 12):
                    actual = [d for d in range(1, blocks * length + 1) if blocks * length // d == reported]
                    expected = subject.candidate_interval(blocks, length, reported)
                    self.assertEqual(expected, (min(actual), max(actual)) if actual else None)
        for invalid in ((0, 10, 1), (1, -1, 1), (1, 1, 0), (True, 1, 1)):
            with self.assertRaises(ValueError):
                subject.candidate_interval(*invalid)

    def test_sampled_max_not_inferred_from_open_bucket(self):
        d = copy.deepcopy(self.doc)
        cold = d['periodic_d_scheduler_samples']['cold']
        cold['running_zero'] -= 1
        cold['running_at_least_two'] += 1
        self.assertIsNone(subject.reconstruct(d)['observed']['cold']['sampled_running_max'])
        cold['samples'] += 1
        with self.assertRaises(ValueError):
            subject.reconstruct(d)


if __name__ == '__main__':
    unittest.main(verbosity=2)
