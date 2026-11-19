#!/usr/bin/env python3
"""Synthetic bounds and invalid-input tests; no metrics or user data."""
import copy
import unittest
from decimal import Decimal
from summarize_hist_bounds_061 import calculate, scope_one


def h(total, groups):
    return {'delta_count': groups[-1][1], 'delta_sum_s': Decimal(str(total)),
            'buckets': [{'upper_s': Decimal(str(x)) if x is not None else None,
                         'cumulative_count': n} for x, n in groups]}


class HistogramBoundsTests(unittest.TestCase):
    def test_finite_open_upper_and_outward_percent(self):
        # Fast: two in (0.5, 1]; slow: three in (1, 3], one in (3, 4].
        result = calculate(h('11', [(0.5, 0), (1, 2), (3, 5), (4, 6), (None, 6)]), Decimal(1))
        self.assertEqual((result['fast_count'], result['slow_count']), (2, 4))
        self.assertEqual(result['slow_sum_lower_s'], '9')
        self.assertFalse(result['slow_sum_lower_exclusive'])
        self.assertEqual(result['slow_sum_upper_s'], '10.0')
        self.assertTrue(result['slow_sum_upper_exclusive'])
        self.assertEqual(result['slow_share_percent_outward_6dp'],
                         {'lower_inclusive': '81.818181', 'upper_inclusive': '90.909091'})

    def test_unknown_fast_lower_and_finite_slow_upper(self):
        # First fast bucket may contain negative values; do not assume zero.
        result = calculate(h('10', [(1, 2), (3, 5), (None, 5)]), Decimal(1))
        self.assertIsNone(result['fast_sum_lower_s'])
        self.assertEqual(result['slow_sum_lower_s'], '8')
        self.assertEqual(result['slow_sum_upper_s'], '9')
        self.assertFalse(result['slow_sum_upper_exclusive'])

    def test_unknown_fast_lower_and_infinite_slow_upper(self):
        result = calculate(h('4', [(1, 1), (None, 2)]), Decimal(1))
        self.assertEqual(result['slow_sum_lower_s'], '3')
        self.assertIsNone(result['slow_sum_upper_s'])
        self.assertIsNone(result['slow_share_percent_outward_6dp']['upper_inclusive'])

    def test_all_slow_and_all_fast(self):
        all_slow = calculate(h('4', [(1, 0), (3, 2), (None, 2)]), Decimal(1))
        self.assertEqual((all_slow['slow_sum_lower_s'], all_slow['slow_sum_upper_s']), ('4', '4'))
        self.assertFalse(all_slow['slow_sum_lower_exclusive'])
        self.assertFalse(all_slow['slow_sum_upper_exclusive'])
        all_fast = calculate(h('1', [(1, 2), (None, 2)]), Decimal(1))
        self.assertEqual((all_fast['slow_sum_lower_s'], all_fast['slow_sum_upper_s']), ('0', '0'))

    def test_lower_bound_strict_when_slow_endpoint_binds(self):
        # Slow > 2 and unknown fast lower: slow lower itself dominates.
        result = calculate(h('1', [(2, 1), (3, 2), (None, 2)]), Decimal(2))
        self.assertEqual(result['slow_sum_lower_s'], '2')
        self.assertTrue(result['slow_sum_lower_exclusive'])
        self.assertEqual(result['slow_sum_upper_s'], '3')

    def test_tied_lower_keeps_open_bound(self):
        # Slow lower 2*2 is open and equals total-fast upper = 6 - 2.
        result = calculate(h('6', [(0, 0), (1, 2), (2, 2), (3, 4), (None, 4)]), Decimal(1))
        self.assertEqual(result['slow_sum_lower_s'], '4')
        self.assertTrue(result['slow_sum_lower_exclusive'])
        self.assertEqual(result['slow_sum_upper_s'], '6')
        self.assertTrue(result['slow_sum_upper_exclusive'])

    def test_invalid(self):
        source = h('11', [(0.5, 0), (1, 2), (3, 5), (4, 6), (None, 6)])
        with self.assertRaises(ValueError):
            calculate(source, Decimal('1.5'))
        bad = copy.deepcopy(source)
        bad['buckets'][2]['cumulative_count'] = 1
        with self.assertRaises(ValueError):
            calculate(bad, Decimal(1))
        bad = copy.deepcopy(source)
        bad['delta_sum_s'] = Decimal('1000')
        with self.assertRaises(ValueError):
            calculate(bad, Decimal(1))
        bad = copy.deepcopy(source)
        bad['buckets'][-1]['upper_s'] = Decimal(5)
        with self.assertRaises(ValueError):
            calculate(bad, Decimal(1))
        bad = copy.deepcopy(source)
        bad['buckets'][1]['upper_s'] = Decimal('NaN')
        with self.assertRaises(ValueError):
            calculate(bad, Decimal(1))
        with self.assertRaises(ValueError):
            scope_one([{'a': 1}, {'a': 1}], a=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
