#!/usr/bin/env python3
"""Synthetic histogram and timestamp parser validation; no private data."""
import unittest
import summarize_stage_snapshots_056 as s


def metric(count,total,a,b,created='10'):
 p='vllm:request_queue_time_seconds';label='model_name="test"'
 return '\n'.join((f'process_start_time_seconds 5',f'{p}_count{{{label}}} {count}',f'{p}_sum{{{label}}} {total}',f'{p}_created{{{label}}} {created}',f'{p}_bucket{{{label},le="1"}} {a}',f'{p}_bucket{{{label},le="+Inf"}} {b}'))


class Tests(unittest.TestCase):
 def test_bucket_and_open_infinity(self):
  rows,proc,labels=s.between(metric(10,5,5,10),metric(12,7,6,12),('vllm:request_queue_time_seconds',),'p',False);self.assertEqual(proc,'same_at_endpoints');self.assertIsNone(labels);r=rows[0];self.assertEqual(r['delta_count'],2);self.assertEqual(r['mean_s'],1);self.assertEqual(r['nearest_rank_quantile_bounds']['50'],{'lower_exclusive_s':None,'upper_inclusive_s':1.0});self.assertEqual(r['nearest_rank_quantile_bounds']['90'],{'lower_exclusive_s':1.0,'upper_inclusive_s':None})
 def test_no_forced_reset_or_missing(self):
  for newer in (metric(9,4,4,9),metric(12,7,6,12,'11'),metric(12,7,7,11)):
   with self.assertRaises(ValueError):s.between(metric(10,5,5,10),newer,('vllm:request_queue_time_seconds',),'p',False)
 def test_labels_and_clock(self):
  self.assertEqual(s.labels('a="x",b="y"'),{'a':'x','b':'y'})
  with self.assertRaises(ValueError):s.labels('a="x",a="y"')
  with self.assertRaises(ValueError):s.duration('2020-01-01T00:00:00','2020-01-01T00:00:01')


if __name__=='__main__':unittest.main(verbosity=2)
