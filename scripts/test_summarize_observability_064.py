#!/usr/bin/env python3
"""Synthetic observability whitelist tests; no service bodies or identifiers."""
import unittest
import summarize_observability_064 as s


def flags():return {'enable_metrics':True,'enable_request_time_stats_logging':False,'enable_trace':False,'export_metrics_to_file':False,'log_requests':False,'log_requests_level':2}


class Tests(unittest.TestCase):
 def test_two_identical_values(self):self.assertEqual(s.values({'a':flags(),'b':[flags()]}),flags())
 def test_missing_wrong_type_and_conflict(self):
  for kind in ('missing','type','conflict'):
   a,b=flags(),flags()
   if kind=='missing':del a['enable_trace'];del b['enable_trace']
   if kind=='type':a['enable_metrics']=1
   if kind=='conflict':b['enable_metrics']=False
   with self.subTest(kind=kind),self.assertRaises(ValueError):s.values({'a':a,'b':b})


if __name__=='__main__':unittest.main(verbosity=2)
