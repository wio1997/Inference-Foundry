#!/usr/bin/env python3
"""Synthetic validation cases only; no real traces or service access."""
import copy,unittest
import check_p_timeline_061 as s


def fixture():
 caps=[]
 for fw,sha in s.TRACES.items():
  caps.append({'framework':fw,'rank':0,'trace_sha256':sha,'selected_event_span_ms':1500.0,'bin_width_ms':1000,'bins':[{'bin':0,'category':'kda','event_count':4,'duration_sum_ms_nonwallclock':1.0},{'bin':1,'category':'kda','event_count':None,'duration_sum_ms_nonwallclock':None}]})
 return {'schema':'glm53-p-selected-events-two-independent-spans-v1','captures':caps}


class Tests(unittest.TestCase):
 def test_valid(self):self.assertTrue(s.validate(fixture()))
 def test_duplicate(self):
  d=fixture();d['captures'][0]['bins'].append(copy.deepcopy(d['captures'][0]['bins'][0]))
  with self.assertRaises(ValueError):s.validate(d)
 def test_small_or_nonfinite(self):
  for case in ('small','nan','bounds','suppressed'):
   d=fixture();x=d['captures'][0]['bins'][0]
   if case=='small':x['event_count']=2
   if case=='nan':x['duration_sum_ms_nonwallclock']=float('nan')
   if case=='bounds':x['bin']=2
   if case=='suppressed':x['event_count']=None
   with self.subTest(case=case),self.assertRaises(ValueError):s.validate(d)


if __name__=='__main__':unittest.main(verbosity=2)
