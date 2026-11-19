#!/usr/bin/env python3
"""Synthetic scope coverage and metric invariants; no live sources."""
import copy,unittest
import summarize_vllm_formal_059 as s


def fixture():
 rows=[]
 for n in (32768,65536):
  for phase in ('cold','hot'):
   for role in ('prefill','decode'):
    mean=28.238173 if n==32768 else 80.198494
    h=[{'delta_count':32,'metric':'vllm:request_queue_time_seconds','mean_s':mean}]
    h.extend({'delta_count':32,'metric':'vllm:synthetic_'+str(i),'mean_s':1} for i in range(4))
    rows.append({'input_tokens':n,'phase':phase,'role':role,'snapshot_elapsed_s':1,'histograms':h})
 return {'results':rows}


class Tests(unittest.TestCase):
 def test_coverage(self):s.checks(fixture())
 def test_duplicate_scope(self):
  x=fixture();x['results'][1]=copy.deepcopy(x['results'][0])
  with self.assertRaises(ValueError):s.checks(x)
 def test_count_and_expected_value(self):
  for action in ('count','mean','missing'):
   x=fixture()
   if action=='count':x['results'][0]['histograms'][0]['delta_count']=31
   elif action=='mean':x['results'][0]['histograms'][0]['mean_s']=2
   else:x['results'][0]['histograms'].pop()
   with self.subTest(action=action),self.assertRaises(ValueError):s.checks(x)
 def test_recursive_reference(self):
  self.assertTrue(s.contains({'a':[{'b':'known'}]},'known'));self.assertFalse(s.contains({'a':'other'},'known'))


if __name__=='__main__':unittest.main(verbosity=2)
