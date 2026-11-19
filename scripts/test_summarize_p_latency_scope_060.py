#!/usr/bin/env python3
"""Synthetic scope checks; no log or user request access."""
import copy,unittest
import summarize_p_latency_scope_060 as s


def fixture():
 rows=[]
 for n in (32768,65536):
  for phase in ('cold','hot'):
   a=65.150760 if n==32768 and phase=='cold' else 1.0
   b=40.595361 if n==32768 and phase=='cold' else 1.0
   rows.append({'input_tokens':n,'phase':phase,'sg_prefill_e2e_streaming':{'delta_count':32,'mean_s':a},'vllm_prefill':{'e2e':{'delta_count':32,'mean_s':b},'e2e_and_ttft_same_count_and_sum':True,'common_bucket_boundary_count':8,'common_bucket_cumulative_deltas_equal':True}})
 return {'results':rows}


class Tests(unittest.TestCase):
 def test_valid(self):s.checks(fixture())
 def test_missing_or_duplicate(self):
  x=fixture();x['results'][1]=copy.deepcopy(x['results'][0])
  with self.assertRaises(ValueError):s.checks(x)
 def test_count_or_sum_mismatch(self):
  for kind in ('count','sum','mean'):
   x=fixture()
   if kind=='count':x['results'][0]['sg_prefill_e2e_streaming']['delta_count']=31
   if kind=='sum':x['results'][0]['vllm_prefill']['e2e_and_ttft_same_count_and_sum']=False
   if kind=='mean':x['results'][0]['vllm_prefill']['e2e']['mean_s']=0
   with self.subTest(kind=kind),self.assertRaises(ValueError):s.checks(x)


if __name__=='__main__':unittest.main(verbosity=2)
