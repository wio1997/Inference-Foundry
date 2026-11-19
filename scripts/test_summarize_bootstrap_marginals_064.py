#!/usr/bin/env python3
"""Synthetic marginal-bucket comparisons; no request or metrics data."""
import copy,unittest
import summarize_bootstrap_marginals_064 as s


def h(counts):
 cumul=0;bins=[]
 for edge,n in zip((.001,.029282,1.,None),counts):
  cumul+=n;bins.append({'upper_s':edge,'cumulative_count':cumul})
 total=counts[0]*0.0005+counts[1]*0.015+counts[2]*0.5+counts[3]*2.0
 return {'delta_count':cumul,'delta_sum_s':total,'mean_s':total/cumul if cumul else 0,'buckets':bins}


class Tests(unittest.TestCase):
 def test_different_bins_and_equal_slow(self):
  r=s.compare(h([1,2,4,0]),h([2,1,4,0]));self.assertEqual(r['p_above_threshold'],4);self.assertTrue(r['each_above_threshold_bin_count_equal']);self.assertEqual(r['same_size_hypothetical_bijection_min_different_bin_pairs'],1)
 def test_unequal_count_not_a_bijection(self):
  r=s.compare(h([1,3,4,0]),h([2,1,4,0]));self.assertIsNone(r['same_size_hypothetical_bijection_min_different_bin_pairs'])
 def test_bad_histogram_rejected(self):
  original=h([1,2,4,0])
  for kind in ('counts','bounds','inf','threshold'):
   x=copy.deepcopy(original)
   if kind=='counts':x['buckets'][1]['cumulative_count']=0
   if kind=='bounds':x['buckets'][1]['upper_s']=.0009
   if kind=='inf':x['buckets'][-1]['upper_s']=10
   if kind=='threshold':x['buckets'][1]['upper_s']=.04
   with self.subTest(kind=kind),self.assertRaises(ValueError):s.compare(x,original)


if __name__=='__main__':unittest.main(verbosity=2)
