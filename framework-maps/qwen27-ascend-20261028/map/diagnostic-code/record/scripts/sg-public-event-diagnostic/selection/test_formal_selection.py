"""Reject request-ID ambiguity and mixed/wrong-rank replay substitutions."""
import unittest
from filter_formal_replays import select_formal_replays as select
class Tests(unittest.TestCase):
 def fixture(self):
  outer=[{'rank_id':rank,'invocation':f'r{rank}','side':side,'work_key':{'requests':[{'rid':'formal'}]}}
   for rank in range(4)for side in ['start','end']]
  observer=[{'event':'sg_request_id_link','external_request_id':'client','rid':'formal'}]
  return outer,observer,{'requests':[{'request_id':'client'}]}
 def test_unique_ids_keep_startup_separate(self):
  outer,obs,manifest=self.fixture();startup={'rank_id':0,'invocation':'startup','side':'start','work_key':{'requests':[{'rid':'startup'}]}}
  selected,excluded,receipt=select(outer+[startup],obs,manifest)
  self.assertEqual(len(selected),8);self.assertEqual(excluded,[startup]);self.assertEqual(receipt['formal_pairs_per_rank'],{0:1,1:1,2:1,3:1})
 def test_ambiguous_external_id_rejected(self):
  outer,obs,manifest=self.fixture();obs.append({**obs[0],'rid':'other'})
  with self.assertRaises(ValueError):select(outer,obs,manifest)
 def test_mixed_request_replay_rejected(self):
  outer,obs,manifest=self.fixture();outer[0]['work_key']['requests'].append({'rid':'startup'})
  with self.assertRaises(ValueError):select(outer,obs,manifest)
 def test_missing_rank_or_duplicate_pair_rejected(self):
  outer,obs,manifest=self.fixture()
  for bad in [outer[:-2],outer+[outer[0]]]:
   with self.subTest(rows=len(bad)),self.assertRaises(ValueError):select(bad,obs,manifest)
if __name__=='__main__':unittest.main()
