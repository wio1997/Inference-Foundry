import unittest
from deepinfer.npu_ops.cp910c import QuerySlice
class Layout(unittest.TestCase):
 def assert_layout(self,q,s,world):
  rows=[QuerySlice.build(q,s,world,i) for i in range(world)]
  global_positions=[p for n,end in zip(q,s) for p in range(end-n,end)]
  for row in rows:
   actual=[p for n,end in zip(row.query_lengths,row.seq_lengths) for p in range(end-n,end)]
   self.assertEqual(actual,global_positions[row.start:row.end])
  self.assertEqual(sum(sum(x.query_lengths) for x in rows),sum(q))
  return rows
 def test_wide(self):
  rows=self.assert_layout([3328,3328,3200,3328],[32768]*4,8)
  self.assertLessEqual(max(sum(x.query_lengths) for x in rows),1648)
 def test_request_crossing(self):
  rows=self.assert_layout([5,7,3],[17,30,9],3)
  self.assertEqual(rows[2].request_indices,(1,2))
  self.assertEqual(rows[2].seq_lengths,(30,9))
 def test_partial_c2(self):
  rows=self.assert_layout([7],[32769],4)
  self.assertEqual([x.seq_lengths for x in rows],[(32764,),(32766,),(32768,),(32769,)])
 def test_empty_ranks(self):
  rows=self.assert_layout([1],[1],8)
  self.assertEqual([len(x.query_lengths) for x in rows],[1,0,0,0,0,0,0,0])
 def test_ragged_prefix(self):
  self.assert_layout([4,1,9,2],[1025,29440,16385,3],8)
unittest.main()
