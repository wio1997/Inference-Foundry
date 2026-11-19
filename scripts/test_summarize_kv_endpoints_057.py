#!/usr/bin/env python3
"""Synthetic, isolated tests for rank selection and gauge fail-closed behavior."""
import unittest
import summarize_kv_endpoints_057 as s


def body():
 label='engine_type="decode",model_name="test",tp_rank="0",moe_ep_rank="0",pp_rank="0"'
 return '\n'.join('sglang:'+name+'{'+label+'} 1' for name in s.GAUGES)


class Tests(unittest.TestCase):
 def test_gauge_selection(self):
  data,ident=s.series(body(),'decode');self.assertEqual(len(data),len(s.GAUGES));self.assertTrue(all(x==1 for x in data.values()));self.assertEqual(len(ident),len(data))
 def test_missing_duplicate_invalid(self):
  t=body();lines=t.splitlines()
  for text in ('\n'.join(lines[:-1]),t+'\n'+lines[0],t.replace('} 1','} NaN',1),t.replace('tp_rank="0"','tp_rank="0",tp_rank="0"',1)):
   with self.assertRaises(ValueError):s.series(text,'decode')
 def test_wrong_role_not_zero(self):
  with self.assertRaises(ValueError):s.series(body(),'prefill')


if __name__=='__main__':unittest.main(verbosity=2)
