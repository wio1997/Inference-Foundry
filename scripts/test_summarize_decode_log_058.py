#!/usr/bin/env python3
"""Synthetic log-field tests; no real logs or request identifiers."""
import unittest
import summarize_decode_log_058 as s


def sample():
 return ('[2026-11-17 23:00:00 TP0 EP0] Decode batch, #running-req: 2, '
         '#full token: 328704, full token usage: 0.95, mamba num: 15, '
         '#prealloc-req: 11, #transfer-req: 3, #queue-req: 0')


class Tests(unittest.TestCase):
 def test_selected_fields(self):
  timestamp,data=s.parse_line(sample());self.assertEqual(timestamp.tzinfo,None);self.assertEqual(data['full_tokens'],328704);self.assertEqual(data['prealloc_queue'],11)
 def test_incomplete_ambiguous_and_zone_fail(self):
  for text in (sample().replace('#transfer-req: 3, ',''),sample()+' #prealloc-req: 9,',sample().replace('TP0','TP1'),sample().replace('23:00:00','23:00:00Z'),sample().replace('23:00:00','23:00:00 UTC+08:00'),sample().replace('23:00:00','23:00:00 +08:00'),sample().replace('23:00:00','23:00:00.123456789'),sample().replace('0.95','0.95e-3'),sample().replace('0.95','0.9'),sample().replace('0.95','0.950'),sample().replace('#prealloc-req: 11,','#prealloc-req: 11BAD,')):
   with self.assertRaises(ValueError):s.parse_line(text)
 def test_quantile_and_missing(self):
  self.assertEqual(s.quantile([1,3],.5),2);self.assertIsNone(s.metrics([]));self.assertIsNone(s.parse_line('unrelated text'))
 def test_half_open_window(self):
  self.assertTrue(s.in_window(0,0,10));self.assertTrue(s.in_window(9.999,0,10));self.assertFalse(s.in_window(10,0,10));self.assertFalse(s.in_window(-1,0,10))


if __name__=='__main__':unittest.main(verbosity=2)
