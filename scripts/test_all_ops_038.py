#!/usr/bin/env python3
"""Regression checks for the 32-rank sanitized aggregate."""
import unittest
from verify_all_ops_038 import verify,validate_dtype,validate_shape,scan_strings


class Tests(unittest.TestCase):
 def test_real_aggregate(self):
  d=verify();self.assertEqual(d['p']['csv_rows'],755447);self.assertEqual(d['d']['csv_rows'],556800)
 def test_dtype_allowlist(self):
  self.assertTrue(validate_dtype('INT8;DT_BF16'));self.assertFalse(validate_dtype('INT64;12345'));self.assertFalse(validate_dtype('unsafe/type'))
 def test_shape_and_privacy_guard(self):
  self.assertTrue(validate_shape('8192,2048;36,2048,4096'));self.assertTrue(validate_shape('unknown'));self.assertFalse(validate_shape('shape/private'))
  with self.assertRaises(ValueError):scan_strings({'nested':['/home/example/private']})
  with self.assertRaises(ValueError):scan_strings({'nested':['.'.join(('192','168','0','1'))]})


if __name__=='__main__':unittest.main(verbosity=2)
