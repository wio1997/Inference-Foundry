import json,tempfile,unittest
from pathlib import Path
import test_compose_sharded_map as fixtures
m=fixtures.m
class LimitTests(unittest.TestCase):
 def test_declared256_limit_accepted_with_all_integrity_checks(self):
  with tempfile.TemporaryDirectory()as t:
   cp,ip,mp,_,_=fixtures.ComposeTests().setup_case(t);ix=json.loads(ip.read_text());ix['limits']['max_rank_detail_json_bytes']=256<<20;ip.write_text(json.dumps(ix));r=m.build(cp,ip,mp,Path(t)/'out',expected=2);self.assertEqual(r['verified_identity_count'],2)
 def test_over256_declared_limit_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   cp,ip,mp,_,_=fixtures.ComposeTests().setup_case(t);ix=json.loads(ip.read_text());ix['limits']['max_rank_detail_json_bytes']=(256<<20)+1;ip.write_text(json.dumps(ix))
   with self.assertRaisesRegex(m.ComposeError,'256 MiB'):m.build(cp,ip,mp,Path(t)/'out',expected=2)
 def test_default204_still_rejects_two_identity_diagnostic(self):
  with tempfile.TemporaryDirectory()as t:
   cp,ip,mp,_,_=fixtures.ComposeTests().setup_case(t)
   with self.assertRaisesRegex(m.ComposeError,'identity_count'):m.build(cp,ip,mp,Path(t)/'out')
if __name__=='__main__':unittest.main()
