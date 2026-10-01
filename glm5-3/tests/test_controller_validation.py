import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
import controller
class ControllerValidation(unittest.TestCase):
 def setup_spec(self,root,stages):
  p=root/"run"/"controller_spec.json";p.parent.mkdir()
  p.write_text(json.dumps({"run_id":"TEST","stages":stages}));return p
 def test_invalid_stage_never_publishes_running(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);locks=root/"locks";locks.mkdir();p=self.setup_spec(root,[{"id":"bad/../id","argv":["unused"]}])
   with patch.object(controller,"LOCK_ROOT",locks),self.assertRaises(ValueError):controller.work(p)
   self.assertFalse((locks/"controller-owner.json").exists());self.assertFalse((p.parent/"state.json").exists())
 def test_bad_pin_never_executes_or_publishes(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);locks=root/"locks";locks.mkdir();source=root/"source";source.write_text("actual")
   p=self.setup_spec(root,[{"id":"one","argv":["unused"],"sources":[{"path":str(source),"sha256":"wrong"}]}])
   with patch.object(controller,"LOCK_ROOT",locks),patch.object(controller,"run_phase") as phase,self.assertRaises(ValueError):controller.work(p)
   phase.assert_not_called();self.assertFalse((locks/"controller-owner.json").exists())
 def test_source_changes_between_stages_are_terminal_no_second_execution(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);locks=root/"locks";locks.mkdir();source=root/"source";source.write_text("original");pin={"path":str(source),"sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
   p=self.setup_spec(root,[{"id":x,"argv":["unused"],"sources":[pin]} for x in ["one","two"]])
   def first(*args,**kwargs):source.write_text("changed")
   with patch.object(controller,"LOCK_ROOT",locks),patch.object(controller.signal,"signal"),patch.object(controller,"run_phase",side_effect=first) as phase:
    self.assertEqual(controller.work(p),1)
   self.assertEqual(phase.call_count,1)
   state=json.loads((p.parent/"state.json").read_text());owner=json.loads((locks/"controller-owner.json").read_text())
   self.assertEqual(state["status"],"failed");self.assertEqual(owner["status"],"failed");self.assertEqual(state["failure_phase"],"two");self.assertEqual(state["completed_stages"],["one"])
if __name__=="__main__":unittest.main()
