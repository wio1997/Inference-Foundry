
import asyncio,json,os,pathlib,subprocess,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]/"runtime"))
from persistent_coupled_placement import PersistentCoupledPlacement
REPLICAS=[{"id":"A","url":"http://A"},{"id":"B","url":"http://B"}]
def groups(epoch="one",members=None):return [{"id":"EP","epoch":epoch,"members":members or ["A","B"]}]
class DurableGroupContracts(unittest.IsolatedAsyncioTestCase):
 async def test_fault_survives_clean_close_same_epoch_and_logical_readd(self):
  with tempfile.TemporaryDirectory()as d:
   path=str(pathlib.Path(d)/"state.json");p=PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=path)
   one=await p.acquire(4,1);peer=await p.acquire(4,1)
   self.assertTrue(await p.release(one,True));self.assertIn(peer.lease_id,p.leases)
   self.assertFalse(await p.release(one,True));await p.release(peer);await p.close()
   q=PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=path)
   self.assertTrue(all(x["group_faulted"]for x in await q.snapshot()))
   await q.remove("A");await q.add(REPLICAS[0])
   with self.assertRaises(RuntimeError):await q.acquire(1,1)
   await q.close()
   fresh=PersistentCoupledPlacement(REPLICAS,groups=groups("two"),fault_state_path=path)
   lease=await fresh.acquire(1,1);await fresh.release(lease);await fresh.close()
 async def test_healthy_close_reopens_and_single_writer_is_enforced(self):
  with tempfile.TemporaryDirectory()as d:
   path=str(pathlib.Path(d)/"state.json");p=PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=path)
   with self.assertRaises(BlockingIOError):PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=path)
   lease=await p.acquire(1,1);await p.release(lease);await p.close()
   self.assertFalse(json.loads(pathlib.Path(path).read_text())["open"])
   q=PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=path)
   self.assertFalse(any(x["group_faulted"]for x in await q.snapshot()));await q.close()
 async def test_actual_process_exit_releases_flock_but_dirty_epoch_stays_quarantined(self):
  with tempfile.TemporaryDirectory()as d:
   path=str(pathlib.Path(d)/"state.json")
   code="import sys,os;sys.path.insert(0,"+repr(str(pathlib.Path(__file__).parents[1]/"runtime"))+");from persistent_coupled_placement import PersistentCoupledPlacement;p=PersistentCoupledPlacement("+repr(REPLICAS)+",groups="+repr(groups())+",fault_state_path="+repr(path)+");os._exit(17)"
   result=subprocess.run([sys.executable,"-c",code],capture_output=True);self.assertEqual(result.returncode,17)
   p=PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=path)
   with self.assertRaises(RuntimeError):await p.acquire(1,1)
   await p.close()
 async def test_persist_failure_retains_dirty_marker_and_releases_owned_lease(self):
  with tempfile.TemporaryDirectory()as d:
   path=str(pathlib.Path(d)/"state.json");p=PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=path)
   lease=await p.acquire(1,1)
   with patch("persistent_coupled_placement.os.replace",side_effect=OSError("fixture disk failure")):
    self.assertTrue(await p.release(lease,True))
   self.assertFalse(p.leases);self.assertTrue(p.journal_error)
   await p.close();self.assertTrue(json.loads(pathlib.Path(path).read_text())["open"])
   q=PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=path)
   with self.assertRaises(RuntimeError):await q.acquire(1,1)
   await q.close()
 async def test_invalid_state_and_membership_drift_fail_before_admission(self):
  with tempfile.TemporaryDirectory()as d:
   path=pathlib.Path(d)/"state.json";path.write_text('{"schema":1,"open":"false","groups":{}}')
   with self.assertRaises(ValueError):PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=str(path))
   path.write_text(json.dumps({"schema":1,"open":False,"groups":{"EP":{"epoch":"one","members":["A"],"faulted":False}}}))
   with self.assertRaises(ValueError):PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=str(path))
 async def test_closed_with_active_lease_is_conservative_and_default_unchanged(self):
  with tempfile.TemporaryDirectory()as d:
   path=str(pathlib.Path(d)/"state.json");p=PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=path)
   await p.acquire(1,1);await p.close();self.assertTrue(json.loads(pathlib.Path(path).read_text())["open"])
   q=PersistentCoupledPlacement(REPLICAS,groups=groups(),fault_state_path=path)
   with self.assertRaises(RuntimeError):await q.acquire(1,1)
   await q.close()
  p=PersistentCoupledPlacement(REPLICAS);lease=await p.acquire(1,1);await p.release(lease,True)
  other=await p.acquire(1,1);self.assertEqual(other.replica.key,"B");await p.release(other);await p.close()
if __name__=="__main__":unittest.main()
