import asyncio,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
from coupled_placement import CoupledPlacement
from coupled_gateway import create_app
CONFIG=[{"id":"DP0","url":"http://native0"},{"id":"DP1","url":"http://native1"}]
GROUPS=[{"id":"EP32","epoch":"verified-native-cohort-A","members":["DP0","DP1"]}]
class GroupContracts(unittest.IsolatedAsyncioTestCase):
 async def test_fault_survives_readd_and_does_not_revoke_peer_lease(self):
  pool=CoupledPlacement(CONFIG,groups=GROUPS)
  a=await pool.acquire(10,20);b=await pool.acquire(10,20)
  self.assertNotEqual(a.replica.key,b.replica.key)
  self.assertTrue(await pool.release(a,True))
  self.assertEqual(len(pool.leases),1)
  with self.assertRaises(RuntimeError):await pool.acquire(1,0)
  await pool.remove(a.replica.key);await pool.add(next(x for x in CONFIG if x["id"]==a.replica.key))
  with self.assertRaises(RuntimeError):await pool.acquire(1,0)
  self.assertTrue(await pool.release(b));self.assertFalse(await pool.release(a,True))
  self.assertTrue(all(x["group_faulted"]for x in await pool.snapshot()))
 async def test_logical_drain_cancel_and_new_controller_epoch(self):
  pool=CoupledPlacement(CONFIG,groups=GROUPS)
  lease=await pool.acquire(2,1);await pool.remove(lease.replica.key)
  peer=await pool.acquire(2,1);self.assertNotEqual(peer.replica.key,lease.replica.key)
  await pool.release(lease);await pool.release(peer)
  await pool.add(next(x for x in CONFIG if x["id"]==lease.replica.key))
  self.assertFalse(any(x["group_faulted"]for x in await pool.snapshot()))
  newer=CoupledPlacement(CONFIG,groups=[{**GROUPS[0],"epoch":"verified-native-cohort-B"}])
  self.assertIsNotNone(await newer.acquire(1,0))
 async def test_independent_and_partition_guards(self):
  pool=CoupledPlacement(CONFIG,groups=[]);lease=await pool.acquire(2,1);await pool.release(lease,True)
  peer=await pool.acquire(2,1);self.assertNotEqual(peer.replica.key,lease.replica.key)
  await pool.release(peer)
  for groups in [[{**GROUPS[0],"members":["unknown"]}],[GROUPS[0],{**GROUPS[0],"id":"other"}]]:
   with self.assertRaises(ValueError):CoupledPlacement(CONFIG,groups=groups)
 async def test_native_sse500_raw_passthrough_and_false_peer_health(self):
  import httpx
  # Actual Run32 native error payload retained as an event; reconstruct SSE
  # framing for the CPU transport fixture. This is not new native inference.
  evidence=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0032/control_DP0/cold_DP0.response.jsonl")
  event=json.loads(evidence.read_text().splitlines()[0])
  payload=event.get("data")
  if payload is None:payload=event.get("raw")
  self.assertIsInstance(payload,str)
  wire=("data: "+payload+"\n\ndata: [DONE]\n\n").encode()
  class Raw(httpx.AsyncByteStream):
   async def __aiter__(self):
    for i in range(0,len(wire),7):yield wire[i:i+7]
  calls=[]
  async def backend(request):
   calls.append((request.method,request.url.host,request.url.path))
   if request.url.path=="/health":return httpx.Response(200,content=b"healthy")
   return httpx.Response(200,headers={"content-type":"text/event-stream"},stream=Raw())
  app=create_app(CONFIG,transport=httpx.MockTransport(backend),groups=GROUPS)
  async with app.router.lifespan_context(app):
   async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://router")as client:
    result=await client.post("/v1/chat/completions",content=b"{}")
    self.assertEqual(result.status_code,200);self.assertEqual(result.content,wire)
    health=await client.get("/healthcheck");self.assertEqual(health.status_code,503)
    blocked=await client.post("/v1/chat/completions",content=b"{}");self.assertEqual(blocked.status_code,503)
    self.assertEqual(sum(path=="/v1/chat/completions"for _,_,path in calls),1)
    self.assertEqual(len(app.state.placement.leases),0)
if __name__=="__main__":unittest.main()
