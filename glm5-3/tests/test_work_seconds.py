import unittest,json,asyncio,httpx
from work_seconds_placement import WorkSecondsPlacement
from response_affinity_gateway_v7 import create_app
import test_prefill_workload_v4 as byte
import test_response_budget_v8 as prior
import test_response_affinity as legacy
CONFIG=[dict(id=k,url="http://"+k,decode_tps=100,prefill_bytes_per_s=1000)for k in["a","b"]]
class WorkContracts(unittest.IsolatedAsyncioTestCase):
 async def test_decode_reservation_can_outweigh_prefill_bytes(self):
  p=WorkSecondsPlacement(CONFIG,"work_seconds",byte.GROUPS)
  await p.acquire_for_owner(32,10000,byte.pointer("a"));b=await p.acquire_for_owner(8192,2000,byte.pointer("b"));await p.output_started(b)
  snap=await p.snapshot();self.assertEqual([x["prefill_input_bytes_hint"]for x in snap],[10000,0]);self.assertEqual([x["reserved_work_seconds_hint"]for x in snap],[10.32,81.92])
  self.assertEqual((await p.acquire_for_owner(32,10000)).replica.key,"a")
  await p.remove("a");self.assertEqual((await p.acquire(32,1)).replica.key,"b")
  with self.assertRaises(RuntimeError):await p.acquire_for_owner(32,1,byte.pointer("a"))
  await p.release(b,backend_failure=True)
  with self.assertRaises(RuntimeError):await p.acquire(32,1)
 async def test_missing_rate_falls_back_to_counts_for_all_eligible_members(self):
  config=[dict(CONFIG[0]),dict(id="b",url="http://b")]
  p=WorkSecondsPlacement(config,"work_seconds",byte.GROUPS)
  await p.acquire_for_owner(32,10000,byte.pointer("a"));await p.acquire_for_owner(8192,2000,byte.pointer("b"))
  self.assertEqual((await p.acquire(32,10000)).replica.key,"a")
  self.assertTrue(all(x["work_ranking_calibrated"]is False for x in await p.snapshot()))
 async def test_first_native_output_removes_only_prefill_hint(self):
  p=WorkSecondsPlacement(CONFIG,"work_seconds",byte.GROUPS);a=await p.acquire_for_owner(8192,10000,byte.pointer("a"))
  self.assertTrue(await p.output_started(a));self.assertEqual((await p.snapshot())[0]["reserved_work_seconds_hint"],81.92)
  self.assertIn(a.lease_id,p.leases);self.assertEqual(a.output_budget,8192)
class WorkGatewayContract(prior.GatewayBudgetContracts):
 async def test_actual_ASGI_three_held_requests_balance_prefill_and_decode_without_body_changes(self):
  class Held(legacy.CPUStores):
   def __init__(self):super().__init__();self.arrived=asyncio.Event();self.release=asyncio.Event();self.received=[]
   async def __call__(self,req):
    if req.method=="POST"and req.url.path=="/v1/responses":
     self.received.append((req.url.host,await req.aread()))
     if len(self.received)==3:self.arrived.set()
     await self.release.wait()
    return await super().__call__(req)
  self.store=Held()
  app=create_app(CONFIG,policy="work_seconds",groups=legacy.GROUPS,response_owner_state_path=str(self.root/'owners.json'),transport=httpx.MockTransport(self.store))
  bodies=[json.dumps(dict(model="glm-53",input=letter*size,request_id="resp_bytes_"+str(i),max_output_tokens=[32,8192,32][i])).encode()for i,(letter,size)in enumerate([("a",10000),("b",2000),("c",10000)])]
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    tasks=[]
    try:
     for i,body in enumerate(bodies):
      tasks.append(asyncio.create_task(c.post("/v1/responses",content=body)))
      for _ in range(100):
       if len(self.store.received)>=i+1:break
       await asyncio.sleep(.01)
      else:raise RuntimeError("held native fixture arrival")
     await asyncio.wait_for(self.store.arrived.wait(),3)
     self.assertEqual([x[0]for x in self.store.received],["a","b","a"])
     self.assertEqual([x[1]for x in self.store.received],bodies)
     self.assertEqual(app.state.response_owner_index.resolve("resp_bytes_2")["replica"],"a")
     self.assertEqual(sum(x["reserved_output_tokens"]for x in await app.state.placement.snapshot()),8256)
    finally:self.store.release.set()
    replies=await asyncio.gather(*tasks);self.assertTrue(all(x.status_code==200 for x in replies))
    self.assertFalse(app.state.placement.leases)
