"""Native owner/fault invariants plus observed prefill byte backlog ordering."""
import unittest,asyncio,json,httpx
from work_seconds_placement import WorkSecondsPlacement as PrefillWorkPlacement
from response_affinity_gateway_v7 import create_app
import test_response_budget_v8 as prior
import test_response_affinity as legacy
CONFIG=[dict(id="a",url="http://a"),dict(id="b",url="http://b")]
GROUPS=[dict(id="ep",epoch="native-e1",members=["a","b"])]
def pointer(key):return dict(replica=key,url="http://"+key,group="ep",epoch="native-e1")
class BytePlacementContracts(unittest.IsolatedAsyncioTestCase):
 async def test_long_prefill_is_not_equivalent_to_medium_with_more_decodes(self):
  p=PrefillWorkPlacement(CONFIG,"prefill_bytes",GROUPS)
  await p.acquire_for_owner(512,10000,pointer("a"))
  await p.acquire_for_owner(1024,3000,pointer("b"))
  for key,count in[("a",3),("b",5)]:
   for _ in range(count):
    lease=await p.acquire_for_owner(64,100,pointer(key));await p.output_started(lease)
  snap=await p.snapshot();self.assertEqual([x["prefilling_requests"]for x in snap],[1,1]);self.assertEqual([x["active_requests"]for x in snap],[4,6])
  self.assertEqual([x["prefill_input_bytes_hint"]for x in snap],[10000,3000])
  selected=await p.acquire(256,10000);self.assertEqual(selected.replica.key,"b")
 async def test_first_output_removes_byte_hint_but_keeps_native_lease(self):
  p=PrefillWorkPlacement(CONFIG,"prefill_bytes",GROUPS)
  a=await p.acquire_for_owner(8192,10000,pointer("a"));await p.acquire_for_owner(32,3000,pointer("b"))
  self.assertTrue(await p.output_started(a));self.assertIn(a.lease_id,p.leases)
  snap=await p.snapshot();self.assertEqual(snap[0]["prefill_input_bytes_hint"],0);self.assertEqual(snap[0]["reserved_output_tokens"],8192)
  self.assertEqual((await p.acquire(32,100)).replica.key,"a")
 async def test_drain_and_coupled_fault_preserve_owner_and_quarantine(self):
  p=PrefillWorkPlacement(CONFIG,"prefill_bytes",GROUPS);a=await p.acquire_for_owner(32,1000,pointer("a"));await p.acquire_for_owner(32,500,pointer("b"))
  await p.remove("b");self.assertEqual((await p.acquire(32,100)).replica.key,"a")
  await p.release(a,backend_failure=True)
  with self.assertRaises(RuntimeError):await p.acquire(32,100)
  with self.assertRaises(RuntimeError):await p.acquire_for_owner(32,100,pointer("b"))
  self.assertTrue(all(x["group_faulted"]for x in await p.snapshot()))
 async def test_invalid_hints_and_policy_are_not_admitted(self):
  p=PrefillWorkPlacement(CONFIG,"prefill_bytes",GROUPS)
  for budget,size in[(0,1),(True,1),(32,-1),(32,1.5)]:
   with self.assertRaises(ValueError):await p.acquire(budget,size)
  with self.assertRaises(ValueError):PrefillWorkPlacement(CONFIG,"unknown",GROUPS)
 async def test_legacy_policy_path_is_preserved(self):
  p=PrefillWorkPlacement(CONFIG,"prefill_aware",GROUPS)
  a=await p.acquire(32,10000);b=await p.acquire(32,10);self.assertEqual([a.replica.key,b.replica.key],["a","b"])
  self.assertEqual((await p.acquire(32,20000)).replica.key,"a")
class ByteGatewayContract(prior.GatewayBudgetContracts):
 # Loader selects only this method; inherited cases are exercised separately.
 async def test_actual_ASGI_three_held_requests_reserve_lighter_owner_without_body_changes(self):
  class Held(legacy.CPUStores):
   def __init__(self):super().__init__();self.arrived=asyncio.Event();self.release=asyncio.Event();self.received=[]
   async def __call__(self,req):
    if req.method=="POST"and req.url.path=="/v1/responses":
     self.received.append((req.url.host,await req.aread()))
     if len(self.received)==3:self.arrived.set()
     await self.release.wait()
    return await super().__call__(req)
  self.store=Held()
  app=create_app(legacy.CONFIG,policy="prefill_bytes",groups=legacy.GROUPS,response_owner_state_path=str(self.root/'owners.json'),transport=httpx.MockTransport(self.store))
  bodies=[json.dumps(dict(model="glm-52",input=letter*size,request_id="resp_bytes_"+str(i),max_output_tokens=8192)).encode()for i,(letter,size)in enumerate([("a",10000),("b",2000),("c",10000)])]
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
     self.assertEqual([x[0]for x in self.store.received],["a","b","b"])
     self.assertEqual([x[1]for x in self.store.received],bodies)
     self.assertEqual(app.state.response_owner_index.resolve("resp_bytes_2")["replica"],"b")
     self.assertEqual(sum(x["reserved_output_tokens"]for x in await app.state.placement.snapshot()),8192*3)
    finally:self.store.release.set()
    replies=await asyncio.gather(*tasks);self.assertTrue(all(x.status_code==200 for x in replies))
    self.assertFalse(app.state.placement.leases)
