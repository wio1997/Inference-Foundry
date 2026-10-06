"""Native-fixture CPU contracts for ownership reservation; no Engine/NPU requests."""
import asyncio,hashlib,json,unittest
from unittest.mock import patch
import httpx
import test_response_affinity as legacy
from response_affinity_gateway_v2 import create_app
legacy.create_app=create_app

class V2Tests(legacy.AffinityTests):
 async def test_concurrent_custom_id_keeps_owner_before_first_response(self):
  class Concurrent(legacy.CPUStores):
   def __init__(self):
    super().__init__();self.arrived=asyncio.Event();self.release=asyncio.Event();self.posts=[]
   async def __call__(self,req):
    if req.method=="POST"and req.url.path=="/v1/responses":
     self.posts.append((req.url.host,await req.aread()))
     if len(self.posts)==1:self.arrived.set();await self.release.wait()
    return await super().__call__(req)
  self.store=Concurrent();app=self.app();ident="resp_custom_concurrent"
  raw=json.dumps(dict(model="glm-53",input="CPU fixture",request_id=ident,max_output_tokens=2)).encode()
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    first=asyncio.create_task(c.post("/v1/responses",content=raw))
    await asyncio.wait_for(self.store.arrived.wait(),5)
    owner_before=app.state.response_owner_index.resolve(ident);self.assertIsNotNone(owner_before)
    try:
     second=await asyncio.wait_for(c.post("/v1/responses",content=raw),5)
     self.assertEqual(second.status_code,200)
     self.assertEqual(app.state.response_owner_index.resolve(ident),owner_before)
    finally:self.store.release.set()
    self.assertEqual((await first).status_code,200)
    self.assertEqual(len({n for n,b in self.store.posts}),1)
    self.assertTrue(all(b==raw for n,b in self.store.posts))
    self.assertEqual(app.state.response_owner_index.resolve(ident),owner_before)
    self.assertEqual((await c.get("/v1/responses/"+ident)).status_code,200)
    self.assertFalse(app.state.placement.leases)
 async def test_cross_owner_previous_custom_conflict_has_no_rpc(self):
  app=self.app()
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    one=(await self.create(c,request_id="resp_cpu_A")).json()
    two=(await self.create(c,request_id="resp_cpu_B")).json()
    self.assertNotEqual(app.state.response_owner_index.resolve(one["id"]),app.state.response_owner_index.resolve(two["id"]))
    before=len(self.store.calls)
    response=await c.post("/v1/responses",json=dict(model="glm-53",input="fixture",previous_response_id=one["id"],request_id=two["id"]))
    self.assertEqual(response.status_code,503);self.assertEqual(len(self.store.calls),before)
    self.assertFalse(app.state.placement.leases)
 async def test_reservation_io_failure_has_no_rpc_or_leaked_lease(self):
  app=self.app()
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    before=len(self.store.calls)
    with patch("response_affinity.os.replace",side_effect=OSError("CPU fixture I/O failure")):
     response=await c.post("/v1/responses",json=dict(model="glm-53",input="fixture",request_id="resp_disk_failure"))
    self.assertEqual(response.status_code,503);self.assertEqual(len(self.store.calls),before)
    self.assertFalse(app.state.placement.leases)
    self.assertIsNone(app.state.response_owner_index.resolve("resp_disk_failure"))
    self.assertFalse(any(g.faulted for g in app.state.placement.execution_groups.values()))

if __name__=="__main__":unittest.main()
