"""Real ASGI/native protocol fixtures; no model execution."""
import asyncio,json,unittest
import httpx
import test_response_affinity_v3 as reserved
import test_response_affinity as legacy
from response_affinity_gateway_v3 import create_app
from request_estimates import estimate_request
from placement import estimate_request as old_estimate
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
reserved.create_app=create_app
legacy.create_app=create_app

class EstimateContracts(unittest.TestCase):
 def test_responses_native_output_budget_and_conflicting_legacy_fields(self):
  valid=dict(model="glm-52",input="fixture",max_output_tokens=8192)
  native=ResponsesRequest.model_validate(valid)
  self.assertEqual(native.max_output_tokens,8192)
  raw=json.dumps({**valid,"max_tokens":32,"max_completion_tokens":64,"n":7}).encode()
  self.assertEqual(estimate_request(raw,"/v1/responses"),(8192,len(raw)))
  self.assertEqual(estimate_request(json.dumps(valid).encode(),"/v1/responses")[0],8192)
 def test_unknown_or_invalid_responses_hints_do_not_raise(self):
  for payload in[b"not JSON",b"[]",b"null",b"\xff",json.dumps({}).encode()]+[json.dumps({"max_output_tokens":v}).encode()for v in[None,True,0,-1,1.5,"8192"]]:
   self.assertEqual(estimate_request(payload,"/v1/responses"),(16,len(payload)))
 def test_other_routes_preserve_original_estimates_exactly(self):
  for payload in[json.dumps({"max_completion_tokens":128,"max_tokens":64,"n":3}).encode(),json.dumps({"max_tokens":32,"best_of":4}).encode(),b"[]",b"malformed"]:
   for path in["/v1/chat/completions","/v1/completions","/custom",None]:
    self.assertEqual(estimate_request(payload,path),old_estimate(payload))

class GatewayBudgetContracts(reserved.V2Tests):
 async def test_live_responses_lease_uses_native_max_output_and_wire_is_exact(self):
  class Held(legacy.CPUStores):
   def __init__(self):
    super().__init__();self.arrived=asyncio.Event();self.release=asyncio.Event()
   async def __call__(self,req):
    if req.method=="POST"and req.url.path=="/v1/responses":
     self.original_body=await req.aread();self.arrived.set();await self.release.wait()
    return await super().__call__(req)
  self.store=Held();app=self.app();raw=json.dumps(dict(model="glm-52",input="fixture",request_id="resp_budget_contract",max_output_tokens=8192,background=True)).encode()
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    task=asyncio.create_task(c.post("/v1/responses",content=raw))
    try:
     await asyncio.wait_for(self.store.arrived.wait(),5)
     leases=list(app.state.placement.leases.values());self.assertEqual(len(leases),1);self.assertEqual(leases[0].output_budget,8192)
     self.assertEqual(sum(r["reserved_output_tokens"]for r in await app.state.placement.snapshot()),8192)
     self.assertIsNotNone(app.state.response_owner_index.resolve("resp_budget_contract"))
     self.assertEqual(self.store.original_body,raw)
    finally:self.store.release.set()
    result=await task;self.assertEqual(result.status_code,200);self.assertEqual(result.json()["status"],"queued")
    self.assertFalse(app.state.placement.leases)
    # Queued native background work is not represented by a completed HTTP lease.
    self.assertEqual(sum(r["reserved_output_tokens"]for r in await app.state.placement.snapshot()),0)
