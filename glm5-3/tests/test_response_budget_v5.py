"""Real ASGI/native protocol fixtures; no model execution."""
import asyncio,json,unittest,sys
import httpx
import test_response_affinity_v3 as reserved
import test_response_affinity as legacy
from response_affinity_gateway_v4 import create_app
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

class CLIBudgetContract(unittest.TestCase):
 def test_script_entrypoint_has_live8192_budget_and_exact_native_wire(self):
  import os,subprocess,tempfile,time,threading,socket,urllib.request,concurrent.futures
  from pathlib import Path
  from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
  root=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3")
  arrived=threading.Event();release=threading.Event();received=[]
  raw=legacy.JSON_RAW
  class NativeFixture(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_GET(self):
    self.send_response(200);self.send_header("content-length","2");self.end_headers();self.wfile.write(b"OK")
   def do_POST(self):
    received.append(self.rfile.read(int(self.headers["Content-Length"])));arrived.set()
    if not release.wait(20):raise RuntimeError("fixture release deadline")
    self.send_response(200);self.send_header("Content-Type","application/json")
    self.send_header("X-Duplicate","one");self.send_header("X-Duplicate","two")
    self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)
  server=ThreadingHTTPServer(("127.0.0.1",0),NativeFixture);t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
  sock=socket.socket();sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close()
  base="http://127.0.0.1:"+str(port);opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  payload=json.dumps(dict(model="glm-52",input="fixture",max_output_tokens=8192)).encode()
  with tempfile.TemporaryDirectory(prefix="glm-gateway-cli-cpu-")as tmp:
   env=dict(os.environ);env.update(GLM_REPLICAS=json.dumps([dict(id="fixture",url="http://127.0.0.1:"+str(server.server_port))]))
   for k in["GLM_EXECUTION_GROUPS","GLM_RESPONSE_OWNER_STATE_PATH","GLM_GROUP_FAULT_STATE_PATH","GLM_ROUTER_AUDIT_DIR","GLM_ROUTER_TRACE_PATH"]:env.pop(k,None)
   f=open(Path(tmp)/"gateway.log","wb");child=subprocess.Popen([sys.executable,str(root/"runtime/response_affinity_gateway_v4.py"),"--host","127.0.0.1","--port",str(port)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   try:
    deadline=time.monotonic()+15
    while True:
     self.assertIsNone(child.poll())
     try:
      with opener.open(base+"/control/replicas",timeout=1)as r:initial=json.load(r)
      break
     except Exception:
      if time.monotonic()>deadline:raise
      time.sleep(.05)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1)as pool:
     def post():
      req=urllib.request.Request(base+"/v1/responses",data=payload,headers={"Content-Type":"application/json"})
      with opener.open(req,timeout=20)as r:return r.status,r.read(),r.headers.get_all("x-duplicate")
     future=pool.submit(post)
     try:
      self.assertTrue(arrived.wait(8))
      with opener.open(base+"/control/replicas",timeout=3)as r:current=json.load(r)
      self.assertEqual(sum(x["reserved_output_tokens"]for x in current["replicas"]),8192)
      self.assertEqual(received,[payload])
     finally:release.set()
     status,wire,duplicate=future.result(timeout=8);self.assertEqual(status,200);self.assertEqual(wire,raw);self.assertEqual(duplicate,["one","two"])
    with opener.open(base+"/control/replicas",timeout=3)as r:final=json.load(r)
    self.assertEqual(sum(x["reserved_output_tokens"]for x in final["replicas"]),0)
   finally:
    release.set()
    if child.poll()is None:
     child.terminate()
     try:child.wait(timeout=8)
     except subprocess.TimeoutExpired:child.kill();child.wait(timeout=8)
    f.close();server.shutdown();server.server_close();t.join(timeout=3)
