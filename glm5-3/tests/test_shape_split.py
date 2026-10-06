import unittest
from shape_split_placement import ShapeSplitPlacement
CONFIG=[dict(id="P",url="http://p"),dict(id="D",url="http://d")]
SHAPE=dict(input_threshold_bytes=8192,prefill_members=["D"],decode_members=["P"])
GROUPS=[dict(id=k,epoch="native-"+k,members=[k])for k in["P","D"]]
class ShapeContracts(unittest.IsolatedAsyncioTestCase):
 def app(self):return ShapeSplitPlacement(CONFIG,"shape_split",GROUPS,shape_config=SHAPE)
 async def test_boundary_reserves_whole_output_and_releases_exactly_once(self):
  p=self.app();small=await p.acquire_for_owner(8192,8191);large=await p.acquire_for_owner(32,8192)
  self.assertEqual((small.replica.key,large.replica.key),("P","D"))
  self.assertEqual(small.replica.active[small.lease_id],(8192,8191))
  self.assertTrue(await p.output_started(small));self.assertFalse(await p.output_started(small))
  self.assertTrue(await p.release(small));self.assertFalse(await p.release(small));self.assertTrue(await p.release(large))
 async def test_bound_owner_overrides_shape_and_rejects_epoch_drift(self):
  p=self.app();owner=dict(replica="D",url="http://d",group="D",epoch="native-D")
  lease=await p.acquire_for_owner(64,10,owner);self.assertEqual(lease.replica.key,"D");await p.release(lease)
  with self.assertRaises(RuntimeError):await p.acquire_for_owner(64,10,{**owner,"epoch":"old"})
 async def test_failed_preferred_domain_falls_back_only_for_unbound_requests(self):
  p=self.app();lease=await p.acquire(32,9000);await p.release(lease,backend_failure=True)
  fallback=await p.acquire(32,9000);self.assertEqual(fallback.replica.key,"P");await p.release(fallback)
  owner=dict(replica="D",url="http://d",group="D",epoch="native-D")
  with self.assertRaises(RuntimeError):await p.acquire_for_owner(64,10,owner)
 async def test_drain_and_no_eligible_domains(self):
  p=self.app();held=await p.acquire(32,9000);await p.remove("D")
  lease=await p.acquire(32,9000);self.assertEqual(lease.replica.key,"P");await p.release(lease)
  await p.remove("P")
  with self.assertRaises(RuntimeError):await p.acquire(32,1)
  await p.release(held);self.assertFalse(p.replicas)
 async def test_invalid_config_and_estimates_fail_before_admission(self):
  for c in[{**SHAPE,"input_threshold_bytes":True},{**SHAPE,"input_threshold_bytes":0},{**SHAPE,"prefill_members":["unknown"]},{**SHAPE,"decode_members":["D"]},{**SHAPE,"decode_members":["P","P"]}]:
   with self.assertRaises(ValueError):ShapeSplitPlacement(CONFIG,"shape_split",GROUPS,shape_config=c)
  p=self.app()
  for b,s in[(True,1),(0,1),(1,-1),(1,True)]:
   with self.assertRaises(ValueError):await p.acquire(b,s)
  self.assertFalse(p.leases)
 async def test_existing_active_count_without_shape_retains_routing(self):
  p=ShapeSplitPlacement(CONFIG,groups=GROUPS)
  one=await p.acquire(32,9000);two=await p.acquire(32,1)
  self.assertNotEqual(one.replica.key,two.replica.key)
  self.assertTrue(all(x["shape_split_hint"]is None for x in await p.snapshot()))
  await p.release(one);await p.release(two)
class ShapeCLI(unittest.TestCase):
 def test_actual_script_routes_both_shapes_and_preserves_bytes(self):
  import os,sys,json,socket,subprocess,tempfile,time,threading,urllib.request
  from pathlib import Path
  from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
  from test_response_affinity import JSON_RAW
  received={"P":[],"D":[]}
  def handler(key):
   class Fixture(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
     self.send_response(200);self.send_header("Content-Length","2");self.end_headers();self.wfile.write(b"OK")
    def do_POST(self):
     received[key].append(self.rfile.read(int(self.headers["Content-Length"])))
     self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("X-Duplicate","one");self.send_header("X-Duplicate","two");self.send_header("Content-Length",str(len(JSON_RAW)));self.end_headers();self.wfile.write(JSON_RAW)
   return Fixture
  servers={k:ThreadingHTTPServer(("127.0.0.1",0),handler(k))for k in ["P","D"]}
  for v in servers.values():threading.Thread(target=v.serve_forever,daemon=True).start()
  sock=socket.socket();sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close()
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));base="http://127.0.0.1:"+str(port)
  payloads=[json.dumps(dict(model="glm-53",input=v,max_output_tokens=32)).encode()for v in ["small","large"*2000]]
  env=dict(os.environ,GLM_PLACEMENT_POLICY="shape_split",GLM_SHAPE_SPLIT=json.dumps(SHAPE),GLM_REPLICAS=json.dumps([dict(id=k,url="http://127.0.0.1:"+str(v.server_port))for k,v in servers.items()]))
  for k in["GLM_EXECUTION_GROUPS","GLM_RESPONSE_OWNER_STATE_PATH","GLM_GROUP_FAULT_STATE_PATH","GLM_ROUTER_TRACE_PATH","GLM_ROUTER_AUDIT_DIR","GLM_PD_PRODUCERS","GLM_PD_NATIVE_PLANS"]:env.pop(k,None)
  with tempfile.TemporaryDirectory(prefix="glm-shape-cli-")as tmp:
   with (Path(tmp)/"gateway.log").open("wb")as log:
    child=subprocess.Popen([sys.executable,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/response_affinity_gateway_v11.py","--host","127.0.0.1","--port",str(port)],env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
     deadline=time.monotonic()+15
     while True:
      self.assertIsNone(child.poll())
      try:
       with opener.open(base+"/control/replicas",timeout=1)as r:snapshot=json.load(r)
       break
      except Exception:
       if time.monotonic()>deadline:raise
       time.sleep(.05)
     self.assertTrue(all(x["placement_policy"]=="shape_split"and x["shape_split_hint"]==SHAPE for x in snapshot["replicas"]))
     for raw in payloads:
      with opener.open(urllib.request.Request(base+"/v1/responses",data=raw,headers={"Content-Type":"application/json"}),timeout=5)as reply:
       self.assertEqual(reply.read(),JSON_RAW);self.assertEqual(reply.headers.get_all("x-duplicate"),["one","two"])
     self.assertEqual(received,{"P":[payloads[0]],"D":[payloads[1]]})
    finally:
     child.terminate()
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
  for server in servers.values():server.shutdown();server.server_close()
