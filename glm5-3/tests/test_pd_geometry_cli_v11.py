"""Actual script/factory integration; CPU loopback fixtures, no GLM inference."""
import json,os,sys,time,tempfile,threading,socket,subprocess,urllib.request,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
class GeometryCLI(unittest.TestCase):
 def test_unknown_PD_geometry_sends_original_once_to_decoder_with_zero_helper(self):
  calls={"P":[],"D":[]};raw=b'{"cpu_fixture":true}\n'
  def handler(role):
   class Fixture(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
     self.send_response(200);self.send_header("content-length","2");self.end_headers();self.wfile.write(b"OK")
    def do_POST(self):
     body=self.rfile.read(int(self.headers["Content-Length"]));calls[role].append(body)
     out=raw if role=="D"else json.dumps(dict(usage=dict(completion_tokens=1),kv_transfer_params=dict(do_remote_prefill=True,remote_host="127.0.0.1",remote_port=28000,remote_engine_id="cpu_fixture",remote_dcp_size=16,remote_pcp_size=1,remote_block_ids=[[0]]))).encode()
     self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("X-Duplicate","one");self.send_header("X-Duplicate","two");self.send_header("Content-Length",str(len(out)));self.end_headers();self.wfile.write(out)
   return Fixture
  servers={k:ThreadingHTTPServer(("127.0.0.1",0),handler(k))for k in calls};threads=[]
  for server in servers.values():
   t=threading.Thread(target=server.serve_forever,daemon=True);t.start();threads.append(t)
  sock=socket.socket();sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close()
  urls={k:"http://127.0.0.1:"+str(s.server_port)for k,s in servers.items()};base="http://127.0.0.1:"+str(port);http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  payload=json.dumps(dict(model="glm-52",messages=[dict(role="user",content="CPU fixture only. "*1024)],max_tokens=32,stream=False)).encode()
  env=dict(os.environ);env.update(GLM_REPLICAS=json.dumps([dict(id="D",url=urls["D"])]),GLM_PD_PRODUCERS=json.dumps({urls["D"]:dict(url=urls["P"],remote_host="127.0.0.1",remote_port=28000,dcp_size=16)}),GLM_PD_NATIVE_PLANS="{}",GLM_PLACEMENT_POLICY="active_count")
  for k in ["GLM_EXECUTION_GROUPS","GLM_RESPONSE_OWNER_STATE_PATH","GLM_GROUP_FAULT_STATE_PATH","GLM_ROUTER_AUDIT_DIR","GLM_ROUTER_TRACE_PATH","GLM_PD_AUDIT_DIR"]:env.pop(k,None)
  with tempfile.TemporaryDirectory(prefix="glm-pd-geometry-cli-")as tmp:
   log=open(Path(tmp)/"gateway.log","wb");child=None
   try:
    child=subprocess.Popen([sys.executable,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/response_affinity_gateway_v11.py","--host","127.0.0.1","--port",str(port)],env=env,stdout=log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+20
    while True:
     self.assertIsNone(child.poll())
     try:
      with http.open(base+"/control/replicas",timeout=1)as r:r.read()
      break
     except Exception:
      if time.monotonic()>deadline:raise
      time.sleep(.05)
    with http.open(urllib.request.Request(base+"/v1/chat/completions",data=payload,headers={"content-type":"application/json"}),timeout=20)as r:
     self.assertEqual(r.status,200);self.assertEqual(r.read(),raw);self.assertEqual(r.headers.get_all("x-duplicate"),["one","two"])
    self.assertEqual(calls,{"P":[],"D":[payload]})
   finally:
    if child is not None and child.poll()is None:
     child.terminate()
     try:child.wait(timeout=8)
     except subprocess.TimeoutExpired:child.kill();child.wait(timeout=8)
    log.close()
    for server in servers.values():server.shutdown();server.server_close()
    for t in threads:t.join(timeout=3)
