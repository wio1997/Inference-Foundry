import unittest,json,asyncio,tempfile,httpx
from pathlib import Path
from native_pd_transport_v2 import NativePDTransport,helper_body
from response_affinity_gateway_v9 import create_app
import test_response_affinity as legacy
PRODUCER={"http://a":dict(url="http://p",remote_host="172.16.10.166",remote_port=28000,dcp_size=16)}
KV=dict(do_remote_prefill=True,remote_host="172.16.10.166",remote_port=28000,
        remote_engine_id="nativeP",remote_dcp_size=16,remote_pcp_size=1,remote_block_ids=[[4,5]],
        remote_block_size=128,num_prompt_blocks=17)
def payload(path="/v1/responses",**kw):
    d=dict(model="glm-53",input="上海 text "*1024,max_output_tokens=32,temperature=.2,seed=31,store=True,request_id="resp_public",cache_salt="cold")
    if path.endswith("chat/completions"):
        d.pop("input");d.pop("max_output_tokens");d.pop("store");d.pop("request_id")
        d.update(messages=[dict(role="user",content="上海 text "*1024)],max_tokens=64,stream=True,stream_options=dict(include_usage=True))
    d.update(kw);return json.dumps(d,ensure_ascii=False,indent=2).encode()
class PDTransportTests(unittest.IsolatedAsyncioTestCase):
 async def test_chat_and_responses_original_public_options_native_wire(self):
  for path in ["/v1/responses","/v1/chat/completions"]:
   calls=[];events=[]
   async def backend(req):
    raw=await req.aread();calls.append((req.url.host,json.loads(raw),req.headers.get("x-request-id"),raw))
    if req.url.host=="p":
     usage={"output_tokens":1,"input_tokens":2111}if path.endswith("responses")else{"completion_tokens":1,"prompt_tokens":2111}
     return httpx.Response(200,json=dict(kv_transfer_params=KV,usage=usage))
    return httpx.Response(200,headers=[(b"content-type",b"text/event-stream"),(b"x-duplicate",b"one"),(b"x-duplicate",b"two")],stream=legacy.Wire(b"event: native\ndata: opaque\n\n"))
   t=NativePDTransport(PRODUCER,httpx.MockTransport(backend),trace=lambda e,**f:events.append((e,f)))
   async with httpx.AsyncClient(transport=t)as c:
    reply=await c.post("http://a"+path,content=payload(path),headers={"x-request-id":"public-trace"})
   self.assertEqual([x[0]for x in calls],["p","a"]);helper=calls[0][1];d=calls[1][1];original=json.loads(payload(path))
   self.assertEqual(d,{**original,"kv_transfer_params":KV});self.assertEqual(calls[1][2],"public-trace")
   self.assertEqual(helper["seed"],original["seed"]);self.assertEqual(helper["cache_salt"],original["cache_salt"]);self.assertFalse(helper["stream"])
   if path.endswith("responses"):
    self.assertFalse(helper["store"]);self.assertNotEqual(helper["request_id"],original["request_id"]);self.assertEqual(helper["max_output_tokens"],1)
   else:self.assertEqual(helper["messages"],original["messages"]);self.assertEqual(helper["max_tokens"],1);self.assertNotIn("stream_options",helper)
   self.assertEqual(reply.content,b"event: native\ndata: opaque\n\n");self.assertEqual(reply.headers.get_list("x-duplicate"),["one","two"])
   self.assertEqual([e for e,f in events],["pd_helper_started","pd_helper_wire","pd_helper_completed","pd_decoder_dispatch"])
 async def test_full_native_fallback_keeps_exact_body_query_for_unknown_and_stateful(self):
  calls=[]
  async def backend(req):calls.append((req.url.host,req.url.query,await req.aread()));return httpx.Response(200,content=b"native")
  async with httpx.AsyncClient(transport=NativePDTransport(PRODUCER,httpx.MockTransport(backend)))as c:
   cases=[payload(previous_response_id="resp_prior"),payload(background=True),payload(tools=[{}]),payload(input=[{"type":"unknown"}]),payload(extra_native_feature=True),payload(max_output_tokens=0),b"{invalid"]
   for raw in cases:
    res=await c.post("http://a/v1/responses?native=1",content=raw);self.assertEqual(res.content,b"native");self.assertEqual(calls[-1],("a",b"native=1",raw))
   await c.get("http://a/v1/responses/resp_id?stream=true")
  self.assertEqual(len(calls),len(cases)+1);self.assertTrue(all(x[0]=="a"for x in calls))
 async def test_invalid_metadata_no_D_and_helper_validation_native_fallback(self):
  for value,status in [(dict(error="bad"),400),(dict(usage={"output_tokens":1},kv_transfer_params={**KV,"remote_port":28001}),200),(dict(usage={"output_tokens":2},kv_transfer_params=KV),200)]:
   calls=[]
   async def backend(req):calls.append(req.url.host);return httpx.Response(status,json=value)
   async with httpx.AsyncClient(transport=NativePDTransport(PRODUCER,httpx.MockTransport(backend)))as c:res=await c.post("http://a/v1/responses",content=payload())
   self.assertEqual(calls,["p","a"]if status==400 else ["p"]);self.assertEqual(res.status_code,400 if status==400 else 502)
 async def test_helper_cancellation_does_not_dispatch_D(self):
  arrived=asyncio.Event();calls=[]
  async def backend(req):
   calls.append(req.url.host);arrived.set();await asyncio.Event().wait()
  async with httpx.AsyncClient(transport=NativePDTransport(PRODUCER,httpx.MockTransport(backend)))as c:
   task=asyncio.create_task(c.post("http://a/v1/responses",content=payload()))
   await asyncio.wait_for(arrived.wait(),2);task.cancel()
   with self.assertRaises(asyncio.CancelledError):await task
  self.assertEqual(calls,["p"])
 async def test_decoder_failure_never_retries_or_recomputes(self):
  calls=[]
  async def backend(req):
   calls.append(req.url.host)
   if req.url.host=="p":return httpx.Response(200,json=dict(usage={"output_tokens":1},kv_transfer_params=KV))
   raise httpx.ReadError("D failure")
  async with httpx.AsyncClient(transport=NativePDTransport(PRODUCER,httpx.MockTransport(backend)))as c:
   with self.assertRaises(httpx.ReadError):await c.post("http://a/v1/responses",content=payload())
  self.assertEqual(calls,["p","a"])
 async def test_actual_V8_PD_owner_restart_and_previous_native_fallback(self):
  stores=legacy.CPUStores();helper_calls=[];D_bodies=[]
  async def backend(req):
   if req.url.host=="p":
    helper_calls.append(json.loads(await req.aread()));return httpx.Response(200,json=dict(usage={"output_tokens":1},kv_transfer_params=KV))
   if req.method=="POST":D_bodies.append(await req.aread())
   return await stores(req)
  with tempfile.TemporaryDirectory()as root:
   config=[dict(id="a",url="http://a")];groups=[dict(id="D-native",epoch="D1",members=["a"])]
   for attempt in [0,1]:
    app=create_app(config,groups=groups,response_owner_state_path=str(Path(root)/"owners.json"),transport=httpx.MockTransport(backend),pd_config=PRODUCER)
    async with app.router.lifespan_context(app):
     async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://gateway")as c:
      if attempt==0:
       res=await c.post("/v1/responses",content=payload());self.assertEqual(res.status_code,200)
       self.assertEqual(res.json()["id"],"resp_public");self.assertEqual(app.state.response_owner_index.resolve("resp_public")["replica"],"a")
       self.assertNotIn(helper_calls[0]["request_id"],app.state.response_owner_index.entries)
      else:
       self.assertEqual((await c.get("/v1/responses/resp_public")).status_code,200)
       raw=payload(previous_response_id="resp_public",request_id="resp_next")
       self.assertEqual((await c.post("/v1/responses",content=raw)).status_code,200);self.assertEqual(D_bodies[-1],raw)
     self.assertFalse(app.state.placement.leases)
   self.assertEqual(len(helper_calls),1)

 async def test_native_validation_rejection_returns_original_D_error_and_body_once(self):
  calls=[];events=[]
  async def backend(req):
   raw=await req.aread();calls.append((req.url.host,raw))
   if req.url.host=="p":return httpx.Response(400,json={"error":{"message":"helper rejected"}})
   return httpx.Response(400,headers=[(b"content-type",b"application/json"),(b"x-duplicate",b"one"),(b"x-duplicate",b"two")],content=b'{"error":{"type":"native_error","message":"original invalid request"}}')
  async with httpx.AsyncClient(transport=NativePDTransport(PRODUCER,httpx.MockTransport(backend),trace=lambda e,**f:events.append((e,f))))as c:
   original=payload(temperature=-1);res=await c.post("http://a/v1/responses",content=original)
  self.assertEqual([h for h,b in calls],["p","a"]);self.assertEqual(calls[1][1],original);self.assertEqual(res.status_code,400)
  self.assertEqual(res.content,b'{"error":{"type":"native_error","message":"original invalid request"}}')
  self.assertEqual(res.headers.get_list("x-duplicate"),["one","two"])
  self.assertIn("pd_helper_native_rejection",[e for e,f in events]);self.assertNotIn("pd_helper_completed",[e for e,f in events])
 async def test_complete_helper_audit_artifacts_have_exact_wire_and_private_mode(self):
  events=[];calls=[]
  helper_wire=json.dumps(dict(usage={"output_tokens":1},kv_transfer_params=KV)).encode()
  async def backend(req):
   calls.append((req.url.host,await req.aread()))
   return httpx.Response(200,content=helper_wire if req.url.host=="p"else b"D native wire")
  with tempfile.TemporaryDirectory()as root:
   async with httpx.AsyncClient(transport=NativePDTransport(PRODUCER,httpx.MockTransport(backend),trace=lambda e,**f:events.append((e,f)),audit_dir=root))as c:
    original=payload();res=await c.post("http://a/v1/responses",content=original)
   artifacts=[]
   for event,fields in events:
    for key in ["original_body_artifact","helper_body_artifact","artifact","native_body_artifact"]:
     if fields.get(key):artifacts.append(fields[key])
   self.assertEqual(len(artifacts),4);self.assertEqual(res.content,b"D native wire")
   import hashlib
   for a in artifacts:
    f=Path(a["path"]);raw=f.read_bytes();self.assertEqual(a["bytes"],len(raw));self.assertEqual(a["sha256"],hashlib.sha256(raw).hexdigest());self.assertEqual(f.stat().st_mode&0o777,0o600)
   raw={Path(a["path"]).name.split(".",1)[1]:Path(a["path"]).read_bytes()for a in artifacts}
   self.assertEqual(raw["original.body"],original);self.assertEqual(raw["helper.body"],calls[0][1]);self.assertEqual(raw["helper.wire"],helper_wire);self.assertEqual(raw["native.body"],calls[1][1])
   self.assertEqual(json.loads(raw["helper.body"])["max_output_tokens"],1);self.assertFalse(json.loads(raw["helper.body"])["store"])
 async def test_helper_timeout_returns502_without_D_dispatch(self):
  calls=[]
  async def backend(req):calls.append(req.url.host);await asyncio.Event().wait()
  async with httpx.AsyncClient(transport=NativePDTransport(PRODUCER,httpx.MockTransport(backend),helper_timeout=.01))as c:
   res=await c.post("http://a/v1/responses",content=payload())
  self.assertEqual(res.status_code,502);self.assertEqual(calls,["p"])
