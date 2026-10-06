import unittest,json,asyncio,tempfile,httpx
from pathlib import Path
from native_pd_transport import NativePDTransport,helper_body
from response_affinity_gateway_v8 import create_app
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
   self.assertEqual([e for e,f in events],["pd_helper_started","pd_helper_completed","pd_decoder_dispatch"])
 async def test_full_native_fallback_keeps_exact_body_query_for_unknown_and_stateful(self):
  calls=[]
  async def backend(req):calls.append((req.url.host,req.url.query,await req.aread()));return httpx.Response(200,content=b"native")
  async with httpx.AsyncClient(transport=NativePDTransport(PRODUCER,httpx.MockTransport(backend)))as c:
   cases=[payload(previous_response_id="resp_prior"),payload(background=True),payload(tools=[{}]),payload(input=[{"type":"unknown"}]),payload(extra_native_feature=True),payload(max_output_tokens=0),b"{invalid"]
   for raw in cases:
    res=await c.post("http://a/v1/responses?native=1",content=raw);self.assertEqual(res.content,b"native");self.assertEqual(calls[-1],("a",b"native=1",raw))
   await c.get("http://a/v1/responses/resp_id?stream=true")
  self.assertEqual(len(calls),len(cases)+1);self.assertTrue(all(x[0]=="a"for x in calls))
 async def test_helper_failure_or_invalid_metadata_never_dispatches_D(self):
  for value,status in [(dict(error="bad"),400),(dict(usage={"output_tokens":1},kv_transfer_params={**KV,"remote_port":28001}),200),(dict(usage={"output_tokens":2},kv_transfer_params=KV),200)]:
   calls=[]
   async def backend(req):calls.append(req.url.host);return httpx.Response(status,json=value)
   async with httpx.AsyncClient(transport=NativePDTransport(PRODUCER,httpx.MockTransport(backend)))as c:res=await c.post("http://a/v1/responses",content=payload())
   self.assertEqual(calls,["p"]);self.assertEqual(res.status_code,502)
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
