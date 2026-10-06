import asyncio,json,gzip,tempfile,unittest,copy,os
from pathlib import Path
from unittest.mock import patch
import httpx
from response_affinity_gateway import create_app
from response_affinity import ResponseOwnerIndex,ResponseAffinityPlacement,ResponsesOwnerObserver,request_owner_key
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
BASE=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0039/restart")
JSON_RAW=(BASE/"responses_json.wire").read_bytes();SSE_RAW=(BASE/"responses_sse.wire").read_bytes()
FIXTURE=json.loads(JSON_RAW);ResponsesResponse.model_validate(FIXTURE)
CONFIG=[{"id":"a","url":"http://a"},{"id":"b","url":"http://b"}]
GROUPS=[{"id":"ep","epoch":"native-e1","members":["a","b"]}]
class Wire(httpx.AsyncByteStream):
 def __init__(self,raw):self.raw=raw
 async def __aiter__(self):
  for i in range(0,len(self.raw),7):yield self.raw[i:i+7]
class CPUStores:
 """CPU fixture transport only; native protocol objects, no Engine/LLM/NPU calls."""
 def __init__(self):self.stores={"a":{},"b":{}};self.calls=[];self.counter=0;self.sse=False;self.encoded=False;self.failread=False
 def reply(self,code,raw,ctype="application/json"):
  headers=[(b"content-type",ctype.encode()),(b"x-duplicate",b"one"),(b"x-duplicate",b"two")]
  if self.encoded:raw=gzip.compress(raw);headers.append((b"content-encoding",b"gzip"))
  headers.append((b"content-length",str(len(raw)).encode()))
  return httpx.Response(code,headers=headers,stream=Wire(raw))
 async def __call__(self,req):
  node=req.url.host;body=await req.aread();self.calls.append((node,req.method,req.url.path,body,req.url.query))
  if req.url.path=="/health":return self.reply(200,b"OK","text/plain")
  if req.method=="POST"and req.url.path=="/v1/responses":
   data=json.loads(body)
   if data.get("previous_response_id")and data["previous_response_id"]not in self.stores[node]:return self.reply(404,b'{"error":{"type":"not_found_error","code":404}}')
   if data.get("stream")and self.sse:
    value=json.loads(SSE_RAW.decode().split("data: ",1)[1].split("\n",1)[0])["response"]
    self.stores[node][value["id"]]=copy.deepcopy(FIXTURE);self.stores[node][value["id"]]["id"]=value["id"]
    return self.reply(200,SSE_RAW,"text/event-stream")
   self.counter+=1;value=copy.deepcopy(FIXTURE);value["id"]=data.get("request_id","resp_cpu_"+str(self.counter));value["previous_response_id"]=data.get("previous_response_id")
   if data.get("background"):
    value.update(status="queued",background=True,usage=None,output=[])
   ResponsesResponse.model_validate(value)
   if data.get("store",True):self.stores[node][value["id"]]=value
   return self.reply(200,json.dumps(value,ensure_ascii=False).encode())
  key=req.url.path[len("/v1/responses/"):];cancel=key.endswith("/cancel")
  if cancel:key=key[:-7]
  value=self.stores[node].get(key)
  if value is None:return self.reply(404,b'{"error":{"type":"not_found_error","code":404}}')
  if cancel:value.update(status="cancelled",usage=None,output=[])
  return self.reply(200,json.dumps(value,ensure_ascii=False).encode())
class AffinityTests(unittest.IsolatedAsyncioTestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.store=CPUStores()
 def tearDown(self):self.temp.cleanup()
 def app(self,name="owners.json",groups=GROUPS):
  return create_app(CONFIG,transport=httpx.MockTransport(self.store),groups=groups,response_owner_state_path=str(self.root/name))
 async def client(self,app):return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://gateway")
 async def create(self,client,**kw):
  body=json.dumps({"model":"glm-53","input":"Return READY. 上海","max_output_tokens":2,**kw},ensure_ascii=False).encode()
  response=await client.post("/v1/responses",content=body,headers={"content-type":"application/json"})
  assert self.store.calls[-1][3]==body,"native sampling/input/body bytes must remain exact"
  return response
 async def test_json_previous_retrieve_cancel_background_and_custom_id(self):
  app=self.app()
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    one=(await self.create(c)).json();two=(await self.create(c)).json()
    self.assertNotEqual(app.state.response_owner_index.resolve(one["id"])["replica"],app.state.response_owner_index.resolve(two["id"])["replica"])
    owner=app.state.response_owner_index.resolve(one["id"])["replica"]
    value=await c.get("/v1/responses/"+one["id"]+"?stream=false&starting_after=0");self.assertEqual(value.status_code,200);self.assertEqual(self.store.calls[-1][0],owner);self.assertIn(b"starting_after=0",self.store.calls[-1][4])
    child=(await self.create(c,previous_response_id=one["id"])).json();self.assertEqual(self.store.calls[-1][0],owner)
    again=(await self.create(c,request_id=child["id"])).json();self.assertEqual(again["id"],child["id"]);self.assertEqual(self.store.calls[-1][0],owner)
    pending=(await self.create(c,previous_response_id=one["id"],background=True)).json();self.assertEqual(pending["status"],"queued");self.assertEqual(self.store.calls[-1][0],owner)
    cancelled=await c.post("/v1/responses/"+pending["id"]+"/cancel");self.assertEqual(cancelled.json()["status"],"cancelled");self.assertEqual(self.store.calls[-1][0],owner)
    self.assertFalse(app.state.placement.leases)
  self.assertEqual((self.root/"owners.json").stat().st_mode&0o777,0o600)
 async def test_actual_native_sse_split_encoded_and_exact_wire(self):
  self.store.sse=True;self.store.encoded=True;app=self.app()
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    response=await self.create(c,stream=True);self.assertEqual(response.content,SSE_RAW)
    self.assertEqual(response.headers.get_list("x-duplicate"),["one","two"])
    observer=ResponsesOwnerObserver()
    ids=[]
    for i in range(0,len(SSE_RAW),3):ids+=observer.feed(SSE_RAW[i:i+3])
    self.assertEqual(len(ids),1);self.assertIsNotNone(app.state.response_owner_index.resolve(ids[0]))
    await c.get("/v1/responses/"+ids[0]);self.assertEqual(self.store.calls[-1][0],"a")
 async def test_clean_restart_then_drain_same_epoch_readd(self):
  app=self.app()
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:ident=(await self.create(c)).json()["id"]
  app=self.app()
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    self.assertEqual((await c.get("/v1/responses/"+ident)).status_code,200)
    await c.delete("/control/replicas/a");before=len(self.store.calls)
    self.assertEqual((await c.get("/v1/responses/"+ident)).status_code,503);self.assertEqual(len(self.store.calls),before)
    await c.post("/control/replicas",json=CONFIG[0])
    self.assertEqual((await c.get("/v1/responses/"+ident)).status_code,200);self.assertEqual(self.store.calls[-1][0],"a")
 async def test_fault_and_native_epoch_change_do_not_relocate_owner(self):
  app=self.app()
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    ident=(await self.create(c)).json()["id"];lease=await app.state.placement.acquire_for_owner(1,0,app.state.response_owner_index.resolve(ident))
    await app.state.placement.release(lease,backend_failure=True);before=len(self.store.calls)
    self.assertEqual((await c.get("/v1/responses/"+ident)).status_code,503);self.assertEqual(len(self.store.calls),before)
  groups=[{**GROUPS[0],"epoch":"native-e2"}];app=self.app(groups=groups)
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    before=len(self.store.calls);self.assertEqual((await c.get("/v1/responses/"+ident)).status_code,503);self.assertEqual(len(self.store.calls),before)
    self.assertEqual((await c.get("/v1/responses/resp_unknown")).status_code,404)
 async def test_owner_journal_single_writer_corruption_and_io_failure(self):
  place=ResponseAffinityPlacement(CONFIG,groups=GROUPS);path=str(self.root/"direct.json");index=ResponseOwnerIndex(place,path)
  with self.assertRaises(BlockingIOError):ResponseOwnerIndex(place,path)
  lease=await place.acquire(2,1)
  with patch("response_affinity.os.replace",side_effect=OSError("fixture disk error")):
   with self.assertRaises(OSError):index.bind("resp_disk",lease)
  self.assertIsNone(index.resolve("resp_disk"));self.assertIsNotNone(index.error);await place.release(lease);index.close()
  Path(path).write_text('{"schema":1,"owners":{"resp_bad":{}}}')
  with self.assertRaises(ValueError):ResponseOwnerIndex(place,path)
 async def test_disabled_feature_preserves_original_stateless_default(self):
  app=create_app(CONFIG,transport=httpx.MockTransport(self.store),groups=[])
  async with app.router.lifespan_context(app):
   async with await self.client(app)as c:
    response=await self.create(c);self.assertEqual(response.status_code,200);self.assertEqual(app.state.response_owner_index.entries,{})
if __name__=="__main__":unittest.main()
