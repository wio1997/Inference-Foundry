import asyncio,json,hashlib,pathlib,sys,subprocess,os,time,typing,importlib,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import guard
from pydantic import BaseModel
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
r=pathlib.Path(__file__).parent;attempts=[];gateway=None
def ref(p):
 raw=p.read_bytes();return {"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
classes={}
# Native's StreamingResponsesResponse alias omits some actual generic renderer
# classes. Recover actual installed response event models, with native overrides.
for modname in ["openai.types.responses","vllm.entrypoints.openai.responses.streaming_events","vllm.entrypoints.openai.responses.protocol"]:
 module=importlib.import_module(modname)
 for name,cls in vars(module).items():
  if not isinstance(cls,type) or not issubclass(cls,BaseModel) or "type"not in cls.model_fields:continue
  for tag in typing.get_args(cls.model_fields["type"].annotation):
   if isinstance(tag,str) and (tag.startswith("response.") or tag=="error"):classes[tag]=cls
atomic_json(r/"native_event_models.json",{tag:cls.__module__+"."+cls.__name__ for tag,cls in classes.items()})
def parse_frame(frame):
 lines=frame.decode("utf8").splitlines();data="\n".join(x[5:].lstrip(" ")for x in lines if x.startswith("data:"));event="\n".join(x[6:].lstrip(" ")for x in lines if x.startswith("event:"))
 if not data:return None
 obj=json.loads(data);tag=obj.get("type");assert tag in classes,("unknown native event",tag)
 classes[tag].model_validate(obj);assert event==tag and tag!="error" and not obj.get("error"),obj
 return obj
def committed(value):
 obj=ResponsesResponse.model_validate(value);assert obj.status in ["completed","incomplete"] and obj.error is None and obj.usage is not None
 usage=obj.usage.model_dump();assert usage["output_tokens"]>0 and usage["input_tokens"]>0 and usage["total_tokens"]==usage["input_tokens"]+usage["output_tokens"]
 assert obj.max_output_tokens==64 and usage["output_tokens"]<=64
 return usage
async def main():
 global gateway
 import httpx
 env=dict(os.environ,GLM_REPLICAS=json.dumps([{"id":"DP0","url":"http://172.16.10.166:9081"},{"id":"DP1","url":"http://172.16.10.167:9900"}]),GLM_ROUTER_TRACE_PATH=str(r/"router_trace.jsonl"),GLM_ROUTER_AUDIT_DIR=str(r/"router_wire"),PYTHONPATH="/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
 with (r/"gateway.log").open("ab")as log:gateway=subprocess.Popen(["python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/replica_gateway.py","--host","0.0.0.0","--port","8002"],stdout=log,stderr=subprocess.STDOUT,env=env)
 async with httpx.AsyncClient(trust_env=False,timeout=120)as client:
  for _ in range(100):
   guard()
   try:
    response=await client.get("http://172.16.10.166:8002/healthcheck")
    if response.status_code==200:break
   except httpx.HTTPError:pass
   await asyncio.sleep(.2)
  else:raise RuntimeError("owned diagnostic gateway readiness failed")
  async def request(name,base,method,path,body=None,expected_status=200):
   guard();entry={"id":name,"method":method,"base":base,"path":path,"expected_status":expected_status,"started_at":utc(),"outcome":"started","output_credit":0};attempts.append(entry);atomic_json(r/"attempts.json",attempts)
   rawbody=json.dumps(body,separators=(",",":")).encode()if body is not None else b"";bp=r/(name+".body");bp.write_bytes(rawbody);entry["body"]=ref(bp)
   blocks=[];events=[];buf=b"";first=None;terminal=[];started=time.monotonic();wp=r/(name+".wire")
   try:
    async with client.stream(method,base+path,content=rawbody,headers={"content-type":"application/json","accept-encoding":"identity","x-request-id":r.name+"-"+name})as response:
     entry["http_status"]=response.status_code;entry["headers"]=list(response.headers.multi_items());ctype=response.headers.get("content-type","").split(";")[0]
     with wp.open("xb")as wire:
      async for block in response.aiter_raw():
       wire.write(block);blocks.append(block)
       if ctype=="text/event-stream":
        buf+=block;assert len(buf)<=4*1024*1024,"oversized native event"
        while b"\n\n"in buf:
         frame,buf=buf.split(b"\n\n",1);obj=parse_frame(frame)
         if obj is not None:
          events.append(obj)
          if first is None and isinstance(obj.get("delta"),str) and obj["delta"] and obj["type"].endswith(".delta"):first=time.monotonic()-started
          if obj["type"]=="response.completed":terminal.append(obj)
     raw=b"".join(blocks);entry["wire"]=ref(wp);entry["elapsed_s"]=time.monotonic()-started;entry["first_output_s"]=first
     assert response.status_code==expected_status,(name,response.status_code,raw[:500].decode(errors="replace"))
     if expected_status!=200:
      obj=json.loads(raw);assert isinstance(obj.get("error"),dict) and obj["error"].get("code")==expected_status
      entry.update(outcome="expected_rejection",error_response=obj,output_credit=0)
     elif ctype=="text/event-stream":
      assert not buf.strip() and len(terminal)==1 and events[0]["type"]=="response.created"
      assert [x["sequence_number"]for x in events]==list(range(len(events)))
      ids={x["response"]["id"]for x in events if isinstance(x.get("response"),dict)};assert len(ids)==1
      value=terminal[0]["response"];usage=committed(value);ep=r/(name+".events.json");atomic_json(ep,events)
      entry.update(outcome="completed",usage=usage,output_credit=usage["output_tokens"],native_response_id=value["id"],status=value["status"],event_count=len(events),events=ref(ep),terminal_type="response.completed",done_sentinel="not emitted by installed native Responses SSE",unknown=False)
     else:
      value=json.loads(raw);usage=committed(value);entry.update(outcome="completed",usage=usage,output_credit=usage["output_tokens"],native_response_id=value["id"],status=value["status"],unknown=False)
   except BaseException as error:
    entry.update(outcome="invalid",error_type=type(error).__name__,error=str(error),output_credit=0)
    if wp.exists():entry["wire"]=ref(wp)
    raise
   finally:entry["finished_at"]=utc();atomic_json(r/"attempts.json",attempts)
  body={"model":"glm-52","input":"Reply with the single word READY.","max_output_tokens":64,"temperature":0,"store":False,"chat_template_kwargs":{"enable_thinking":False}}
  dp0="http://172.16.10.166:9081";dp1="http://172.16.10.167:9900";proxy="http://172.16.10.166:8002"
  for name,base,stream in [("direct0_json",dp0,False),("direct1_sse",dp1,True),("proxy_json",proxy,False),("proxy_sse",proxy,True)]:
   await request(name,base,"POST","/v1/responses",dict(body,stream=stream))
  bad=dict(body,background=True,store=True)
  await request("direct_background",dp0,"POST","/v1/responses",bad,400)
  await request("proxy_background",proxy,"POST","/v1/responses",bad,400)
  unknown="resp_"+r.name+"_missing"
  for name,base in [("direct_unknown",dp1),("proxy_unknown",proxy)]:
   await request(name,base,"GET","/v1/responses/"+unknown+"?stream=false&starting_after=3",None,404)
  await request("proxy_previous",proxy,"POST","/v1/responses",dict(body,previous_response_id=unknown),404)
  values={e["id"]:e for e in attempts};assert values["direct_background"]["error_response"]==values["proxy_background"]["error_response"] and values["direct_unknown"]["error_response"]==values["proxy_unknown"]["error_response"]
  for rank,base in [("DP0",dp0),("DP1",dp1)]:
   for _ in range(60):
    response=await client.get(base+"/metrics");response.raise_for_status();text=response.text
    idle=all((nums:=re.findall(r"^vllm:"+name+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M)) and all(float(n)==0 for n in nums)for name in ["num_requests_running","num_requests_waiting"])
    if idle:break
    await asyncio.sleep(.2)
   assert idle; (r/("final_"+rank+".metrics")).write_text(text)
  for _ in range(60):
   placement=(await client.get(proxy+"/control/replicas")).json()
   if all(x["active_requests"]==0 for x in placement["replicas"]):break
   await asyncio.sleep(.1)
  assert all(x["active_requests"]==0 for x in placement["replicas"]);atomic_json(r/"final_placement.json",placement)
async def owned():
 async def watch():
  while True:guard();await asyncio.sleep(2)
 run=asyncio.create_task(main());watcher=asyncio.create_task(watch())
 try:
  done,_=await asyncio.wait([run,watcher],return_when=asyncio.FIRST_COMPLETED)
  if watcher in done:watcher.result();raise RuntimeError("owner watcher stopped")
  run.result()
 finally:
  run.cancel();watcher.cancel();await asyncio.gather(run,watcher,return_exceptions=True)
  if gateway is not None:
   gateway.terminate()
   try:gateway.wait(timeout=10)
   except subprocess.TimeoutExpired:gateway.kill();gateway.wait()
   atomic_json(r/"gateway_terminal.json",{"pid":gateway.pid,"exit_code":gateway.returncode,"owned":True})
asyncio.run(owned())
assert len(attempts)==9 and sum(x["outcome"]=="completed"for x in attempts)==4 and sum(x["outcome"]=="expected_rejection"for x in attempts)==5
atomic_json(r/"responses_summary.json",{"valid":True,"new_client_attempts":9,"new_completed_inference_requests":4,"new_effective_output_tokens":sum(x["output_credit"]for x in attempts),"expected_rejections":5,"native_store":False,"contract":"native typed Responses JSON and named SSE response.completed usage; native does not emit Chat DONE sentinel","limits":["No native state-store/background execution support enabled; no stateful affinity claim","Short protocol windows are not performance/capacity/KEEP","ConfiguredGLM no-tools/stateless Responses only; additional native branches remain open"]})
