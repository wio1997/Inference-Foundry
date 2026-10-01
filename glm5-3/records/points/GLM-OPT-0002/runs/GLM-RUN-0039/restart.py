import asyncio,json,hashlib,pathlib,sys,subprocess,os,time,typing,importlib,re,signal
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
from pydantic import BaseModel
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
from sse_observer import NativeSSEObserver
start_watchdog()
r=pathlib.Path(__file__).parent;d=r/"restart";d.mkdir();attempts=[];gateway=None
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
 obj=ResponsesResponse.model_validate(value);assert obj.status in ["completed","incomplete"] and not value.get("error") and obj.usage is not None
 usage=obj.usage.model_dump();assert usage["output_tokens"]>0 and usage["input_tokens"]>0 and usage["total_tokens"]==usage["input_tokens"]+usage["output_tokens"]
 assert obj.max_output_tokens==64 and usage["output_tokens"]<=64
 return usage

async def main():
 global gateway
 import httpx
 journal=r/"fault_state.json";prior=json.loads(journal.read_text())
 assert not prior["open"] and all(not x["faulted"]for x in prior["groups"].values())
 groups=json.loads((r/"execution_groups.json").read_text());assert len(groups)==1 and prior["groups"][groups[0]["id"]]["epoch"]==groups[0]["epoch"]
 replicas=[{"id":"DP0","url":"http://172.16.10.166:9081"},{"id":"DP1","url":"http://172.16.10.166:9082"},{"id":"DP2","url":"http://172.16.10.167:9900"},{"id":"DP3","url":"http://172.16.10.167:9901"}]
 env=dict(os.environ,GLM_REPLICAS=json.dumps(replicas),GLM_EXECUTION_GROUPS=json.dumps(groups),GLM_GROUP_FAULT_STATE_PATH=str(journal),GLM_ROUTER_TRACE_PATH=str(d/"router_trace.jsonl"),GLM_ROUTER_AUDIT_DIR=str(d/"native_wire"),PYTHONPATH="/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
 log=(d/"gateway.log").open("wb")
 gateway=subprocess.Popen([sys.executable,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/persistent_coupled_gateway.py","--port","8002"],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 async with httpx.AsyncClient(trust_env=False,timeout=180)as client:
  async def idle():
   for replica in replicas:
    response=await client.get(replica["url"]+"/metrics");response.raise_for_status();text=response.text;(d/("final_"+replica["id"]+".metrics")).write_text(text)
    for key in ["num_requests_running","num_requests_waiting"]:
     vals=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M);assert vals and all(float(v)==0 for v in vals)
  try:
   for _ in range(100):
    guard();assert gateway.poll() is None
    try:
     response=await client.get("http://127.0.0.1:8002/healthcheck")
     if response.status_code==200:break
    except httpx.HTTPError:pass
    await asyncio.sleep(.1)
   else:raise RuntimeError("same epoch restart API unavailable")
   current=json.loads(journal.read_text());assert current["open"] and all(not x["faulted"]for x in current["groups"].values())
   snap=(await client.get("http://127.0.0.1:8002/control/replicas")).json()
   assert all(x["native_owner_epoch"]==groups[0]["epoch"] and x["fault_journal_enabled"] and x["fault_journal_error"] is None and not x["group_faulted"]for x in snap["replicas"])
   atomic_json(d/"before.json",{"same_epoch_restart_admissible":True,"journal":current,"placement":snap,"journal_mode":oct(journal.stat().st_mode&0o777)})
   assert journal.stat().st_mode&0o777==0o600
   response_body={"model":"glm-52","input":"Reply with the single word READY.","max_output_tokens":64,"temperature":0,"store":False,"chat_template_kwargs":{"enable_thinking":False}}
   chat={"model":"glm-52","messages":[{"role":"user","content":"Explain admission control briefly."}],"max_tokens":128,"ignore_eos":True,"stream":True,"temperature":0,"seed":20260930,"stream_options":{"include_usage":True}}
   completion={"model":"glm-52","prompt":"Serving concurrency is ","max_tokens":64,"ignore_eos":True,"stream":True,"temperature":0,"seed":20260930,"stream_options":{"include_usage":True}}
   for name,path,body in [("chat","/v1/chat/completions",chat),("completion","/v1/completions",completion),("responses_json","/v1/responses",dict(response_body,stream=False)),("responses_sse","/v1/responses",dict(response_body,stream=True))]:
    guard();row={"id":name,"status":"started","output_credit":0};attempts.append(row);atomic_json(d/"attempts.json",attempts)
    bp=d/(name+".body.json");bp.write_bytes(json.dumps(body,separators=(",",":"),ensure_ascii=False).encode());row["body"]=ref(bp)
    raw=b"";wp=d/(name+".wire")
    try:
     async with client.stream("POST","http://127.0.0.1:8002"+path,content=bp.read_bytes(),headers={"Content-Type":"application/json","Accept-Encoding":"identity","X-Request-ID":r.name+"-restart-"+name})as reply:
      row["http_status"]=reply.status_code
      with wp.open("xb")as stream:
       async for block in reply.aiter_raw():stream.write(block);raw+=block
     row["wire"]=ref(wp);assert row["http_status"]==200
     if name.startswith("responses"):
      if body["stream"]:
       frames=[parse_frame(x)for x in raw.replace(b"\r\n",b"\n").split(b"\n\n")if x.strip()];frames=[x for x in frames if x is not None]
       assert frames[0]["type"]=="response.created" and [x["sequence_number"]for x in frames]==list(range(len(frames)))
       terminals=[x for x in frames if x["type"]=="response.completed"];assert len(terminals)==1
       value=terminals[0]["response"];atomic_json(d/(name+".events.json"),frames)
      else:value=json.loads(raw)
      usage=committed(value);assert value["status"]=="completed"
      text="".join(c.get("text","")for o in value["output"]for c in o.get("content",[])if c.get("type")=="output_text")
      assert text.strip()=="READY"
      row.update(usage=usage,output_credit=usage["output_tokens"],terminal="response.completed"if body["stream"]else"completed",unknown=False,done_sentinel="native Responses no [DONE]")
     else:
      observer=NativeSSEObserver(collect_contract=True);observer.feed(raw);contract=observer.contract()
      assert contract["done"] and not contract["unknown"] and not contract["native_error"] and contract["finish_reasons"]=={"0":"length"} and contract["usage"]["completion_tokens"]==body["max_tokens"]
      row.update(usage=contract["usage"],output_credit=contract["usage"]["completion_tokens"],contract=contract)
     row["status"]="completed"
    except BaseException as error:
     row.update(status="invalid",output_credit=0,error=str(error))
     if wp.exists():row["wire"]=ref(wp)
     raise
    finally:atomic_json(d/"attempts.json",attempts)
   for _ in range(100):
    snap=(await client.get("http://127.0.0.1:8002/control/replicas")).json()
    if all(x["active_requests"]==0 for x in snap["replicas"]):break
    await asyncio.sleep(.1)
   else:raise RuntimeError("restart gateway lease remains")
   assert all(not x["group_faulted"] and x["fault_journal_error"] is None for x in snap["replicas"])
   await idle();atomic_json(d/"final_placement.json",snap)
   # Byte-exact transparent client wire against original gateway-upstream audit.
   trace=[json.loads(l)for l in (d/"router_trace.jsonl").read_text().splitlines()];leases=[x for x in trace if x["event"]=="lease_acquired"];assert len(leases)==4
   for row in attempts:
    lease=next(x for x in leases if x["request_header_id"]==r.name+"-restart-"+row["id"])
    assert lease["body_sha256"]==row["body"]["sha256"]
    events=[x for x in trace if x.get("lease_id")==lease["lease_id"]]
    rel=[x for x in events if x["event"]=="lease_released"];assert len(rel)==1 and rel[0]["released"] and not rel[0]["backend_failure"]
    audit=next(x for x in events if x["event"]=="upstream_stream_contract");assert audit["audit_error"] is None
    upstream=pathlib.Path(audit["wire_path"]);assert ref(upstream)["sha256"]==audit["wire_sha256"] and upstream.read_bytes()==pathlib.Path(row["wire"]["path"]).read_bytes()
  finally:
   if gateway is not None and gateway.poll() is None:
    os.killpg(gateway.pid,signal.SIGTERM)
    try:await asyncio.to_thread(gateway.wait,15)
    except subprocess.TimeoutExpired:os.killpg(gateway.pid,signal.SIGKILL);gateway.wait()
   log.close()
   end=json.loads(journal.read_text());atomic_json(d/"journal_after_shutdown.json",end)
   assert not end["open"] and all(not x["faulted"]for x in end["groups"].values()),"clean journal close failed"
 atomic_json(d/"summary.json",{"valid":True,"new_completed_inference_requests":len(attempts),"new_effective_output_tokens":sum(x["output_credit"]for x in attempts),"native_epoch_unchanged":True,"gateway_clean_restart":True,"native_fault_injected":False,"limits":["Healthy sameepoch restart/transparentAPI E2E only; failure/unclean branches CPU42 separate","No native recover/newphysicalepoch/OSpowerloss/storage-disaster proof"]})
asyncio.run(main())
