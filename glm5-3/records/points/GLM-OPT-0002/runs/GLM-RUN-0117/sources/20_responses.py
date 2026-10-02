import asyncio,json,hashlib,pathlib,sys,subprocess,os,time,typing,importlib,re,signal,math
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
from pydantic import BaseModel
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
start_watchdog()
r=pathlib.Path(__file__).parent;attempts=[];gateway=None;logs=[];credit={};events=[]
BASE="http://127.0.0.1:8002";NATIVE="http://172.16.10.166:9081";NATIVES={"D0":NATIVE,"D1":"http://172.16.10.167:9900"};owners=r/"response_owners.json";journal=pathlib.Path(str(owners)+".fault")
native_background_owned={}
groups=json.loads((r/"execution_groups.json").read_text());config=[{"id":k,"url":v}for k,v in NATIVES.items()]
def ref(p):
 raw=p.read_bytes();return {"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
def event(kind,**kw):events.append({"at":utc(),"event":kind,**kw});atomic_json(r/"state_events.json",events)
classes={}
for modname in ["openai.types.responses","vllm.entrypoints.openai.responses.streaming_events","vllm.entrypoints.openai.responses.protocol"]:
 for name,cls in vars(importlib.import_module(modname)).items():
  if isinstance(cls,type)and issubclass(cls,BaseModel)and"type"in cls.model_fields:
   for tag in typing.get_args(cls.model_fields["type"].annotation):
    if isinstance(tag,str)and(tag.startswith("response.")or tag=="error"):classes[tag]=cls
atomic_json(r/"native_event_models.json",{k:c.__module__+"."+c.__name__ for k,c in classes.items()})
def response_commit(value,expected=32):
 obj=ResponsesResponse.model_validate(value);assert obj.status in ["completed","incomplete"]and not value.get("error")and obj.usage is not None
 u=obj.usage.model_dump();assert u["input_tokens"]>0 and u["output_tokens"]==expected and u["total_tokens"]==u["input_tokens"]+u["output_tokens"]
 assert obj.max_output_tokens==expected
 if obj.status=="incomplete":assert obj.incomplete_details is not None and obj.incomplete_details.reason=="max_output_tokens"
 old=credit.setdefault(obj.id,{"usage":u,"status":obj.status});assert old["usage"]==u
 return u
def sse(raw,start_at=0):
 events=[]
 for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
  lines=frame.decode().splitlines();data="\n".join(x[5:].lstrip(" ")for x in lines if x.startswith("data:"));tag="\n".join(x[6:].strip()for x in lines if x.startswith("event:"))
  if not data:continue
  assert data!="[DONE]";x=json.loads(data);assert x["type"] in classes and x["type"]==tag and x["type"]!="error" and not x.get("error")
  classes[x["type"]].model_validate(x);events.append(x)
 assert [x["sequence_number"]for x in events]==list(range(start_at,start_at+len(events)))
 terminal=[x for x in events if x["type"]=="response.completed"];assert len(terminal)==1 and events[-1]==terminal[0]
 ids={x["response"]["id"]for x in events if "response"in x};assert len(ids)==1
 return terminal[0]["response"],events
async def main():
 global gateway
 import httpx
 async with httpx.AsyncClient(trust_env=False,timeout=180,headers={"accept-encoding":"identity"})as client:
  async def metrics(label,need_idle=False):
   vals={k:0 for k in ["num_requests_running","num_requests_waiting","generation_tokens_total"]}
   for key,url in NATIVES.items():
    res=await client.get(url+"/metrics");res.raise_for_status();p=r/(label+"_"+key+".metrics");p.write_bytes(res.content)
    for metric in vals:
     rows=re.findall(r"^vllm:"+metric+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",res.text,re.M);assert rows;vals[metric]+=sum(float(x)for x in rows)
   if need_idle:assert vals["num_requests_running"]==vals["num_requests_waiting"]==0
   event("metrics",label=label,values=vals);return vals
  async def idle(label):
   for i in range(90):
    vals=await metrics(label+"_"+str(i))
    if vals["num_requests_running"]==vals["num_requests_waiting"]==0:return vals
    await asyncio.sleep(.5)
   raise RuntimeError("native background/cancel drain notverified")
  async def start(label):
   global gateway
   guard();env=dict(os.environ,GLM_PD_NATIVE_PLANS=json.dumps({NATIVES["D1"]:dict(producer=next(x for x in json.loads((r.parent/"GLM-RUN-0109/planned_launch.json").read_text())if x["node"]=="166"),decoder=next(x for x in json.loads((r.parent/"GLM-RUN-0109/planned_launch.json").read_text())if x["node"]=="167"),connector_source_sha256="f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533")}),GLM_PLACEMENT_POLICY="shape_split",GLM_SHAPE_SPLIT=json.dumps(dict(input_threshold_bytes=8192,prefill_members=["D1"],decode_members=["D0"])),GLM_REPLICAS=json.dumps(config),GLM_PD_AUDIT_DIR=str(r/"native_PD_raw"),GLM_PD_PRODUCERS=json.dumps({NATIVES["D1"]:dict(url=NATIVES["D0"],remote_host="172.16.10.166",remote_port=28000,dcp_size=16)}),GLM_EXECUTION_GROUPS=json.dumps(groups),GLM_RESPONSE_OWNER_STATE_PATH=str(owners),GLM_ROUTER_TRACE_PATH=str(r/"router_trace.jsonl"),GLM_ROUTER_AUDIT_DIR=str(r/"router_wire"),PYTHONPATH="/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
   log=(r/(label+".gateway.log")).open("wb");logs.append(log)
   gateway=subprocess.Popen(["python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/response_affinity_gateway_v11.py","--port","8002"],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   for _ in range(100):
    assert gateway.poll()is None
    try:
     res=await client.get(BASE+"/healthcheck")
     if res.status_code==200:break
    except httpx.HTTPError:pass
    await asyncio.sleep(.1)
   else:raise RuntimeError("owned affinitygateway notready")
   event("gateway_started",label=label,pid=gateway.pid)
  async def stop(label):
   global gateway
   os.kill(gateway.pid,signal.SIGTERM);await asyncio.to_thread(gateway.wait,15);assert gateway.returncode in [0,-signal.SIGTERM]
   f=json.loads(journal.read_text());assert f["open"]is False and all(not v["faulted"]for v in f["groups"].values())
   event("gateway_clean_stopped",label=label,exit_code=gateway.returncode,fault_journal=ref(journal));gateway=None
  async def request(name,method,path,payload=None,expected_status=200):
   guard();raw=json.dumps(payload,ensure_ascii=False,separators=(",",":")).encode()if payload is not None else b"";body=r/(name+".body");body.write_bytes(raw)
   row={"id":name,"method":method,"path":path,"body":ref(body),"started_at":utc(),"outcome":"started"};attempts.append(row);atomic_json(r/"attempts.json",attempts);wp=r/(name+".wire")
   async with client.stream(method,BASE+path,content=raw,headers={"content-type":"application/json","x-request-id":r.name+"-"+name})as res:
    row["http_status"]=res.status_code;row["headers"]=res.headers.multi_items()
    with wp.open("wb")as f:
     async for block in res.aiter_raw():f.write(block)
   row["wire"]=ref(wp);assert row["http_status"]==expected_status
   for _ in range(100):
    snap=(await client.get(BASE+"/control/replicas")).json()
    if not any(x["active_requests"]for x in snap["replicas"]):break
    await asyncio.sleep(.02)
   else:raise RuntimeError("HTTP lease not released")
   t=[json.loads(x)for x in (r/"router_trace.jsonl").read_text().splitlines()];leases=[x for x in t if x["event"]=="lease_acquired"and x.get("request_header_id")==r.name+"-"+name];assert len(leases)==1;lease=leases[0]
   wires=[x for x in t if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];assert len(wires)==1
   wire=wires[0];assert wire["audit_error"]is None and wire["wire_bytes"]==row["wire"]["bytes"] and wire["wire_sha256"]==row["wire"]["sha256"] and pathlib.Path(wire["wire_path"]).read_bytes()==wp.read_bytes()
   release=[x for x in t if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]];assert len(release)==1 and release[0]["released"]and not release[0]["backend_failure"]
   assert lease["replica"]in NATIVES and lease["body_sha256"]==row["body"]["sha256"]
   if method=="POST"and path.split("?",1)[0]=="/v1/responses"and isinstance(payload,dict)and type(payload.get("max_output_tokens"))is int and payload["max_output_tokens"]>0:
    assert lease["output_budget"]==payload["max_output_tokens"]
    row["gateway_native_output_budget"]=lease["output_budget"]

   from response_affinity import request_owner_key
   owner_key=request_owner_key(path.split("?",1)[0],method,raw)
   owner=json.loads(owners.read_text())["owners"].get(owner_key)if owner_key else None
   if owner is not None:assert owner["replica"]==lease["replica"]
   row["native_owner"]=lease["replica"]
   row.update(outcome="native_http_valid",lease_id=lease["lease_id"],affinity_applied=lease["affinity_applied"],native_wire={"path":wire["wire_path"],"bytes":wire["wire_bytes"],"sha256":wire["wire_sha256"]},finished_at=utc());atomic_json(r/"attempts.json",attempts)
   if "text/event-stream"in res.headers.get("content-type","")and path.startswith("/v1/chat/completions"):
    from sse_observer import NativeSSEObserver
    ob=NativeSSEObserver(max_bytes=2097152,collect_contract=True);ob.feed(wp.read_bytes());value=ob.contract();assert value["done"]and not value["native_error"]and not value["unknown"];return value,row
   return (sse(wp.read_bytes(),1 if method=="GET"and "starting_after=0"in path else 0)[0]if "text/event-stream"in res.headers.get("content-type","")else json.loads(wp.read_bytes())),row
  def payload(**kw):
   return {"model":"glm-52","input":"Explain a concrete GLM serving example. 上海","max_output_tokens":32,"temperature":0,"ignore_eos":True,"seed":20260930,"store":True,**kw}
  try:
   before=await idle("initial");await start("first")
   for label in ["gateway_first","gateway_second"]:
    if label=="gateway_second":
     res=await client.delete(BASE+"/control/replicas/D0");res.raise_for_status()
    env=dict(os.environ,GLM_TOOL_REPLICA=label,GLM_TOOL_URL=BASE,GLM_TOOL_METRICS_URLS=json.dumps(NATIVES))
    z=await asyncio.to_thread(subprocess.run,[sys.executable,str(r/"tools.py")],env=env,capture_output=True,timeout=1000)
    (r/(label+".tools.stdout")).write_bytes(z.stdout);(r/(label+".tools.stderr")).write_bytes(z.stderr);z.check_returncode()
    if label=="gateway_second":
     res=await client.post(BASE+"/control/replicas",json=config[0]);res.raise_for_status()
    if label=="gateway_first":
     # A harmless native models GET rotates the next tied-endpoint choice.
     res=await client.get(BASE+"/v1/models");res.raise_for_status()
   await idle("after_tools")
   # Drain only logical P admission; PP2 D has no KV connector, V11 must preserve original native D path before any P helper.
   res=await client.delete(BASE+"/control/replicas/D0");res.raise_for_status()
   for label,stream in [("compat_json",False),("compat_sse",True)]:
    text=("This is a padding line for the inference state test.\n"*1024)+"\nExplain a concrete GLM serving example."
    value,row=await request(label,"POST","/v1/responses",payload(input=text,stream=stream,request_id="resp_glm_run117_"+label,cache_salt=r.name+"-NEW-"+label))
    response_commit(value);assert row["native_owner"]=="D1"and value["usage"]["input_tokens_details"]["cached_tokens"]==0and value["usage"]["input_tokens"]>11000
    retrieved,rr=await request(label+"_retrieve","GET","/v1/responses/"+value["id"]);assert retrieved==value and rr["affinity_applied"]
   canonical=json.loads((r/"canonical_81932.body.json").read_text());canonical.pop("kv_transfer_params",None);canonical.update(max_tokens=64,stream=True,stream_options=dict(include_usage=True),cache_salt=r.name+"-NEW-cold-chat",return_token_ids=True)
   chat,row=await request("compat_chat","POST","/v1/chat/completions",canonical);assert row["native_owner"]=="D1"and chat["usage"]==dict(prompt_tokens=81932,completion_tokens=64,total_tokens=81996)and chat["finish_reasons"]=={"0":"length"}
   credit["compat_chat"]={"usage":chat["usage"],"api":"chat","status":"length"}
   res=await client.post(BASE+"/control/replicas",json=config[0]);res.raise_for_status();await idle("after_public_compatibility")
   base,br=await request("base_json","POST","/v1/responses?glm_contract=state",payload(request_id="resp_glm_run117_base"));response_commit(base);assert base["id"]=="resp_glm_run117_base"
   got,gr=await request("retrieve_base","GET","/v1/responses/"+base["id"]+"?stream=false&starting_after=0");assert got==base and gr["affinity_applied"]
   res=await client.delete(BASE+"/control/replicas/D0");res.raise_for_status()
   other,otherrow=await request("other_base_json","POST","/v1/responses",payload(request_id="resp_glm_run117_other"))
   res=await client.post(BASE+"/control/replicas",json=config[0]);res.raise_for_status()
   response_commit(other);assert other["id"]=="resp_glm_run117_other"
   assert otherrow["native_owner"]!=br["native_owner"]
   othergot,othergetrow=await request("retrieve_other","GET","/v1/responses/"+other["id"]);assert othergot==other and othergetrow["affinity_applied"]
   # Conflicting custom/previous owners must reject without dispatching an RPC.
   conflict=payload(request_id=other["id"],previous_response_id=base["id"])
   raw=json.dumps(conflict,ensure_ascii=False,separators=(",",":")).encode();(r/"owner_conflict.body").write_bytes(raw)
   trace_before=(r/"router_trace.jsonl").read_bytes()
   res=await client.post(BASE+"/v1/responses",content=raw,headers={"content-type":"application/json"})
   (r/"owner_conflict.wire").write_bytes(res.content);assert res.status_code==503
   assert (r/"router_trace.jsonl").read_bytes()==trace_before
   event("cross_owner_conflict_rejected_before_RPC",status=res.status_code,body=ref(r/"owner_conflict.body"),wire=ref(r/"owner_conflict.wire"))
   otherchild,ocr=await request("other_previous_json","POST","/v1/responses",payload(previous_response_id=other["id"]))
   response_commit(otherchild);assert ocr["affinity_applied"]and ocr["native_owner"]==otherrow["native_owner"]and otherchild["previous_response_id"]==other["id"]
   child,cr=await request("previous_json","POST","/v1/responses",payload(previous_response_id=base["id"]));response_commit(child);assert child["previous_response_id"]==base["id"]and cr["affinity_applied"];assert child["usage"]["input_tokens"]>base["usage"]["input_tokens"]
   streamed,sr=await request("stored_sse","POST","/v1/responses",payload(stream=True));response_commit(streamed);data,ev=sse(pathlib.Path(sr["wire"]["path"]).read_bytes());atomic_json(r/"stored_sse.events.json",ev)
   got,_=await request("retrieve_sse","GET","/v1/responses/"+streamed["id"]);response_commit(got);assert got["id"]==streamed["id"]
   # Sampling/logprob contracts exercise installed distributed native paths.
   chat={"model":"glm-52","messages":[{"role":"user","content":"Give several scheduling examples."}],"max_tokens":32,"temperature":.7,"top_k":20,"top_p":.9,"seed":20260930,"ignore_eos":True,"stream":False,"logprobs":True,"top_logprobs":3}
   val,_=await request("random_chat_logprobs","POST","/v1/chat/completions",chat);assert val["usage"]["completion_tokens"]==32 and val["choices"][0]["finish_reason"]=="length";lp=val["choices"][0]["logprobs"]["content"];assert lp and all(math.isfinite(x["logprob"])and len(x["top_logprobs"])>=3 for x in lp)
   credit["chat_random"]={"usage":val["usage"],"status":"length","api":"chat"}
   pending,pr=await request("background_cancel_create","POST","/v1/responses",payload(background=True,max_output_tokens=8192,previous_response_id=base["id"]));assert pending["status"]in ["queued","in_progress"]and pending["usage"]is None and pr["affinity_applied"]
   native_background_owned[pending["id"]]=NATIVES[pr["native_owner"]]
   for _ in range(100):
    m=await metrics("background_live")
    if m["num_requests_running"]>0:break
    await asyncio.sleep(.1)
   else:raise RuntimeError("native background active job neverobserved")
   snap=(await client.get(BASE+"/control/replicas")).json();assert all(x["active_requests"]==0 for x in snap["replicas"]);event("native_job_outlives_zero_HTTP_leases",native=m,placement=snap)
   await stop("while_native_background_active");await start("restart")
   polled,rr=await request("retrieve_after_restart","GET","/v1/responses/"+pending["id"]);assert polled["id"]==pending["id"]and rr["affinity_applied"]
   cancelled,cc=await request("cancel_after_restart","POST","/v1/responses/"+pending["id"]+"/cancel");assert cancelled["status"]=="cancelled"and cc["affinity_applied"];event("background_cancelled_zero_credit",native_id=pending["id"],output_credit=0);native_background_owned.pop(pending["id"],None);await idle("after_cancel")
   got,gr=await request("base_after_restart","GET","/v1/responses/"+base["id"]);assert got==base and gr["affinity_applied"]
   got,gr=await request("other_after_restart","GET","/v1/responses/"+other["id"]);assert got==other and gr["affinity_applied"]
   removed=br["native_owner"];res=await client.delete(BASE+"/control/replicas/"+removed);res.raise_for_status();assert len(res.json()["replicas"])==1
   trace_before=(r/"router_trace.jsonl").read_bytes();res=await client.get(BASE+"/v1/responses/"+base["id"]);assert res.status_code==503
   (r/"drained_owner.wire").write_bytes(res.content);assert (r/"router_trace.jsonl").read_bytes()==trace_before
   res=await client.post(BASE+"/control/replicas",json=next(x for x in config if x["id"]==removed));res.raise_for_status()
   got,gr=await request("base_after_readd","GET","/v1/responses/"+base["id"]);assert got==base and gr["affinity_applied"]
   bg,bgr=await request("background_complete_create","POST","/v1/responses",payload(background=True))
   assert bg["status"]in ["queued","in_progress"]and bg["usage"]is None;native_background_owned[bg["id"]]=NATIVES[bgr["native_owner"]]
   for i in range(120):
    got,_=await request("background_poll_"+str(i),"GET","/v1/responses/"+bg["id"])
    if got["status"]in ["completed","incomplete"]:response_commit(got);native_background_owned.pop(bg["id"],None);break
    assert got["status"]in ["queued","in_progress"];await asyncio.sleep(.2)
   else:raise RuntimeError("bounded native background didnotcomplete")
   bs,bsr=await request("background_stored_sse","POST","/v1/responses",payload(background=True,stream=True));response_commit(bs)
   replay,replayrow=await request("background_sse_replay","GET","/v1/responses/"+bs["id"]+"?stream=true&starting_after=0");response_commit(replay);assert replay["id"]==bs["id"]and replayrow["affinity_applied"]
   invalid,ir=await request("native_invalid_parameter","POST","/v1/responses",payload(max_output_tokens=-1),expected_status=400);assert invalid.get("error")or invalid.get("message")or invalid.get("object")=="error"
   _,nr=await request("unknown_native404","GET","/v1/responses/resp_unknown_glm_run117",expected_status=404);assert not nr["affinity_applied"]
   await idle("final");snap=(await client.get(BASE+"/control/replicas")).json();atomic_json(r/"final_placement.json",snap);await stop("final")
   ownerdata=json.loads(owners.read_text());assert owners.stat().st_mode&0o777==0o600
   for key in [base["id"],other["id"],otherchild["id"],child["id"],streamed["id"],pending["id"],bg["id"],bs["id"]]:
    row=ownerdata["owners"][key];assert row=={"replica":row["replica"],"url":NATIVES[row["replica"]],"group":next(g["id"]for g in groups if row["replica"]in g["members"]),"epoch":next(g["epoch"]for g in groups if row["replica"]in g["members"])}
   summary={"valid":True,"new_inference_attempts":9,"new_completed_inference_requests":8,"new_effective_output_tokens":384,"cancelled_requests":1,"cancelled_output_credit":0,"native_credit":credit,"http_attempts":len(attempts),"native_STORE":True,"actual_native_cohort":json.loads((r/"adopted_model_identities.json").read_text()),"contracts":["Stored JSON/customnativeID/previous_response_id","Stored foreground/background typed Responses SSE completed+usage/noChatDONE; native starting_after replay retainssequence","Retrieve/cancel exactowner/privatejournal","Native background survives cleanproxyrestart with0HTTPleases/nativejobpositive","Logicaldrain/readd samephysicalepoch retains state","Native distributed random topk/topp/chat logprobs","Unknownnative404 preserved"],"limits":["Two nativeAPI store domains validated; no physicalAPI fault/state replication proof; concurrent native duplicate-ID acceptance untested","Proxy journal containsownershippointers, not nativeAPIstate replication; nativeAPIrestart changesepoch/unavailable","Cancelled nativegeneration mayincrease nativecounters but no outputcredit; GETpolls not newinference","Native logprobs finite/topcount validatesfunctionalwire, no quality/accuracy comparison","No throughput/KEEP/stability/hardwarebound"]}
   summary["new_completed_inference_requests"]=len(credit);summary["new_effective_output_tokens"]=sum(x["usage"].get("output_tokens",x["usage"].get("completion_tokens",0))for x in credit.values());summary["new_inference_attempts"]=len(credit)+1
   assert len(credit)==11 and summary["new_effective_output_tokens"]==384
   atomic_json(r/"responses_summary.json",summary);event("complete",outputs=384,completed=11,cancel_credit=0)
  finally:
   for ident in list(native_background_owned):
    try:
     guard();res=await client.post(native_background_owned[ident]+"/v1/responses/"+ident+"/cancel");path=r/("cleanup_"+ident+".wire");path.write_bytes(res.content);event("owned_native_background_cleanup",native_id=ident,status=res.status_code,wire=ref(path),output_credit=0)
    except BaseException as error:event("owned_native_background_cleanup_unknown",native_id=ident,error=str(error),output_credit=0)
   if gateway and gateway.poll()is None:
    os.kill(gateway.pid,signal.SIGTERM)
    try:await asyncio.to_thread(gateway.wait,15)
    except subprocess.TimeoutExpired:os.kill(gateway.pid,signal.SIGKILL);await asyncio.to_thread(gateway.wait)
   for log in logs:log.close()
asyncio.run(main())
