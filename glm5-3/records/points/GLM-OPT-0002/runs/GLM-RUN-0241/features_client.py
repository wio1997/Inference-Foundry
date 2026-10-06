from pathlib import Path
import asyncio,json,hashlib,sys,time,typing,importlib,re,math
r=Path(__file__).parent;sys.path.insert(0,str(r/"runtime_bundle"))
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
from pydantic import BaseModel
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
import httpx
start_watchdog()
mode=sys.argv[1];rows=[];credits={};native_background_owned={}
state=r.parent/"GLM-RUN-0125";trace=state/"router_trace.jsonl";owners=state/"response_owners.json"
NATIVES={"D0":"http://172.16.10.166:9081","D1":"http://172.16.10.167:9900"};BASE="http://127.0.0.1:8000"
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def persist():atomic_json(r/(mode+"_attempts.json"),rows)
classes={}
for modname in ["openai.types.responses","vllm.entrypoints.openai.responses.streaming_events","vllm.entrypoints.openai.responses.protocol"]:
 for name,cls in vars(importlib.import_module(modname)).items():
  if isinstance(cls,type)and issubclass(cls,BaseModel)and"type"in cls.model_fields:
   for tag in typing.get_args(cls.model_fields["type"].annotation):
    if isinstance(tag,str)and(tag.startswith("response.")or tag=="error"):classes[tag]=cls
atomic_json(r/"native_event_models.json",{k:c.__module__+"."+c.__name__ for k,c in classes.items()})
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

def chat(value,raw,stream):
 if stream:
  from sse_observer import NativeSSEObserver
  ob=NativeSSEObserver(collect_contract=True);ob.feed(raw);c=ob.contract();assert c["done"]and not c["unknown"]and not c["native_error"]
  tokens=[];content=[];reasoning=[]
  for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
   data=b"\n".join(x[5:].lstrip(b" ")for x in frame.splitlines()if x.startswith(b"data:"))
   if not data or data==b"[DONE]":continue
   v=json.loads(data)
   for x in v.get("choices",[]):
    tokens.extend(x.get("token_ids")or[]);d=x.get("delta",{});content.append(d.get("content")or"");reasoning.append(d.get("reasoning")or d.get("reasoning_content")or"")
  return dict(usage=c["usage"],token_ids=tokens,content="".join(content),reasoning="".join(reasoning),finish=c["finish_reasons"])
 choice=value["choices"][0];msg=choice["message"]
 return dict(usage=value["usage"],token_ids=choice.get("token_ids")or[],content=msg.get("content")or"",reasoning=msg.get("reasoning")or msg.get("reasoning_content")or"",finish={"0":choice["finish_reason"]})
def credit_chat(name,value,raw,stream,row):
 c=chat(value,raw,stream);u=c["usage"];assert len(c["token_ids"])==u["completion_tokens"]>0and u["total_tokens"]==u["prompt_tokens"]+u["completion_tokens"]
 row["chat"]=c;credits[name]=dict(owner=row["native_owner"],outputs=u["completion_tokens"],prompts=u["prompt_tokens"],kind="chat");persist();return c
def credit_response(value,row,expected=None):
 obj=ResponsesResponse.model_validate(value);u=value["usage"];assert obj.status in ["completed","incomplete"]and not value.get("error")and u["output_tokens"]>0and u["total_tokens"]==u["input_tokens"]+u["output_tokens"]
 if expected is not None:assert u["output_tokens"]==expected
 credits.setdefault(value["id"],dict(owner=row["native_owner"],outputs=u["output_tokens"],prompts=u["input_tokens"],kind="responses"));row["response_id"]=value["id"];row["usage"]=u;persist()
def text_response(value):
 return "".join(c.get("text","")for item in value["output"]if item["type"]=="message"for c in item["content"]if c["type"]=="output_text")
schema={"type":"object","properties":{"city":{"type":"string","enum":["Shanghai"]}},"required":["city"],"additionalProperties":False}
def response_body(**kw):
 return {"model":"glm-52","input":"Explain a concrete GLM serving example. 上海","max_output_tokens":32,"temperature":0,"ignore_eos":True,"seed":20260930,"store":True,"chat_template_kwargs":{"enable_thinking":False},**kw}
def chat_body(**kw):
 x=dict(model="glm-52",messages=[{"role":"user","content":"Compute 2+2. Answer the final value only."}],max_tokens=64,temperature=0,seed=20260930,return_token_ids=True,chat_template_kwargs={"enable_thinking":True});x.update(kw);return x
async def main():
 async with httpx.AsyncClient(trust_env=False,timeout=180,headers={"accept-encoding":"identity"})as client:
  async def metrics(label):
   out={}
   for key,url in NATIVES.items():
    v=await client.get(url+"/metrics");v.raise_for_status();f=r/(mode+"_"+label+"_"+key+".metrics");f.write_bytes(v.content);d={}
    for l in v.text.splitlines():
     if l.startswith("vllm:"):
      k=l.split("{")[0].split()[0];d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
    out[key]=d
   return out
  async def idle(label):
   for i in range(120):
    m=await metrics(label)
    if all(m[k].get("vllm:num_requests_running")==m[k].get("vllm:num_requests_waiting")==0for k in NATIVES):return m
    await asyncio.sleep(.5)
   raise RuntimeError("native idle not reached")
  async def req(name,method,path,body=None,owner=None,code=200,dispatch=True):
   guard();raw=json.dumps(body,ensure_ascii=False,separators=(",",":")).encode()if body is not None else b"";bp=r/(mode+"_"+name+".body");bp.write_bytes(raw)
   header=r.name+"-"+mode+"-"+name;row=dict(name=name,method=method,path=path,body=ref(bp),header_id=header,status="started");rows.append(row);persist();beforetrace=trace.read_bytes();wp=r/(mode+"_"+name+".wire");started=time.monotonic()
   async with client.stream(method,BASE+path,content=raw,headers={"content-type":"application/json","x-request-id":header})as res:
    row["http_status"]=res.status_code;row["headers"]=res.headers.multi_items()
    with wp.open("wb")as f:
     async for block in res.aiter_raw():f.write(block)
   row["wire"]=ref(wp);row["wall_s"]=time.monotonic()-started;persist();assert row["http_status"]==code,(name,row["http_status"],wp.read_text())
   for _ in range(100):
    snap=(await client.get(BASE+"/control/replicas")).json()
    if all(not x["active_requests"]for x in snap["replicas"]):break
    await asyncio.sleep(.02)
   else:raise RuntimeError("HTTP lease release absent")
   if not dispatch:
    assert trace.read_bytes()==beforetrace;row["status"]="prelease_rejection_valid";persist();return json.loads(wp.read_bytes()),row
   records=[json.loads(x)for x in trace.read_text().splitlines()];leases=[x for x in records if x["event"]=="lease_acquired"and x.get("request_header_id")==header];assert len(leases)==1
   lease=leases[0];assert owner is None or lease["replica"]==owner;assert lease["body_sha256"]==ref(bp)["sha256"]
   wires=[x for x in records if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];rel=[x for x in records if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]]
   assert len(wires)==len(rel)==1and rel[0]["released"]and not rel[0]["backend_failure"]and wires[0]["audit_error"]is None
   assert Path(wires[0]["wire_path"]).read_bytes()==wp.read_bytes()and wires[0]["wire_sha256"]==row["wire"]["sha256"]and wires[0]["wire_bytes"]==row["wire"]["bytes"]
   row.update(native_owner=lease["replica"],lease_id=lease["lease_id"],affinity_applied=lease["affinity_applied"],status="native_http_valid")
   value=sse(wp.read_bytes(),1 if method=="GET"and"starting_after=0"in path else 0)[0]if"text/event-stream"in res.headers.get("content-type","")and path.startswith("/v1/responses")else {}if"text/event-stream"in res.headers.get("content-type","")else json.loads(wp.read_bytes());persist();return value,row
  before=await metrics("before")
  try:
   if mode=="features":
    await idle("initial")
    for name,rf,stream in [("chat_schema",{"type":"json_schema","json_schema":{"name":"city","strict":True,"schema":schema}},False),("chat_json_object",{"type":"json_object"},True)]:
     body=chat_body(messages=[{"role":"user","content":"Return only a JSON object with city equal to Shanghai."}],response_format=rf,chat_template_kwargs={"enable_thinking":False},max_tokens=128,stream=stream)
     if stream:body["stream_options"]={"include_usage":True}
     v,row=await req(name,"POST","/v1/chat/completions",body,"D1");c=credit_chat(name,v,Path(row["wire"]["path"]).read_bytes(),stream,row);assert json.loads(c["content"])=={"city":"Shanghai"}
    for budget in [0,8]:
     name="thinking_"+str(budget);v,row=await req(name,"POST","/v1/chat/completions",chat_body(thinking_token_budget=budget),"D1");c=credit_chat(name,v,Path(row["wire"]["path"]).read_bytes(),False,row)
     assert 154842 in c["token_ids"]and c["token_ids"].index(154842)<=budget and "4"in c["content"],c
     row["thinking_end_token_index"]=c["token_ids"].index(154842);row["requested_budget"]=budget;persist()
    for name,budget in [("thinking_null",None),("thinking_unlimited",-1)]:
     v,row=await req(name,"POST","/v1/chat/completions",chat_body(thinking_token_budget=budget,chat_template_kwargs={"enable_thinking":False},max_tokens=32),"D0");credit_chat(name,v,Path(row["wire"]["path"]).read_bytes(),False,row)
    await req("thinking_invalid_bool","POST","/v1/chat/completions",chat_body(thinking_token_budget=True),"D1",400)
    base,br=await req("response_plain","POST","/v1/responses",response_body(request_id="resp_glm_run241_base",cache_salt=r.name+"-base"),"D1");credit_response(base,br,32);atomic_json(r/"base_response.json",base)
    v,row=await req("base_get","GET","/v1/responses/"+base["id"],owner="D1");assert v==base and row["affinity_applied"]
    body=response_body(previous_response_id=base["id"],request_id="resp_glm_run241_schema",cache_salt=r.name+"-schema");body.update(input="Return only a JSON object with city equal to Shanghai.",max_output_tokens=128,ignore_eos=False,text={"format":{"type":"json_schema","name":"city","strict":True,"schema":schema}})
    v,row=await req("response_schema_chain","POST","/v1/responses",body,"D1");credit_response(v,row);assert v["previous_response_id"]==base["id"]and row["affinity_applied"]and json.loads(text_response(v))=={"city":"Shanghai"}
    v,row=await req("response_sse","POST","/v1/responses",response_body(stream=True,request_id="resp_glm_run241_sse",cache_salt=r.name+"-sse"),"D1");credit_response(v,row,32)
    got,gr=await req("response_sse_get","GET","/v1/responses/"+v["id"],owner="D1");assert got==v and gr["affinity_applied"]
    await req("response_invalid","POST","/v1/responses",response_body(max_output_tokens=-1),"D1",400)
    await req("unknown_response","GET","/v1/responses/resp_unknown_run241",code=404)
    await req("retired_230","GET","/v1/responses/resp_glm_run230_D0_new",code=503,dispatch=False)
    v,row=await req("old_STORE211","GET","/v1/responses/resp_glm_run211_D1_new",owner="D1");assert v==json.loads((r.parent/"GLM-RUN-0239/resp_glm_run211_D1_new.wire").read_text())
   elif mode=="bgstart":
    await idle("initial");body=response_body(background=True,previous_response_id="resp_glm_run241_base",request_id="resp_glm_run241_cancel",max_output_tokens=8192)
    v,row=await req("create","POST","/v1/responses",body,"D1");assert v["status"]in["queued","in_progress"]and v["usage"]is None and row["affinity_applied"]
    native_background_owned[v["id"]]=NATIVES[row["native_owner"]];atomic_json(r/"pending_response.json",v)
    for _ in range(100):
     m=await metrics("live");snap=(await client.get(BASE+"/control/replicas")).json()
     if m["D1"].get("vllm:num_requests_running",0)>0and all(not x["active_requests"]for x in snap["replicas"]):break
     await asyncio.sleep(.1)
    else:raise RuntimeError("native background request not active after zero HTTP leases")
    native_background_owned.clear()
   elif mode=="bgfinish":
    pending=json.loads((r/"pending_response.json").read_text());native_background_owned[pending["id"]]=NATIVES["D1"]
    v,row=await req("retrieve_after_restart","GET","/v1/responses/"+pending["id"],owner="D1");assert v["id"]==pending["id"]and v["status"]in["queued","in_progress"]and row["affinity_applied"]
    v,row=await req("cancel_after_restart","POST","/v1/responses/"+pending["id"]+"/cancel",owner="D1");assert v["status"]=="cancelled"and row["affinity_applied"];native_background_owned.clear();await idle("after_cancel")
    v,row=await req("base_after_restart","GET","/v1/responses/resp_glm_run241_base",owner="D1");assert v==json.loads((r/"base_response.json").read_text())and row["affinity_applied"]
    rep={"id":"D1","url":NATIVES["D1"]};z=await client.delete(BASE+"/control/replicas/D1");z.raise_for_status()
    try:
     await req("drained_owner","GET","/v1/responses/resp_glm_run241_base",code=503,dispatch=False)
     await req("no_compatible_new","POST","/v1/responses",response_body(request_id="resp_glm_run241_noeligible"),code=503,dispatch=False)
    finally:
     z=await client.post(BASE+"/control/replicas",json=rep);z.raise_for_status()
    v,row=await req("base_readd","GET","/v1/responses/resp_glm_run241_base",owner="D1");assert v==json.loads((r/"base_response.json").read_text())and row["affinity_applied"]
    v,row=await req("background_create","POST","/v1/responses",response_body(background=True,request_id="resp_glm_run241_bg32"),"D1");assert v["status"]in["queued","in_progress"]and v["usage"]is None
    native_background_owned[v["id"]]=NATIVES["D1"]
    for i in range(120):
     got,gr=await req("poll_"+str(i),"GET","/v1/responses/"+v["id"],owner="D1")
     if got["status"]in["completed","incomplete"]:credit_response(got,gr,32);native_background_owned.clear();break
     await asyncio.sleep(.2)
    else:raise RuntimeError("native background did not complete")
    v,row=await req("background_sse","POST","/v1/responses",response_body(background=True,stream=True,request_id="resp_glm_run241_bg_sse"),"D1");credit_response(v,row,32)
    replay,rr=await req("background_sse_replay","GET","/v1/responses/"+v["id"]+"?stream=true&starting_after=0",owner="D1");assert replay==v and rr["affinity_applied"]
   else:raise ValueError(mode)
   after=await metrics("after")if mode=="bgstart"else await idle("final")
   delta={k:{n:after[k][n]-before[k][n]for n in before[k]if n.endswith("_total")}for k in NATIVES}
   if mode=="features":
    for k in NATIVES:
     c=[v for v in credits.values()if v["owner"]==k];assert delta[k]["vllm:generation_tokens_total"]==sum(v["outputs"]for v in c)and delta[k]["vllm:prompt_tokens_total"]==sum(v["prompts"]for v in c)and delta[k]["vllm:request_success_total"]==len(c)and delta[k]["vllm:num_preemptions_total"]==0
   snap=(await client.get(BASE+"/control/replicas")).json();assert len(snap["replicas"])==2and all(not x["active_requests"]and not x["draining"]and not x["group_faulted"]for x in snap["replicas"])
   atomic_json(r/(mode+"_summary.json"),dict(valid=True,mode=mode,rows=rows,credits=credits,completed=len(credits),effective_outputs=sum(v["outputs"]for v in credits.values()),native_delta=delta,placement=snap,cancel_output_credit=0 if mode=="bgfinish"else None,limits=["Real finite feature/lifecycle contract evidence; no performance KEEP or stable capacity","Cancelled generation increases native counters with zero credited output; polling is not new inference"]))
   print(json.dumps(dict(valid=True,mode=mode,completed=len(credits),outputs=sum(v["outputs"]for v in credits.values()))))
  except BaseException as error:
   atomic_json(r/(mode+"_failure.json"),dict(error_type=type(error).__name__,error=str(error),rows=rows,credits=credits));raise
  finally:
   for ident,url in native_background_owned.items():
    try:
     z=await client.post(url+"/v1/responses/"+ident+"/cancel");(r/(mode+"_cleanup_"+ident+".wire")).write_bytes(z.content)
    except BaseException as error:atomic_json(r/(mode+"_cleanup_unknown.json"),dict(error=str(error)))
asyncio.run(main())
