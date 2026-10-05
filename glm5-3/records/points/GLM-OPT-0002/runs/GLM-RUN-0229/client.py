from pathlib import Path
import json,sys,hashlib,time,re,urllib.request,urllib.error,importlib,typing
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
from pydantic import BaseModel
from sse_observer import NativeSSEObserver
start_watchdog();r=Path(__file__).parent;mode=sys.argv[1];state=r.parent/"GLM-RUN-0125";trace=state/"router_trace.jsonl";owners=state/"response_owners.json";rows=[];credit=0
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));base="http://127.0.0.1:8000"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def ctl(path,method="GET",body=None):
 guard();req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers={"content-type":"application/json"},method=method)
 with opener.open(req,timeout=15)as res:return json.loads(res.read())
def sample(label):
 groups=["D0","D1"]if mode!="survivor"else["D1"];values={}
 for key in groups:
  url="http://172.16.10.166:9081"if key=="D0"else"http://172.16.10.167:9900"
  with opener.open(url+"/metrics",timeout=10)as res:b=res.read()
  (r/("client_"+mode+"_"+label+"_"+key+".metrics")).write_bytes(b);v={}
  for metric in ["num_requests_running","num_requests_waiting","generation_tokens_total","prompt_tokens_total","num_preemptions_total"]:
   m=re.findall(r"^vllm:"+metric+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M);assert m;v[metric]=sum(float(x)for x in m)
  assert v["num_requests_running"]==v["num_requests_waiting"]==0;values[key]=v
 return values
def req(name,method,path,payload=None,status=200,owner=None):
 label="client_"+mode+"_"+name+"_"+str(len(rows));body=b""if payload is None else json.dumps(payload,ensure_ascii=False,separators=(",",":")).encode();bf=r/(label+".body");bf.write_bytes(body)
 prior=trace.read_bytes();request=urllib.request.Request(base+path,data=body if method=="POST"else None,headers={"content-type":"application/json","accept-encoding":"identity","x-request-id":r.name+"-"+label},method=method)
 start=time.monotonic()
 try:
  with opener.open(request,timeout=180)as response:raw=response.read();actual=response.status;ctype=response.headers.get("content-type","")
 except urllib.error.HTTPError as e:raw=e.read();actual=e.code;ctype=e.headers.get("content-type","")
 wf=r/(label+".wire");wf.write_bytes(raw);row=dict(name=name,mode=mode,method=method,path=path,body=ref(bf),wire=ref(wf),status=actual,wall_s=time.monotonic()-start);rows.append(row);atomic_json(r/("client_"+mode+"_attempts.json"),rows);assert actual==status
 if status==503:
  assert trace.read_bytes()==prior;row["rejected_before_lease_and_RPC"]=True
 else:
  for _ in range(150):
   snap=ctl("/control/replicas")
   if all(not x["active_requests"]for x in snap["replicas"]):break
   time.sleep(.02)
  else:raise RuntimeError("native HTTP lease not released")
  events=[json.loads(l)for l in trace.read_text().splitlines()];leases=[x for x in events if x["event"]=="lease_acquired"and x.get("request_header_id")==r.name+"-"+label];assert len(leases)==1
  lease=leases[0];contracts=[x for x in events if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];releases=[x for x in events if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]];assert len(contracts)==len(releases)==1
  c=contracts[0];assert c["audit_error"]is None and c["wire_bytes"]==len(raw)and c["wire_sha256"]==row["wire"]["sha256"]and Path(c["wire_path"]).read_bytes()==raw
  assert releases[0]["released"]and not releases[0]["backend_failure"]and lease["body_sha256"]==row["body"]["sha256"]
  if owner is not None:assert lease["replica"]==owner
  row.update(native_owner=lease["replica"],affinity_applied=lease["affinity_applied"],lease_id=lease["lease_id"],native_wire=dict(path=c["wire_path"],bytes=c["wire_bytes"],sha256=c["wire_sha256"]))
 atomic_json(r/("client_"+mode+"_attempts.json"),rows)
 if "text/event-stream"in ctype and "/v1/responses"in path:
  classes={}
  for mod in ["openai.types.responses","vllm.entrypoints.openai.responses.streaming_events","vllm.entrypoints.openai.responses.protocol"]:
   for cls in vars(importlib.import_module(mod)).values():
    if isinstance(cls,type)and issubclass(cls,BaseModel)and "type"in cls.model_fields:
     for tag in typing.get_args(cls.model_fields["type"].annotation):
      if isinstance(tag,str):classes[tag]=cls
  ev=[]
  for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
   data="\n".join(x[5:].lstrip(" ")for x in frame.decode().splitlines()if x.startswith("data:"))
   if not data:continue
   val=json.loads(data);assert val["type"]in classes and val["type"]!="error";classes[val["type"]].model_validate(val);ev.append(val)
  assert [x["sequence_number"]for x in ev]==list(range(len(ev)))
  assert ev[-1]["type"]=="response.completed"and len([x for x in ev if x["type"]=="response.completed"])==1
  atomic_json(r/(label+".events.json"),ev);return ev[-1]["response"],row
 return json.loads(raw),row
def retained_D0_before():
 f=r.parent/"GLM-RUN-0214/client_final_summary.json";c=json.loads(f.read_text());stored_row=next(x for x in c["requests"]if x["name"]=="newD0_create")
 value,row=req("D0_retained","GET","/v1/responses/resp_glm_run214_D0_new",owner="D0")
 ResponsesResponse.model_validate(value);assert Path(row["wire"]["path"]).read_bytes()==Path(stored_row["wire"]["path"]).read_bytes()and row["affinity_applied"]
def unavailable():pass
def semantics(owner):
 global credit
 for name,prompt,answer in [("add2","Compute 2 + 2. Reply with only the number.","4"),("add17","Compute 17 + 25. Reply with only the number.","42"),("literal","Reply with exactly this text and nothing else: GLM_OK_731","GLM_OK_731")]:
  body=dict(model="glm-52",messages=[dict(role="user",content=prompt)],temperature=0,seed=20260930,max_tokens=32,ignore_eos=False,stream=False,return_token_ids=True,cache_salt=r.name+"-"+mode+"-"+name,chat_template_kwargs=dict(enable_thinking=False))
  v,row=req(name,"POST","/v1/chat/completions",body,owner=owner);u=v["usage"];c=v["choices"][0];assert c["message"]["content"].strip()==answer and c["finish_reason"]=="stop"
  assert u["total_tokens"]==u["prompt_tokens"]+u["completion_tokens"]and u["completion_tokens"]>0
  assert len(c["token_ids"])==u["completion_tokens"]
  credit+=u["completion_tokens"];row.update(usage=u,semantic_pass=True,effective_output_tokens=u["completion_tokens"]);atomic_json(r/("client_"+mode+"_attempts.json"),rows)
def resp_commit(v,row,n):
 global credit
 obj=ResponsesResponse.model_validate(v);u=v["usage"];assert obj.status in ["completed","incomplete"]and not v.get("error")and u["output_tokens"]==n and u["total_tokens"]==u["input_tokens"]+u["output_tokens"]
 assert obj.max_output_tokens==n;credit+=n;row.update(usage=u,effective_output_tokens=n);atomic_json(r/("client_"+mode+"_attempts.json"),rows)
def retained_D1():
 f=r.parent/"GLM-RUN-0211/client_final_summary.json";c=json.loads(f.read_text());stored_row=next(x for x in c["requests"]if x["name"]=="newD1_create")
 value,row=req("D1_retained","GET","/v1/responses/resp_glm_run211_D1_new",owner="D1")
 ResponsesResponse.model_validate(value);assert Path(row["wire"]["path"]).read_bytes()==Path(stored_row["wire"]["path"]).read_bytes()and row["affinity_applied"]
def retained_D0_210():
 c=json.loads((r.parent/"GLM-RUN-0210/client_final_summary.json").read_text());stored_row=next(x for x in c["requests"]if x["name"]=="newD0_create")
 value,row=req("D0_210_retained","GET","/v1/responses/resp_glm_run210_D0_new",owner="D0")
 ResponsesResponse.model_validate(value);assert Path(row["wire"]["path"]).read_bytes()==Path(stored_row["wire"]["path"]).read_bytes()and row["affinity_applied"]
def old_D0_unavailable():
 journal=owners.read_bytes()
 for number,id in [(214,"resp_glm_run214_D0_new"),(223,"resp_glm_run223_D0_new"),(226,"resp_glm_run226_D0_new")]:
  for name,method,path,payload in [("retired"+str(number)+"_get","GET","/v1/responses/"+id,None),("retired"+str(number)+"_previous","POST","/v1/responses",dict(model="glm-52",input="Continue.",previous_response_id=id,max_output_tokens=8,store=True))]:
   req(name,method,path,payload,status=503);assert owners.read_bytes()==journal
before=sample("before")
try:
 retained_D1();unavailable()
 if mode=="before":
  retained_D0_before()
 elif mode=="survivor":
  old_D0_unavailable();snap=ctl("/control/replicas");a={x["id"]:x for x in snap["replicas"]};assert a["D0"]["group_faulted"]and not a["D1"]["group_faulted"]
 elif mode=="final":
  old_D0_unavailable();ctl("/control/replicas/D1","DELETE")
  try:
   semantics("D0")
   payload=dict(model="glm-52",input="Explain a concrete GLM serving example. 上海",request_id="resp_glm_run229_D0_new",max_output_tokens=32,temperature=0,ignore_eos=True,seed=20260930,store=True,cache_salt=r.name+"-newD0")
   v,row=req("newD0_create","POST","/v1/responses",payload,owner="D0");resp_commit(v,row,32);assert v["id"]=="resp_glm_run229_D0_new"
  finally:
   config=json.loads((r/"restored/service_config.json").read_text());peers=config["environment"]["GLM_REPLICAS"];peers=json.loads(peers)if isinstance(peers,str)else peers;peer=next(x for x in peers if x["id"]=="D1");ctl("/control/replicas","POST",peer)
  got,rr=req("newD0_retrieve","GET","/v1/responses/"+v["id"],owner="D0");assert got==v and rr["affinity_applied"]
  payload=dict(model="glm-52",input="Continue briefly.",previous_response_id=v["id"],max_output_tokens=16,temperature=0,ignore_eos=True,seed=20260930,store=True,stream=True)
  child,cr=req("newD0_previous","POST","/v1/responses",payload,owner="D0");resp_commit(child,cr,16);assert child["previous_response_id"]==v["id"]and cr["affinity_applied"]
  got,rr=req("newD0_child_retrieve","GET","/v1/responses/"+child["id"],owner="D0");assert got==child and rr["affinity_applied"]
  retained_D1()
 else:raise ValueError("unknown client mode")
 after=sample("after");delta={key:{metric:after[key][metric]-before[key][metric]for metric in["generation_tokens_total","prompt_tokens_total","num_preemptions_total"]}for key in before}
 assert sum(x["generation_tokens_total"]for x in delta.values())==credit and all(x["num_preemptions_total"]==0for x in delta.values())
 assert delta["D1"]["generation_tokens_total"]==0
 out=dict(at=utc(),valid=True,mode=mode,requests=rows,effective_output_tokens=credit,native_delta=delta,zero_helper=True,no_replay=True,STORE_replication=False)
 atomic_json(r/("client_"+mode+"_summary.json"),out);print(json.dumps(dict(valid=True,mode=mode,output_tokens=credit,native_delta=delta)),flush=True)
except BaseException as e:
 atomic_json(r/("client_"+mode+"_summary.json"),dict(at=utc(),valid=False,mode=mode,error_type=type(e).__name__,error=str(e),requests=rows,effective_output_tokens=credit));raise
