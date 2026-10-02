from pathlib import Path
import json,hashlib,sys,typing,importlib,math
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json
from sse_observer import NativeSSEObserver
from pydantic import BaseModel
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0046";rows=json.loads((r/"attempts.json").read_text());summary=json.loads((r/"responses_summary.json").read_text());events=json.loads((r/"state_events.json").read_text())
trace=[json.loads(x)for x in (r/"router_trace.jsonl").read_text().splitlines()];leases=[x for x in trace if x["event"]=="lease_acquired"];assert len(leases)==len(rows)==22
classes={}
for name in ["openai.types.responses","vllm.entrypoints.openai.responses.streaming_events","vllm.entrypoints.openai.responses.protocol"]:
 for cls in vars(importlib.import_module(name)).values():
  if isinstance(cls,type)and issubclass(cls,BaseModel)and "type"in cls.model_fields:
   for tag in typing.get_args(cls.model_fields["type"].annotation):
    if isinstance(tag,str):classes[tag]=cls
def checkedref(ref):
 p=Path(ref["path"]);raw=p.read_bytes();assert len(raw)==ref["bytes"]and hashlib.sha256(raw).hexdigest()==ref["sha256"];return raw
commits={};attempts=[];cancels=[]
for row in rows:
 assert row["outcome"]=="native_http_valid";body=checkedref(row["body"]);raw=checkedref(row["wire"]);native=checkedref(row["native_wire"]);assert raw==native
 acquire=[x for x in leases if x.get("lease_id")==row["lease_id"]];assert len(acquire)==1;a=acquire[0];assert a["body_sha256"]==hashlib.sha256(body).hexdigest()and a["request_header_id"]==r.name+"-"+row["id"]and a["replica"]=="TP32"and a["affinity_applied"]==row["affinity_applied"]
 seq=[x for x in trace if x.get("lease_id")==row["lease_id"]];headers=[x for x in seq if x["event"]=="upstream_headers"];assert len(headers)==1 and headers[0]["status"]==row["http_status"]
 rel=[x for x in seq if x["event"]=="lease_released"];assert len(rel)==1 and rel[0]["released"]and not rel[0]["backend_failure"]
 wire=[x for x in seq if x["event"]=="upstream_stream_contract"];assert len(wire)==1 and wire[0]["audit_error"]is None and wire[0]["wire_sha256"]==row["native_wire"]["sha256"]
 if row["http_status"]==404:
  assert row["id"]=="unknown_native404"and json.loads(raw)["error"]["type"]=="invalid_request_error";attempts.append({"id":row["id"],"output_credit":0,"status":404});continue
 obj=None;sse_events=[]
 if raw.startswith(b"event:"):
  for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
   lines=frame.decode().splitlines();data="\n".join(x[5:].lstrip(" ")for x in lines if x.startswith("data:"));tag="\n".join(x[6:].strip()for x in lines if x.startswith("event:"))
   if not data:continue
   assert data!="[DONE]";ev=json.loads(data);assert tag==ev["type"]and tag in classes and tag!="error";classes[tag].model_validate(ev);sse_events.append(ev)
  start=1 if row["method"]=="GET"and"starting_after=0"in row["path"]else 0
  assert [x["sequence_number"]for x in sse_events]==list(range(start,start+len(sse_events)))
  assert sse_events[-1]["type"]=="response.completed"and sum(x["type"]=="response.completed"for x in sse_events)==1
  obj=sse_events[-1]["response"]
 else:obj=json.loads(raw)
 if row["path"].startswith("/v1/chat/completions"):
  u=obj["usage"];assert u["completion_tokens"]==32 and u["total_tokens"]==u["prompt_tokens"]+32 and obj["choices"][0]["finish_reason"]=="length"
  for lp in obj["choices"][0]["logprobs"]["content"]:assert math.isfinite(lp["logprob"])and len(lp["top_logprobs"])>=3
  commits["chat"]={"output_tokens":32,"source":row["id"],"native_id":obj["id"]};attempts.append({"id":row["id"],"native_usage":u,"new_inference":True});continue
 native_obj=ResponsesResponse.model_validate(obj);payload=json.loads(body)if body else None
 if native_obj.status in ["completed","incomplete"]:
  assert native_obj.usage is not None;u=native_obj.usage.model_dump();assert u["output_tokens"]==32 and u["input_tokens"]>0 and u["total_tokens"]==u["input_tokens"]+32
  value={"output_tokens":32,"usage":u,"native_id":native_obj.id}
  if native_obj.id in commits:assert commits[native_obj.id]["usage"]==u
  else:
   assert row["method"]=="POST"and row["path"].split("?")[0]=="/v1/responses"or row["id"].startswith("background_poll")
   commits[native_obj.id]=value
 elif native_obj.status=="cancelled":
  assert row["id"]=="cancel_after_restart"and row["affinity_applied"];cancels.append(native_obj.id)
 else:assert native_obj.status in ["queued","in_progress"]and native_obj.usage is None
 if payload and payload.get("previous_response_id"):assert row["affinity_applied"]and obj["previous_response_id"]==payload["previous_response_id"]
 if row["method"]=="GET"and row["http_status"]==200:assert row["affinity_applied"]
 attempts.append({"id":row["id"],"native_id":native_obj.id,"native_status":native_obj.status,"native_usage":None if native_obj.usage is None else native_obj.usage.model_dump(),"SSE_events":len(sse_events),"native_wire":row["native_wire"],"body":row["body"],"lease_id":row["lease_id"]})
assert len(commits)==6 and sum(x["output_tokens"]for x in commits.values())==192 and len(cancels)==1
creates=[x for x in rows if x["method"]=="POST"and x["path"].split("?")[0]in ["/v1/responses","/v1/chat/completions"]];assert len(creates)==7
live=[x for x in events if x["event"]=="native_job_outlives_zero_HTTP_leases"];assert len(live)==1 and live[0]["native"]["num_requests_running"]>0 and all(x["active_requests"]==0 for x in live[0]["placement"]["replicas"])
clean=[x for x in events if x["event"]=="gateway_clean_stopped"];assert len(clean)==2 and all(x["exit_code"]in [0,-15]for x in clean)
groups=json.loads((r/"execution_groups.json").read_text());owners=json.loads((r/"response_owners.json").read_text());assert (r/"response_owners.json").stat().st_mode&0o777==0o600
for key,v in owners["owners"].items():assert v=={"replica":"TP32","url":"http://172.16.10.166:9081","group":groups[0]["id"],"epoch":groups[0]["epoch"]}
journal=json.loads((r/"response_owners.json.fault").read_text());assert not journal["open"]and all(not x["faulted"]for x in journal["groups"].values())
# Independent raw tool validation using exact frozen validator.
tool_summary=json.loads((r/"tools_TP32/summary.json").read_text());assert tool_summary["valid"]and len(tool_summary["requests"])==4
source=(r/"tools.py").read_text();a=source.index("  calls={};");validator=source[a:source.index("  attempt.update(",a)]
tools=[]
for row in tool_summary["requests"]:
 raw=checkedref(row["wire"]);ns={"raw":raw,"stream":row["stream"],"name":row["name"],"json":json,"NativeSSEObserver":NativeSSEObserver};exec(compile("if True:\n"+validator,"frozen native toolvalidator","exec"),ns)
 assert ns["usage"]==row["usage"]and ns["finish"]==row["finish_reasons"]and list(ns["calls"].values())==row["tools"]
 tools.append({"name":row["name"],"usage":row["usage"],"wire":row["wire"]})
outputs=192+sum(x["usage"]["completion_tokens"]for x in tools)
out={"run_id":r.name,"measurement_valid":True,"functional_acceptance":True,"new_inference_attempts":11,"new_completed_inference_requests":10,"new_effective_output_tokens":outputs,"Responses_state":{"new_inference_attempts":7,"completed":6,"outputs":192,"cancelled_credit":0,"http_rows":22,"native_unique_commit_ledger":commits,"attempts":attempts,"background_native_vs_HTTP_lease":live[0],"clean_proxy_restarts":clean,"owner_journal":{"path":str(r/"response_owners.json"),"sha256":hashlib.sha256((r/"response_owners.json").read_bytes()).hexdigest(),"entries":len(owners["owners"])}},"native_tools":tools,"configuration":"Native reduce_samplingFalse/STORE1, DP1TP32DCP1EP32/K5/FULL/16384/.80. Actual ordinarynative randomtopk/topp/logprobs, not distributedreduce True path.","limits":summary["limits"]+["Raw state replay/GETs deduped0newcommit, actualcancelcommittednativecounters not outputcredit","Native background survives cleanproxystop, not nativeAPIrestart/fault proof"]}
atomic_json(j/"protocol_reduction.json",out);print(json.dumps({"commits":10,"outputs":outputs,"state_outputs":192,"cancel_credit":0,"http":22}))
