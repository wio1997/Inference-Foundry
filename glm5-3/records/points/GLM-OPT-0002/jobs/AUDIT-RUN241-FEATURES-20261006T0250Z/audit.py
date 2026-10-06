from pathlib import Path
import sys,json,hashlib,urllib.request,re
j=Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=Path(job["inputs"][0]["path"]);sys.path.insert(0,str(r/"runtime_bundle"))
from phase_runner import same_process,utc,atomic_json
from native_identity_observer import observe
from native_engines_service_config import checked_config
def ref(f):
 raw=Path(f).read_bytes();return dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
st=json.loads((r/"state.json").read_text());assert st["status"]=="completed" and st["completed_stages"]==["prepare","features","bgstart","restart","bgfinish","finish"] and not same_process(st["owner"])
spec=json.loads((r/"controller_spec.json").read_text());assert hashlib.sha256((r/"controller_spec.json").read_bytes()).hexdigest()==st["spec_sha256"]
for x in spec["stages"][0]["sources"]:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes() and hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]
summaries={k:json.loads((r/(k+"_summary.json")).read_text())for k in ["features","bgstart","bgfinish"]}
trace=[json.loads(l)for l in(r.parent/"GLM-RUN-0125/router_trace.jsonl").read_text().splitlines()]
credits={}
for mode,summary in summaries.items():
 assert summary["valid"]
 for name,value in summary["credits"].items():assert name not in credits;credits[name]=value
 for row in summary["rows"]:
  assert ref(row["body"]["path"])==row["body"] and ref(row["wire"]["path"])==row["wire"]
  ls=[x for x in trace if x["event"]=="lease_acquired" and x.get("request_header_id")==row["header_id"]]
  if row["status"]=="prelease_rejection_valid":assert not ls and row["http_status"]==503;continue
  assert len(ls)==1 and ls[0]["replica"]==row["native_owner"] and ls[0]["lease_id"]==row["lease_id"] and ls[0]["body_sha256"]==row["body"]["sha256"]
  wires=[x for x in trace if x["event"]=="upstream_stream_contract" and x.get("lease_id")==row["lease_id"]];rel=[x for x in trace if x["event"]=="lease_released" and x.get("lease_id")==row["lease_id"]]
  assert len(wires)==len(rel)==1 and rel[0]["released"] and not rel[0]["backend_failure"] and wires[0]["audit_error"]is None
  assert Path(wires[0]["wire_path"]).read_bytes()==Path(row["wire"]["path"]).read_bytes() and wires[0]["wire_sha256"]==row["wire"]["sha256"]
  if "chat"in row:assert len(row["chat"]["token_ids"])==row["chat"]["usage"]["completion_tokens"]
 for k,v in summary["native_delta"].items():assert v["vllm:num_preemptions_total"]==0
feature_rows={x["name"]:x for x in summaries["features"]["rows"]}
for name in ["chat_schema","chat_json_object"]:assert feature_rows[name]["native_owner"]=="D1" and json.loads(feature_rows[name]["chat"]["content"])=={"city":"Shanghai"}
for budget in [0,8]:
 row=feature_rows["thinking_"+str(budget)];assert row["native_owner"]=="D1" and row["thinking_end_token_index"]==budget and row["chat"]["token_ids"][budget]==154842 and "4"in row["chat"]["content"]
for name in ["thinking_null","thinking_unlimited"]:assert feature_rows[name]["native_owner"]=="D0"
for name in ["thinking_invalid_bool","response_invalid"]:assert feature_rows[name]["http_status"]==400
assert feature_rows["unknown_response"]["http_status"]==404 and feature_rows["retired_230"]["http_status"]==503
for name in ["response_plain","base_get","response_schema_chain","response_sse","response_sse_get","old_STORE211"]:assert feature_rows[name]["native_owner"]=="D1"
schema_response=json.loads(Path(feature_rows["response_schema_chain"]["wire"]["path"]).read_text());text="".join(c.get("text","")for item in schema_response["output"]if item["type"]=="message"for c in item["content"]if c["type"]=="output_text")
assert json.loads(text)=={"city":"Shanghai"} and schema_response["previous_response_id"]=="resp_glm_run241_base" and feature_rows["response_schema_chain"]["affinity_applied"]
brows={x["name"]:x for x in summaries["bgfinish"]["rows"]}
assert summaries["bgfinish"]["cancel_output_credit"]==0 and not any("cancel"in name for name in credits)
for name in ["retrieve_after_restart","cancel_after_restart","base_after_restart","base_readd","background_sse_replay"]:assert brows[name]["native_owner"]=="D1" and brows[name]["affinity_applied"]
assert json.loads(Path(brows["cancel_after_restart"]["wire"]["path"]).read_text())["status"]=="cancelled"
assert brows["drained_owner"]["status"]==brows["no_compatible_new"]["status"]=="prelease_rejection_valid"
restart=json.loads((r/"restart_summary.json").read_text());assert restart["native_background_live_before"] and restart["HTTP_leases0"] and restart["native_model_signals"]==0
for mode in ["features","bgstart","bgfinish"]:
 exe=json.loads((r/(mode+"_execution.json")).read_text());assert exe["status"]=="succeeded" and not exe["native_client_alive"] and not exe["signal_attempts"]
 acks=[json.loads(l)for l in(r/(mode+".stdout")).read_text().splitlines()if l.startswith('{"event":')];assert [a["event"]for a in acks]==["task_acl_init","task_acl_finalize"] and all(a["returncode"]==0for a in acks)
summary=json.loads((r/"feature_summary.json").read_text());assert summary["functional_acceptance"] and len(credits)==summary["completed"]==11 and sum(v["outputs"]for v in credits.values())==summary["effective_output_tokens"]==179
def metrics(path):
 out={}
 for l in Path(path).read_text().splitlines():
  if l.startswith("vllm:"):
   k=l.split("{")[0].split()[0];out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
delta={};cancel={}
for key,node in [("D0","166"),("D1","167")]:
 a=metrics(r/("before_"+node+".metrics"));b=metrics(r/("terminal_"+node+".metrics"));d={k:b[k]-a[k]for k in a if k.endswith("_total")};delta[key]=d
 c=[v for v in credits.values()if v["owner"]==key];output=sum(v["outputs"]for v in c);prompts=sum(v["prompts"]for v in c)
 assert d["vllm:request_success_total"]==len(c) and d["vllm:num_preemptions_total"]==0
 cancel[key]=dict(uncredited_generation=d["vllm:generation_tokens_total"]-output,uncredited_prompt=d["vllm:prompt_tokens_total"]-prompts)
 assert cancel[key]["uncredited_generation"]>=0 and cancel[key]["uncredited_prompt"]>=0
 if key=="D0":assert cancel[key]["uncredited_generation"]==cancel[key]["uncredited_prompt"]==0
 else:assert cancel[key]["uncredited_generation"]>0 and cancel[key]["uncredited_prompt"]==60
conf,_=checked_config(r/"restored/service_config.json");proof=json.loads((r/"restored/public_service_proof.json").read_text());assert same_process(proof["host"]) and proof["marker"]["gateway_version"]=="V14" and ref(proof["config"]["path"])==proof["config"]
fresh=observe(r/"restored/service_config.json");assert all(g["status"]=="healthy"for g in fresh["groups"]) and sum(h["expected_native_workers"]for g in fresh["groups"]for h in g["hosts"])==32
acks=[json.loads(l)for l in(r/"restored/public.gateway.log").read_text().splitlines()if l.startswith('{"event":')];assert len(acks)==1 and acks[0]["event"]=="task_acl_init" and acks[0]["returncode"]==0
old=r.parent/"GLM-RUN-0240";acks=[json.loads(l)for l in(old/"restored/public.gateway.log").read_text().splitlines()if l.startswith('{"event":')];assert [a["event"]for a in acks]==["task_acl_init","task_acl_finalize"] and all(a["returncode"]==0for a in acks)
with urllib.request.urlopen("http://127.0.0.1:8000/control/replicas",timeout=8)as response:placement=json.loads(response.read())
assert len(placement["replicas"])==2 and all(not x["group_faulted"] and not x["active_requests"] and not x["draining"]for x in placement["replicas"])
for id,baseline in [("resp_glm_run241_base",r/"base_response.json"),("resp_glm_run211_D1_new",r.parent/"GLM-RUN-0239/resp_glm_run211_D1_new.wire")]:
 with urllib.request.urlopen("http://127.0.0.1:8000/v1/responses/"+id,timeout=8)as response:raw=response.read()
 assert json.loads(raw)==json.loads(baseline.read_text())
out=dict(at=utc(),functional_acceptance=True,run_id=r.name,completed=11,effective_output_tokens=179,cancelled_requests=1,cancelled_output_credit=0,uncredited_cancel_native_counters=cancel,native_delta=delta,public_proof=proof,native_fresh=fresh,source_count=len(spec["stages"][0]["sources"]),checks=["Frozen sources/native32/publicV14 unchanged epochs","Actual JSONschema/jsonobject semantics, budget0/8 native closure at positions0/8, null/unlimited V2/native400","Response V1 new+previous schema native STORE/typed foreground-background SSE/replay/400404/oldretired503","Native background8192 survived exactpublic cleanSDK0 restart with zeroHTTPleases; sameowner retrieved/cancelled, no output credit","Logical V1 drain excludes newV1Response/owner before RPC; sameepoch readd retrieves state","SDK three clients init/final0/native counter conservation including cancellation cost"],limits=summary["limits"],Current=None)
f=j/"reduction.json";atomic_json(f,out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run241 real native features and public-restart background lifecycle VALID: 11/179, one cancellation zero credit, native32 unchanged",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(f),locator="readonly frozen-source and native functional lifecycle audit")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,completed=11,outputs=179,cancel_counters=cancel)))
