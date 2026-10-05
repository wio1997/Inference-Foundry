from pathlib import Path
import json,hashlib,sys
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0229"
sys.path.insert(0,str(r/"runtime_bundle"))
from phase_runner import utc,atomic_json,same_process
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def metric(f):
 out={}
 for l in Path(f).read_text().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{")[0].split()[0]
  if k.startswith("vllm:"):out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
state=json.loads((r/"state.json").read_text());assert state["status"]=="failed"and state["failure_phase"]=="matrix"and state["completed_stages"]==["prepare","adopt"]and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text());m=json.loads((r/"manifest.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==195and ref(r/"controller_spec.json")["sha256"]==state["spec_sha256"]==m["spec"]["sha256"]
for x in pins:assert ref(x["path"])["sha256"]==x["sha256"]and Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()
a=json.loads((r/"cap3N1/arrival_summary.json").read_text());assert not a["functional_acceptance"]and a["actual_new_requests"]==1and a["effective_public_output_tokens"]==0
row=a["requests"][0];assert not row["completed"]and row["expected_prompt_tokens"]==21and row["effective_public_output_credit"]==0and row["http_status"]==200and row["error_type"]=="AssertionError"
assert ref(row["wire"]["path"])==row["wire"]
body=r/"cap3N1/cap3N1_0.body.json";assert ref(body)["sha256"]==row["body_sha256"];b=json.loads(body.read_text());assert b["chat_template_kwargs"]=={"enable_thinking":False}
frames=[];done=False
for frame in Path(row["wire"]["path"]).read_bytes().replace(b"\r\n",b"\n").split(b"\n\n"):
 data=b"\n".join(l[5:].lstrip(b" ")for l in frame.splitlines()if l.startswith(b"data:"))
 if data==b"[DONE]":done=True
 elif data:frames.append(json.loads(data))
ids=[t for f in frames for c in f.get("choices",[])for t in c.get("token_ids",[])];usage=[f["usage"]for f in frames if f.get("usage")];finish=[c["finish_reason"]for f in frames for c in f.get("choices",[])if c.get("finish_reason")]
assert done and len(ids)==256and usage[-1]==dict(prompt_tokens=15,completion_tokens=256,total_tokens=271)and finish==["length"]
trace=[json.loads(l)for l in(p/"runs/GLM-RUN-0125/router_trace.jsonl").read_text().splitlines()]
leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==r.name+"-cap3N1_0"];assert len(leases)==1and leases[0]["replica"]=="D0"and leases[0]["body_sha256"]==row["body_sha256"]
wire=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==leases[0]["lease_id"]];release=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==leases[0]["lease_id"]]
assert len(wire)==len(release)==1and wire[0]["audit_error"]is None and release[0]["released"]and not release[0]["backend_failure"]and Path(wire[0]["wire_path"]).read_bytes()==Path(row["wire"]["path"]).read_bytes()
delta={}
for node in["D0","D1"]:
 before=metric(r/("matrix_before_"+node+".metrics"));after=metric(r/("failure_native_drain_"+node+".metrics"));delta[node]={k:after[k]-v for k,v in before.items()if k.endswith("_total")}
 assert after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==after["vllm:kv_cache_usage_perc"]==0
assert delta["D0"]["vllm:generation_tokens_total"]==256and delta["D0"]["vllm:prompt_tokens_total"]==15and delta["D0"]["vllm:request_success_total"]==1and delta["D0"]["vllm:num_preemptions_total"]==0and all(v==0for v in delta["D1"].values())
ev=[json.loads(l)for l in(r/"matrix_client.stdout").read_text().splitlines()if l.startswith('{"event":')];assert ev[0]["event"]=="task_acl_init"and ev[-1]["event"]=="task_acl_finalize"and ev[0]["returncode"]==ev[-1]["returncode"]==0
assert json.loads((r/"matrix_client.receipt.json").read_text())["exit_code"]==1
assert not(r/"functional_summary.json").exists()and json.loads((r/"matrix_failure.json").read_text())["partial"]==[]
src=(r/"matrix_client.py").read_text();assert 'count"]==21'in src
fixed=json.loads((p/"jobs/AUDIT-RUN230-QUEUE-FUNCTION-20261005T1545Z-v2/reduction.json").read_text());assert fixed["functional_acceptance"]and fixed["effective_output_tokens"]==3643
limits=["OriginalRun229 workflowFAILED/measurementINVALID, actualnative256IDs and15prompt completed; originalcredit0 preserved, no TTFT/POT reconstruction/performance credit","GPT fixture tokenization omitted generation chat_template_kwargs.enable_thinkingfalse expected21; repairednewRun230 actualsamekwargs15/14matrix requests valid","Readonly audit only, no native/model/public/policy/inference replay; native engine nofault establishedbyactualwire/drain counters and230sameepoch","FullnativeAPI/thinkingbudget/SLOcapacity unknown, noKEEP/CurrentNone"]
out=dict(at=utc(),run_id=r.name,source_count=len(pins),workflow_verdict="INVALID",measurement_valid=False,functional_native_request_valid=True,actual_native_output_tokens=256,actual_native_new_completed=1,original_effective_credit=0,actual_prompt_tokens=15,wrong_expected_prompt_tokens=21,wire=ref(row["wire"]["path"]),body=ref(body),token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),native_delta=delta,SDKinit_finalize0=True,original_script_exit=1,replay=False,verdict="INVALID",Current=None,limits=limits)
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run229 originalFAILED fixture expected21vsactual15, native256IDs-wire-counters VALID but originalmeasurementINVALID/credit0; noGPUreplay",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="originalfailed-source195-256IDs-wire-15prompt-nativecounterdelta-SDK0")],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(dict(valid=True,original_measurement="INVALID",actual_outputs=256)))
