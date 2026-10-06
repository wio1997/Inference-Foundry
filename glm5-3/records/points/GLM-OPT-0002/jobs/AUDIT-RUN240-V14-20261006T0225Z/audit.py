from pathlib import Path
import sys,json,hashlib,urllib.request
j=Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=Path(job["inputs"][0]["path"]);sys.path.insert(0,str(r/"runtime_bundle"))
from phase_runner import same_process,utc,atomic_json
from native_identity_observer import observe
from native_engines_service_config import checked_config
def ref(f):
 raw=Path(f).read_bytes();return dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
st=json.loads((r/"state.json").read_text());assert st["status"]=="completed"and st["completed_stages"]==["prepare","switch","function"]and not same_process(st["owner"])
spec=json.loads((r/"controller_spec.json").read_text());assert hashlib.sha256((r/"controller_spec.json").read_bytes()).hexdigest()==st["spec_sha256"]
for x in spec["stages"][0]["sources"]:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]
summary=json.loads((r/"feature_summary.json").read_text());assert summary["functional_acceptance"]and summary["completed"]==4and summary["effective_output_tokens"]==100
expected={"auto":"D0","required":"D1","named":"D1","none":"D0"}
trace=[json.loads(l)for l in(r.parent/"GLM-RUN-0125/router_trace.jsonl").read_text().splitlines()]
for row in summary["requests"]:
 assert row["native_owner"]==expected[row["name"]]and row["valid"]and len(row["token_ids"])==row["usage"]["completion_tokens"]and ref(row["wire"]["path"])==row["wire"]
 lease=next(x for x in trace if x["event"]=="lease_acquired"and x.get("lease_id")==row["lease_id"])
 assert lease["replica"]==row["native_owner"]and lease["body_sha256"]==ref(r/"tools_V14_mixed"/(row["name"]+".body.json"))["sha256"]
 wires=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==row["lease_id"]];release=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==row["lease_id"]]
 assert len(wires)==len(release)==1and release[0]["released"]and not release[0]["backend_failure"]and Path(wires[0]["wire_path"]).read_bytes()==Path(row["wire"]["path"]).read_bytes()
 if row["name"]!="none":assert row["tools"][0]["name"]=="get_weather"and json.loads(row["tools"][0]["arguments"])=={"city":"Shanghai"}
 else:assert not row["tools"]and row["content_chars"]>0
acks=[json.loads(l)for l in(r/"tools.stdout").read_text().splitlines()if l.startswith('{"event":')];assert [a["event"]for a in acks]==["task_acl_init","task_acl_finalize"]and all(a["returncode"]==0for a in acks)
exe=json.loads((r/"tools_execution.json").read_text());assert exe["status"]=="succeeded"and not exe["native_client_alive"]and not exe["signal_attempts"]
for key,c in summary["native_credits"].items():
 d=summary["native_delta"][key];assert d["vllm:generation_tokens_total"]==c["outputs"]and d["vllm:prompt_tokens_total"]==c["prompts"]and d["vllm:request_success_total"]==c["completed"]and d["vllm:num_preemptions_total"]==0
conf,_=checked_config(r/"restored/service_config.json");proof=json.loads((r/"restored/public_service_proof.json").read_text());assert same_process(proof["host"])and proof["marker"]["gateway_version"]=="V14"and ref(proof["config"]["path"])==proof["config"]
fresh=observe(r/"restored/service_config.json");assert all(g["status"]=="healthy"for g in fresh["groups"])
for group in fresh["groups"]:assert sum(h["expected_native_workers"]for h in group["hosts"])==16
for f in [r/"restored/public.gateway.log"]:
 acks=[json.loads(l)for l in f.read_text().splitlines()if l.startswith('{"event":')];assert len(acks)==1and acks[0]["event"]=="task_acl_init"and acks[0]["returncode"]==0
with urllib.request.urlopen("http://127.0.0.1:8000/control/replicas",timeout=8)as response:placement=json.loads(response.read())
assert len(placement["replicas"])==2and all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]for x in placement["replicas"])
with urllib.request.urlopen("http://127.0.0.1:8000/v1/responses/resp_glm_run211_D1_new",timeout=8)as response:raw=response.read()
assert json.loads(raw)==json.loads((r/"before_STORE211.wire").read_text())
out=dict(at=utc(),functional_acceptance=True,run_id=r.name,completed=4,effective_output_tokens=100,public_proof=proof,source_count=len(spec["stages"][0]["sources"]),native_fresh=fresh,native_credits=summary["native_credits"],native_delta=summary["native_delta"],checks=["All frozen source/spec identities match","Mixed real tools native tokenIDs/semantic/body/wire/usage/lease","SDK client init/final0/native32 live roots/epochs unchanged","Public V14 init0/both healthy/STORE211 byte exact"],limits=summary["limits"],Current=None)
f=j/"reduction.json";atomic_json(f,out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run240 completed; actual V14 mixed tool routing 4/100 valid with native32/SDK/STORE unchanged",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(f),locator="readonly frozen-source and actual E2E audit")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,completed=4,outputs=100)))
