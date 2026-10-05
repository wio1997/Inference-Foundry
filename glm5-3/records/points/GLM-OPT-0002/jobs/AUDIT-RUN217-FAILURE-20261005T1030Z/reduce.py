from pathlib import Path
import sys,json,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0217";folder=r/"restored"
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def metric(b):
 out={}
 for l in b.decode().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{")[0].split()[0]
  if k.startswith("vllm:"):out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
state=json.loads((r/"state.json").read_text());assert state["status"]=="failed"and not same_process(state["owner"])and state["completed_stages"]==["prepare","adopt"]
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==180
for st in spec["stages"]:assert st["sources"]==pins
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(x["path"])["sha256"]==x["sha256"]
for n in["prepare","adopt"]:
 a=json.loads((r/(n+".phase.json")).read_text());assert a["status"]=="succeeded"and a["exit_code"]==0
phase=json.loads((r/"switch.phase.json").read_text());assert phase["status"]=="failed"and phase["exit_code"]==1
roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text());plans=json.loads((folder/"standalone_launch.json").read_text())
args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["python3","-c",(r/"live_probe.py").read_text()])]
z=subprocess.run(args,input=json.dumps(dict(owner=roots["node1"],NPU_count=16)).encode(),capture_output=True,timeout=120);(j/"D1.owner.stdout").write_bytes(z.stdout);z.check_returncode();v=json.loads(z.stdout)
expected={x["pid"]:x["identity"]for x in members["node1"]["owned_targets"]}
assert v["npu_worker_pids"]==members["node1"]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
dead={x["pid"]:not same_process(dict(pid=x["pid"],**x["identity"]))for x in members["node0"]["owned_targets"]};assert all(dead.values())
raw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60);(j/"166.npu_smi.stdout").write_text(raw)
npus=re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+[^|]+\|\s+\d+\s+\|",raw,re.M);assert not npus
native=Path(plans["node0"]["log"]).read_bytes();(j/"native216_failed.stdout").write_bytes(native);text=native.decode(errors="replace")
assert "Index out of range in dimension 0" in text and "exceeds bounds 3384" in text and "sample_tokens" in text
events=[json.loads(l)for l in text.splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
representative=[]
for l in text.splitlines():
 if ("Worker_PP1_TP0_DCP0 pid=1535589"in l and any(t in l for t in[" File ","RuntimeError:","Since the operator"]))or"[dump_input.py:79]"in l or"Index out of range in dimension 0"in l:
  if l not in representative:representative.append(l)
(j/"native_error_excerpt.txt").write_text("\n".join(representative)+"\n")
client=json.loads((r/"client_final_summary.json").read_text());assert not client["valid"]and client["effective_output_tokens"]==0
rows=client["requests"];assert [x["name"]for x in rows]==["D1_retained","retired214_get","retired214_previous","add2"]
assert rows[-1]["status"]==500and all(x["status"]==503and x["rejected_before_lease_and_RPC"]for x in rows[1:3])
stored=json.loads((p/"runs/GLM-RUN-0211/client_final_summary.json").read_text());source=next(x for x in stored["requests"]if x["name"]=="newD1_create")
assert Path(rows[0]["wire"]["path"]).read_bytes()==Path(source["wire"]["path"]).read_bytes()
for row in rows:
 for k in["body","wire"]:assert ref(row[k]["path"])==row[k]
events=[json.loads(l)for l in(r/"client_final.stdout").read_text().splitlines()if l.startswith('{"event":')]
assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and all(x["returncode"]==0for x in events)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://172.16.10.167:9900/metrics",timeout=15)as response:b=response.read()
(j/"D1.metrics").write_bytes(b);before=metric((r/"before_167.metrics").read_bytes());after=metric(b)
assert all(after[k]==v for k,v in before.items()if k.endswith("_total"))
assert after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==after["vllm:kv_cache_usage_perc"]==0
with http.open("http://127.0.0.1:8000/control/replicas",timeout=15)as response:placement=json.loads(response.read())
byid={x["id"]:x for x in placement["replicas"]};assert byid["D0"]["group_faulted"]and not byid["D1"]["group_faulted"]and all(not x["active_requests"]for x in placement["replicas"])
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and public["SDK_init0"]
retired=json.loads((r/"retire214_public.json").read_text());assert retired["SDKinit_finalize0"]and retired["models_signalled"]==0
oldpublic=json.loads((r/"old_public_service_proof.json").read_text());assert not same_process(oldpublic["host"])
assert (p/"runs/GLM-RUN-0214/restored/identity_observer/terminal.json").exists()
out=dict(at=utc(),run_id=r.name,driver_terminal_failed=True,functional_acceptance=False,verdict="INVALID",source_count=180,native_V2_ready_fit_prior=True,actual_first_native_request=dict(prompt_tokens=19,max_output_tokens=32,K=2,temperature=0,HTTP500=True,effective_output_tokens=0),native_all_original_D0_targets_inactive=True,D0_NPU_processes=0,D1_native16_same=True,D1_all_totals_unchanged=True,STORE211_byteexact=True,retired214STORE503_prelease_RPC=True,SDKclient_init_finalize0=True,native216_API_SDKinit_finalize0=True,public214_SDKinit_finalize0=True,public217_SDKinit0active=True,public=public,D0_group_faulted=True,D1_group_healthy=True,error_source=ref(j/"native216_failed.stdout"),error_excerpt=ref(j/"native_error_excerpt.txt"),D0_inactive_targets=dead,placement=placement,Current=None,limits=["Run217 actualnativeV2 firstrequestfailed indexoutofrange bounds3384; realE2E rejects prototype, no fullfunction/performanceKEEP","Asynchronous stack at PPbroadcast doesnotlocate originating index operation; exactcauseunknown, diagnosticblocking run/sourcebinding investigation justified","No D1 inference/signals or counters changed, public217 retained with D0fault/healthyD1, no automaticoldqueue/replay","No edit toexecuted216217source/spec/raw; capturefitnotrequestcorrectness; nextRun uniquecontroller only"])
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="217 actualV2 firstnative19prompt request HTTP500/indexoutofrange D0exit0NPU; frozen source180/SDK0/public217D0fault/D1retained16-STORE-counters readonlyaudit VALID, RunfunctionINVALID",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="frozen180-firstnative500-indexbounds3384-D0dead-D1retained-SDK0")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(audit_VALID=True,functionINVALID=True,D0inactive=True,D1native16=True)))
