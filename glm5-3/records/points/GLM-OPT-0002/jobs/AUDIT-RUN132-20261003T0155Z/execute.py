from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from standalone_service_config import checked_config
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0132";old=r.parent/"GLM-RUN-0131"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
owner=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text());assert owner["run_id"]==r.name and owner["status"]=="completed"
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"]
for pin in pins:
 assert Path(pin["path"]).read_bytes()==Path(pin["snapshot"]).read_bytes()and ref(Path(pin["path"]))["sha256"]==pin["sha256"]
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());live={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=100);(j/(key+".stdout")).write_bytes(z.stdout);(j/(key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 live[key]=dict(root=v["root"],NPU16_same=True)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://172.16.10.166:9081/metrics",timeout=10)as response:raw=response.read()
(j/"final.metrics").write_bytes(raw)
def values(b):
 return {k:sum(float(x)for x in re.findall(r"^vllm:"+k+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M))for k in ["num_requests_running","num_requests_waiting","generation_tokens_total","prompt_tokens_total","request_success_total","prefix_cache_queries_total","prefix_cache_hits_total","num_preemptions_total"]}
end=values(raw);assert end["num_requests_running"]==end["num_requests_waiting"]==0
initial=values((r/"pilot_initial.metrics").read_bytes());delta={k:end[k]-initial[k]for k in end if k not in["num_requests_running","num_requests_waiting"]}
pilot=json.loads((r/"pilot_summary.json").read_text());assert pilot["measurement_valid"]and pilot["functional_acceptance"]and pilot["effective_public_output_tokens"]==96
assert [x["usage"]["prompt_tokens"]for x in pilot["requests"]]==[21,81932]
for row in pilot["requests"]:
 item=row["wire"];assert ref(Path(item["path"]))==item
 assert row["contract"]["done"]and row["contract"]["finish_reasons"]=={"0":"length"}and not row["contract"]["native_error"]and row["completed"]
smoke=json.loads((r/"public_smoke_summary.json").read_text());assert smoke["functional_acceptance"]and smoke["effective_output_tokens"]==80
attempts=json.loads((r/"public_smoke_attempts.json").read_text());assert [x["status"]for x in attempts]==[200,200,200,200,503,503,200]
for row in attempts:assert ref(Path(row["wire"]["path"]))==row["wire"]
for name in ["native_pilot","native_public_smoke"]:
 rows=[json.loads(l)for l in(r/(name+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert [x["event"]for x in rows]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in rows)
assert delta["generation_tokens_total"]==176 and delta["request_success_total"]==5 and delta["num_preemptions_total"]==0
config,_=checked_config(r/"service_config.json");proof=json.loads((r/"public_service_proof.json").read_text());ident=proof["host"]
assert same_process(ident)
assert [v.decode()for v in Path("/proc/"+str(ident["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
ports=subprocess.check_output(["ss","-ltnp"],text=True);line=next(l for l in ports.splitlines()if re.search(r":8000\s",l));assert "pid="+str(ident["pid"])+","in line
with http.open("http://127.0.0.1:8000/healthcheck",timeout=10)as response:h=json.loads(response.read())
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as response:placement=json.loads(response.read())
assert h["status"]=="ok"and h["request_num"]==0and len(placement["replicas"])==1and placement["replicas"][0]["active_requests"]==0and not placement["replicas"][0]["group_faulted"]
trace=r.parent/"GLM-RUN-0125/router_trace.jsonl";events=[json.loads(l)for l in trace.read_text().splitlines()]
own=[e for e in events if e.get("event")=="lease_acquired"and e.get("created_at","")>""]
assert proof["container_marker"]["native_domains"]==config["native_domains"]
ack=[json.loads(l)for l in(r/"public.gateway.log").read_text().splitlines()if l.startswith('{"event":')];assert len(ack)==1and ack[0]["event"]=="task_acl_init"and ack[0]["returncode"]==0
reuse=json.loads((r/"reused131_audit.json").read_text());assert reuse["overall_verdict"]=="INVALID"and reuse["semantic_partial_valid"]and reuse["effective_output_tokens"]==11
for f in [r/"pilot_final.json",r/"public_smoke_final.json",r/"adopt_initial.json"]:assert json.loads(f.read_text())["NPU32_same"]and json.loads(f.read_text())["idle"]
index=[ref(f)for f in sorted(r.iterdir())if f.is_file()]
atomic_json(j/"artifact_index.json",index)
out=dict(at=utc(),run_id=r.name,measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",effective_output_tokens=176,completed_native_requests=5,helper_count=0,SDK_client_init_finalize0=True,semantic131_reused_not_recredited=11,prior131_overall_verdict="INVALID",source_count=len(pins),source_unchanged=True,native_epochs=live,NPU32_same=True,API_count=1,geometry="DP1TP16PP2DCP16/world32/local16/42,36/noEP_AllGather/noKV/K3noSPGraph32",native_cohort="GLM-COHORT-0131",pilot=pilot,public_smoke=smoke,native_counter_delta=delta,public_host=ident,public8000_retained=True,public_SDK_init0_active=True,idle=True,artifact_index=ref(j/"artifact_index.json"),limits=["Diagnostic176 outputs and three previous exactanswers reused; no formalSLO/stability/capacity/KEEP","CrossnodeTP16PP2 jointlychanges TP/DCP/locality/resourceparticipation/trajectory versuslocalTP8PP2; no isolatedgain orqualitycomparison","FulltypedSSE/tools/background/restart/error/concurrentcustomID fornewphysicalclass pending","Ownerjournal125 preservesbindings only; oldphysicalnativeSTORE data not replicated"])
atomic_json(j/"reduction.json",out)
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="needs_decision",summary="Run132 valid diagnostic176 outputs/SDK0/samecoupled131 nativeNPU32/public8000 healthy; noKEEP/capacity",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="complete diagnostic and retained native/public identity audit")],unknowns=out["limits"],decision_request="Proceed fullAPI and discriminating variableload research",next_check_at=None));print(json.dumps(dict(valid=True,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),delta=delta,public=ident)))
