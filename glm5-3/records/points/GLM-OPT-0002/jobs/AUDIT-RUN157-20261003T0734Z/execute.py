from pathlib import Path
import sys,json,hashlib,subprocess,shlex,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0157"
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for x in spec["stages"][0]["sources"]:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(x["path"])["sha256"]==x["sha256"]
a=json.loads((r/"prepare.phase.json").read_text());b=json.loads((r/"experiment.phase.json").read_text());assert a["status"]=="succeeded"and b["status"]=="failed"and b["exit_code"]==1
v=json.loads((r/"formal/formal_summary.json").read_text());assert not v["valid"]and v["verdict"]=="INVALID"and "client_final_newD1_create.wire"in v["error"]
assert not (r/"formal/benchmark_command.json").exists()and not list((r/"formal").glob("benchmark/full_execution.json"))
events=[json.loads(l)for l in(r/"native_formal.stdout").read_text().splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
def counters(f):
 out={}
 for l in f.read_text().splitlines():
  if l.startswith("vllm:")and any(x in l.split("{")[0]for x in["_total","num_requests_running","num_requests_waiting"]):
   k=l.split("{")[0].split()[0];out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
delta={}
for key in["D0","D1"]:
 before=counters(r/"formal"/("before_reset_"+key+".metrics"));after=counters(r/"formal"/("cleanup_"+key+".metrics"));assert before==after and after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==0;delta[key]={k:after[k]-v for k,v in before.items()}
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());physical={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 ids={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);physical[key]=dict(NPU16_same=True,root=v["root"])
restore=json.loads((r/"restore_summary.json").read_text());assert restore["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=9)and restore["D1_unchanged_policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=4096,prefill_cadence=1,serial=1)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=15)as res:placement=json.loads(res.read())
assert all(not x["active_requests"]and not x["group_faulted"]and x["work_ranking_calibrated"]for x in placement["replicas"])
out=dict(at=utc(),run_id=r.name,measurement_valid=False,verdict="INVALID",source_pins=len(spec["stages"][0]["sources"]),cause="LegacySTORE evidence filename missed unique row suffix; original frozenfixture failed beforebenchmark/no nativeinference",native_delta=delta,new_inference=0,effective_output_tokens=0,SDKinit_finalize0=True,native32same=physical,restore=restore,public_healthy_idle=True,model_operations=0,frontend_operations=0,limits=["INVALID fixture is not nativeperformance rejection orcapacity evidence; source/raw kept","D0privatepolicy7→8→9 applied andrestored idle; D1private8192t4096serial1 unchanged; no requests toforceSELECTED marker"])
atomic_json(j/"reduction.json",out);atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run157 INVALID premeasurementSTOREfilenamefixture/no nativeinference0/SDK0/native32same/D0restore1024serial9, D1unchanged; originalraw preserved",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="terminalfailedfixture/source/SDK/counters/restore/native32")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(out))
