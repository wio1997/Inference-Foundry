from pathlib import Path
import sys,json,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0221";folder=r/"restored";state125=p/"runs/GLM-RUN-0125"
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def metric(b):
 out={}
 for l in b.decode().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{")[0].split()[0]
  if k.startswith("vllm:"):out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
s=json.loads((r/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==179and all(x["sources"]==pins for x in spec["stages"])
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(x["path"])["sha256"]==x["sha256"]
for n in["prepare","fault","restore","switch","diagnose"]:
 v=json.loads((r/(n+".phase.json")).read_text());assert v["status"]=="succeeded"and v["exit_code"]==0and not v["timed_out"]
c=json.loads((r/"diagnostic_client_summary.json").read_text());assert c["HTTP_status"]in[200,500]and c["effective_output_tokens"]in[0,2]
for k in["body","wire"]:assert ref(c[k]["path"])==c[k]
prior=json.loads((p/"runs/GLM-RUN-0217/client_final_add2_3.body").read_text());body=json.loads(Path(c["body"]["path"]).read_text());prior.pop("cache_salt");body.pop("cache_salt");assert prior==body
events=[json.loads(l)for l in(r/"diagnostic_client.stdout").read_text().splitlines()if l.startswith('{"event":')]
assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and all(x["returncode"]==0for x in events)
roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text());plans=json.loads((folder/"standalone_launch.json").read_text())
args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["python3","-c",(r/"live_probe.py").read_text()])]
z=subprocess.run(args,input=json.dumps(dict(owner=roots["node1"],NPU_count=16)).encode(),capture_output=True,timeout=120);z.check_returncode();(j/"D1.owner.stdout").write_bytes(z.stdout);v=json.loads(z.stdout)
assert v["npu_worker_pids"]==members["node1"]["npu_worker_pids"];expected={x["pid"]:x["identity"]for x in members["node1"]["owned_targets"]};assert all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
raw=Path(plans["node0"]["log"]).read_bytes();(j/"native221.stdout").write_bytes(raw);text=raw.decode(errors="replace")
assert text.count("npu model runner v2 is in developing")==16and text.count("GLM_SFA_DCP_METADATA_DIAGNOSTIC_INSTALLED")==16and"8/8"in text and"Graph capturing finished"in text
metadata=[json.loads(l.split("GLM_SFA_DCP_METADATA_DIAGNOSTIC ",1)[1])for l in text.splitlines()if "GLM_SFA_DCP_METADATA_DIAGNOSTIC "in l];assert metadata
atomic_json(j/"runtime_metadata.json",metadata)
mismatches=[x for x in metadata if x["query_sum"]!=x["repeat_output_size"]or x["num_actual_tokens"]>x["query_sum"]]
actual_query_equal=all(x["query_GPU"]==x["query_CPU"]for x in metadata)
excerpt=[l for l in text.splitlines()if any(k in l for k in["GLM_SFA_DCP_METADATA_DIAGNOSTIC ","Index out of range"," File ","RuntimeError:","Dumping scheduler output"])]
(j/"error_metadata_excerpt.txt").write_text("\n".join(excerpt)+"\n")
inactive={x["pid"]:not same_process(dict(pid=x["pid"],**x["identity"]))for x in members["node0"]["owned_targets"]}
npuraw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60);(j/"166.npu_smi.stdout").write_text(npuraw)
npus={int(x)for x in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+[^|]+\|\s+\d+\s+\|",npuraw,re.M)}
if c["HTTP_status"]==500:
 assert not c["semantic_pass"]and c["effective_output_tokens"]==0and all(inactive.values())and not npus
 assert"Index out of range in dimension 0"in text and"exceeds bounds 3384"in text
else:
 assert c["semantic_pass"]and c["effective_output_tokens"]==2and len(npus)==16and npus==set(members["node0"]["npu_worker_pids"])
 live=subprocess.run(["python3","-c",(r/"live_probe.py").read_text()],input=json.dumps(dict(owner=roots["node0"],NPU_count=16)).encode(),capture_output=True,timeout=120);live.check_returncode();(j/"D0.owner.stdout").write_bytes(live.stdout)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://172.16.10.167:9900/metrics",timeout=15)as res:b=res.read()
(j/"D1.metrics").write_bytes(b);before=metric((r/"before_167.metrics").read_bytes());after=metric(b);assert all(after[k]==v for k,v in before.items()if k.endswith("_total"))and after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==0
with http.open("http://127.0.0.1:8000/control/replicas",timeout=15)as res:placement=json.loads(res.read())
by={x["id"]:x for x in placement["replicas"]};assert not by["D1"]["group_faulted"]and all(not x["active_requests"]for x in placement["replicas"])
assert by["D0"]["group_faulted"]==(c["HTTP_status"]==500)
retired=json.loads((r/"retire219_public.json").read_text());assert retired["SDKinit_finalize0"]and retired["models_signalled"]==0and(p/"runs/GLM-RUN-0219/restored/identity_observer/terminal.json").exists()
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and public["SDK_init0"]
config=json.loads((folder/"service_config.json").read_text());epochs={x["id"]:x["epoch"]for x in config["native_domains"]}
out=dict(at=utc(),run_id=r.name,diagnostic_measurement_valid=True,functional_acceptance=c["semantic_pass"],verdict="INCONCLUSIVE"if c["semantic_pass"]else"INVALID",source_count=179,Graph8fit=True,native_math_edits=0,metadata_observation_adds_synchronization=True,prior217_body_same_except_coldsalt=True,HTTP_status=c["HTTP_status"],effective_output_tokens=c["effective_output_tokens"],runtime_metadata=ref(j/"runtime_metadata.json"),runtime_metadata_rows=len(metadata),query_GPU_CPU_equal=actual_query_equal,extent_mismatches=mismatches,D0_allinactive=all(inactive.values()),D0_inactive_targets=inactive,D0_NPU=len(npus),D1_native16_allcounters_retained=True,public=public,native_epochs=epochs,SDKclient_init_finalize0=True,public219_SDKfinal0=True,public221_SDKinit0active=True,native_error=ref(j/"native221.stdout"),excerpt=ref(j/"error_metadata_excerpt.txt"),Current=None,limits=["NativeGraph sameK2/SFADCP requestdiagnostic withmetadata sync; actualfields/IndexCheck determinequeryextent hypothesis, no performanceclaim","Original217asynchronous error falsefirststack location; no operator/guard/mathematical edits, originalSFA functionstillcalled","Eager219 success onecase notfullAPI; thinkingbudgetgap open, no KEEP/stablecapacity/globalbound"])
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="221 originalGraph/nativeV2 metadata diagnostic readonlyaudit source179/Graph8/SDK0/D1retained actualfields andnativefirstrequest outcome, noperformanceKEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="sameGraph179-actualquerymetadata-SDK0-nativeoutcome-D1retained")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(HTTP_status=c["HTTP_status"],actual_mismatches=len(mismatches),GPU_CPUequal=actual_query_equal,D0_NPU=len(npus))))
