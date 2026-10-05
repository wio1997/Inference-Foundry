from pathlib import Path
import sys,json,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0220";base=p/"runs/GLM-RUN-0219/restored"
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])and s["completed_stages"]==[]
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==178and all(st["sources"]==pins for st in spec["stages"])
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(x["path"])["sha256"]==x["sha256"]
error=(r/"install_native220.stderr").read_text();assert"SyntaxError: unexpected character after line continuation character"in error
assert not Path("/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp220").exists()and not Path("/data/tiankuan/wio/glm52-pd/deploy/logs/local_pp220_166.log").exists()
roots=json.loads((base/"standalone_root_identities.json").read_text());members=json.loads((base/"standalone_native_members.json").read_text());physical={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);z.check_returncode();(j/(key+".owner.stdout")).write_bytes(z.stdout);v=json.loads(z.stdout);expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);physical[key]=ref(j/(key+".owner.stdout"))
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def metric(b):
 out={}
 for l in b.decode().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{")[0].split()[0]
  if k.startswith("vllm:"):out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
for node,port in[("166",9081),("167",9900)]:
 with http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=15)as res:b=res.read()
 (j/(node+".metrics")).write_bytes(b);v=metric(b);assert v["vllm:num_requests_running"]==v["vllm:num_requests_waiting"]==0
 if node=="166":assert v["vllm:generation_tokens_total"]==2and v["vllm:prompt_tokens_total"]==19and v["vllm:request_success_total"]==1
 before=metric((r/("before_"+node+".metrics")).read_bytes());assert all(v[k]==n for k,n in before.items()if k.endswith("_total"))
public=json.loads((base/"public_service_proof.json").read_text());assert same_process(public["host"])and public["SDK_init0"]
with http.open("http://127.0.0.1:8000/control/replicas",timeout=15)as res:placement=json.loads(res.read())
assert all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]for x in placement["replicas"])
out=dict(at=utc(),run_id=r.name,driver_verdict="INVALID",readonly_no_GPU_operations_VALID=True,source_count=178,original_error="privateinstaller doubleescaped newlines SyntaxError beforeexecutingpython",native219211_same32_idle_alltotals=True,physical=physical,public219_same_sdkactive=True,new_private220_absent=True,new_native220_log_absent=True,new_models=0,new_inference=0,signals=0,Current=None,limits=["220FAILED/INVALIDpreparefixture, noGPU/request repeated; exact219/211/public219 preserved","NewRun221 corrects installer source beforefreeze, repeats controlpreparation only thennewauthorizedGraphdiagnostic"])
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="220 failedinstallerfixture escapednewline SyntaxError beforemutation; frozen178/native219211same32 counters/public219/private220absent readonlyaudit VALID, noGPUreplay",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="frozen178-noGPUmutation-native32-counters-same-public219")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(audit_VALID=True,models=0,inference=0,NPU32same=True)))
