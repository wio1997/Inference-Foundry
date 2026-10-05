from pathlib import Path
import json,ast,hashlib,subprocess,shlex,urllib.request,re,sys,types
from datetime import datetime,timezone
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0208"
def ref(f):
 raw=f.read_bytes();return dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
prev=json.loads((p/"runs/GLM-RUN-0207/state.json").read_text());assert prev["status"]=="completed"and not same_process(prev["owner"])
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def check(label):
 proof={};counters={}
 for key,o in roots.items():
  args=["python3","-c",(r/"live_probe.py").read_text()]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=90);f=j/(label+"_"+key+".stdout");f.write_bytes(z.stdout);z.check_returncode();v=json.loads(z.stdout)
  known={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
  assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(known[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);proof[key]=ref(f)
  with http.open("http://172.16.10."+o["host"]+":"+str(9081 if o["host"]=="166"else 9900)+"/metrics",timeout=15)as response:raw=response.read()
  (j/(label+"_"+key+".metrics")).write_bytes(raw);vals={}
  for l in raw.decode().splitlines():
   if l.startswith("vllm:"):
    name=l.split("{")[0].split()[0]
    if name.endswith("_total")or name in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]:vals[name]=vals.get(name,0)+float(l.rsplit(" ",1)[1])
  assert vals["vllm:num_requests_running"]==vals["vllm:num_requests_waiting"]==vals["vllm:kv_cache_usage_perc"]==0;counters[key]=vals
 return proof,counters
before,bc=check("before")

args=["docker","exec","-e","PYTHONPATH="+str(j/"candidate")+":/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:/data/tiankuan/wio/Inference-Foundry/glm5-3/tests","glm52-single","python3",str(j/"test_candidate.py")]
z=subprocess.run(args,capture_output=True,timeout=180);(j/"test.stdout").write_bytes(z.stdout);(j/"test.stderr").write_bytes(z.stderr);z.check_returncode()
tests=json.loads(z.stdout.decode().splitlines()[-1]);assert tests["successful"]and tests["tests"]>=7
after,ac=check("after");assert bc==ac
sources=[ref(f)for f in (j/"candidate").glob("*.py")]
out=dict(valid=True,at=datetime.now(timezone.utc).isoformat(),tests=tests,candidate_sources=sources,native32_same_idle_vllm_counters=True,physical_before=before,physical_after=after,live_runtime_edits=0,nativepolicywrites=0,models=0,inference=0,NPUtensors=0,SDKcalls=0,Current=None,limits=["CPU mocks and staticcompiler proof; lease-empty does not assertnative GPUidle or progress","Native source-KV-MTP-operators-epochs unchanged; realcandidate publicE2E functional and latearrival workload comparison required","Onlyunboundsmall requests eligible; allpreferreddecode pools busy and availableprefillpeer lease-empty; atomic admission prevents simultaneousborrow; STOREowner/epoch/fault/drain unchanged"])
f=j/"reduction.json";f.write_text(json.dumps(out,ensure_ascii=False,indent=2)+chr(10))
result=dict(schema_version=1,job_id=j.name,status="completed",summary="CPU lease-empty short spill candidate/shapecompiler/fullbody-headers-STORE-affinity-nativefault-drain/concurrentadmission contracts VALID; native32 unchanged/no livecode-policy-model-inference",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(f),locator="CPUcandidate/contracts/native32")],unknowns=out["limits"],decision_request=None,next_check_at=None);(j/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10));print(json.dumps(dict(valid=True,proof=ref(f))))
