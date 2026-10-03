from pathlib import Path
import json,subprocess,sys,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0161";roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def probe(label):
 out={}
 for key,o in roots.items():
  args=["python3","-c",(r/"live_probe.py").read_text()]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=100);(j/(label+"_"+key+".stdout")).write_bytes(z.stdout);(j/(label+"_"+key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
  expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 for node,port in[("166",9081),("167",9900)]:
  b=http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=15).read();(j/(label+"_"+node+".metrics")).write_bytes(b);d={}
  for l in b.decode().splitlines():
   if not l or l.startswith("#"):continue
   k=l.split("{")[0].split()[0]
   if k in ["vllm:request_success_total","vllm:generation_tokens_total","vllm:prompt_tokens_total","vllm:prefix_cache_queries_total","vllm:prefix_cache_hits_total","vllm:num_preemptions_total","vllm:num_requests_running","vllm:num_requests_waiting"]:d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
  assert d["vllm:num_requests_running"]==d["vllm:num_requests_waiting"]==0;out[node]=d
 return out

assert json.loads((r/"prepare.phase.json").read_text())["exit_code"]==1and not (r/"experiment.phase.json").exists()
log=(r/"prepare.log").read_text();assert 's["status"]=="failed"'in log and "AssertionError"in log
spec=json.loads((r/"controller_spec.json").read_text())
for row in spec["stages"][0]["sources"]:assert Path(row["path"]).read_bytes()==Path(row["snapshot"]).read_bytes()and hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()==row["sha256"]
actual=probe("terminal")
prior=json.loads((p/"jobs/TOKEN-INPUT-CPU3-20261003T0837Z/reduction.json").read_text())["native_metrics"]
assert actual==prior
out=dict(at=utc(),run_id=r.name,execution_verdict="INVALID",readonly_terminal_acceptance=True,failure="Copiedprepare retained158failed-state assertion against actual160completed; before allresource/HTTP operations",source_pins=len(spec["stages"][0]["sources"]),native32_same=True,native_counter_delta0=True,new_native_inference_outputs=0,client_SDK_operations=0,model_public_privatepolicy_ops=0,native_metrics=actual,original_source_raw_preserved=True)
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="161originalINVALID atprepare previous-state fixture; source103/raw kept; readonlyterminalaudit native32same/counter0/newinference0/noSDK-model-public-privatepolicyops.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="Failurebeforeallops/source/native32/counter0")],unknowns=["No nativepromptreuse performance/semantic credit"],decision_request=None,next_check_at=None));print(json.dumps(out))
