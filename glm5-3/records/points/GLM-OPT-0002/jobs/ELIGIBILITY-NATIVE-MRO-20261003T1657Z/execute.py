from pathlib import Path
import json,subprocess,sys,hashlib,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0189"
owner=json.loads((r/"state.json").read_text());assert owner["status"]=="completed"and not same_process(owner["owner"])
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def probe(label):
 for key,o in roots.items():
  args=["python3","-c",(r/"live_probe.py").read_text()]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=100)
  (j/(label+"_"+key+".stdout")).write_bytes(z.stdout);(j/(label+"_"+key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
  expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
  assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
import urllib.request,re
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def totals(label):
 out={}
 for node,port in[("166",9081),("167",9900)]:
  with http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=15)as res:b=res.read()
  (j/(label+"_"+node+".metrics")).write_bytes(b)
  vals={}
  for l in b.decode().splitlines():
   if not l.startswith("vllm:"):continue
   k=l.split("{")[0].split()[0]
   if k.endswith("_total"):vals[k]=vals.get(k,0)+float(l.rsplit(" ",1)[1])
  out[node]=vals
 return out
before_totals=totals("before_totals")
probe("before")
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+str(j/"config.py")
z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=240)
(j/"client.stdout").write_bytes(z.stdout);(j/"client.stderr").write_bytes(z.stderr);z.check_returncode()
events=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')]
assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
v=next(x for x in events if x["event"]=="CPU_geometry_config")
assert v["config_accepted"]and (v["TP"],v["PP"],v["DCP"])==(4,4,4)and v["workers"]==v["NPU_tensor_allocations"]==v["model_instances"]==v["inference"]==0
probe("after")
assert totals("after_totals")==before_totals
sources=[]
code="from pathlib import Path;import json,hashlib;a="+repr(["/vllm-workspace/vllm/vllm/config/speculative.py","/vllm-workspace/vllm-ascend/vllm_ascend/platform.py","/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py"])+";print(json.dumps([dict(path=n,bytes=len(Path(n).read_bytes()),sha256=hashlib.sha256(Path(n).read_bytes()).hexdigest())for n in a]))"
q=subprocess.run(["docker","exec","glm52-single","python3","-c",code],capture_output=True,timeout=30);q.check_returncode();sources=json.loads(q.stdout)
out=dict(at=utc(),valid=True,kind="actual_imported_native_scheduler_source_identity",actual={k:x for k,x in v.items()if k!="actual_method_source"},actual_method_source_ref=ref(Path(v["actual_schedule_method"]["snapshot"])),SDK_init_finalize0=[events[0],events[-1]],native32_same=True,completed_prior_controller=owner["owner"],sources=sources+[ref(Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/issue_budget_scheduler_v5.py"))],
 limits=["Actual imported native fullCLI source identities and snapshots only; no NPU workers/tensors/models/inference; same32 and vllm counters/SDK0; not cadence or performance proof"])
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Actual imported native scheduler class/method paths and source identities captured; native32 same/counters0/SDK0/no inference",
 execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="CPU actualfullconfig/source/Graph/dynamicguard/SDK/native32")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,K=v["K"],evidence=e)))
