from pathlib import Path
import json,subprocess,sys,hashlib,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0180"
owner=json.loads((r/"state.json").read_text());assert owner["status"]=="running"and same_process(owner["owner"])
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
probe("before")
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+str(j/"config.py")
z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=240)
(j/"client.stdout").write_bytes(z.stdout);(j/"client.stderr").write_bytes(z.stderr);z.check_returncode()
events=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')]
assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
v=next(x for x in events if x["event"]=="CPU_geometry_config")
assert v["config_accepted"]and v["K"]==1 and (v["TP"],v["PP"],v["DCP"])==(4,4,4)and v["workers"]==v["NPU_tensor_allocations"]==v["model_instances"]==v["inference"]==0
probe("after")
sources=[]
code="from pathlib import Path;import json,hashlib;a="+repr(["/vllm-workspace/vllm/vllm/config/speculative.py","/vllm-workspace/vllm-ascend/vllm_ascend/platform.py","/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py"])+";print(json.dumps([dict(path=n,bytes=len(Path(n).read_bytes()),sha256=hashlib.sha256(Path(n).read_bytes()).hexdigest())for n in a]))"
q=subprocess.run(["docker","exec","glm52-single","python3","-c",code],capture_output=True,timeout=30);q.check_returncode();sources=json.loads(q.stdout)
out=dict(at=utc(),valid=True,kind="CPU_static_K1_DCP4_fullCLI",actual=v,SDK_init_finalize0=[events[0],events[-1]],native32_same=True,active_controller=owner["owner"],sources=sources,
 limits=["Native fullCLI accepts staticK1/TP4PP4DCP4 andGraphquerylen2; no model/NPUtensors/inference launched; CPUvalid notNPUfit/performance/KEEP",
 "DynamicK+DCP4 nativeguard rejection mustremain; no bypass/proposer/math change",
 "Run180ownedcontroller actively infers while CPUjob reads; nativecounterincrease is expected existing work, not CPUjobmodel inference"])
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="CPUonly staticK1 TP4PP4DCP4/fullCLI/Graphquerylen2 accepted; dynamicK+DCP4 nativerejected; same32/SDK0/noCPUmodeloperations; realK1fitE2E pending",
 execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="CPU actualfullconfig/source/Graph/dynamicguard/SDK/native32")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,K=v["K"],geometry=v["K1_graph_geometry"],evidence=e)))
