from pathlib import Path
import json,sys,subprocess,shlex,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0155";s=json.loads((r/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-RUN155-20261003T0626Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="28b9c3f37ba871420126bb7712b6caadfbf646456b9b07f039b8af560c3c192b"
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
def probe(label):
 data={}
 for key,o in roots.items():
  args=["python3","-c",(r/"live_probe.py").read_text()]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(label+"_"+key+".stdout")).write_bytes(z.stdout);(j/(label+"_"+key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout);assert v["npu_worker_pids"]==members[key]["npu_worker_pids"];expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);data[key]=dict(root_same=True,NPU16_same=True)
 return data
before=probe("before");cases=[]
for name in["baseline","PP38_40","PCP2_DCP16","PCP2_DCP2","PCP2_DCP1"]:
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137:$PYTHONPATH; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+str(j/"inner.py")+" "+name
 z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=160);(j/(name+".stdout")).write_bytes(z.stdout);(j/(name+".stderr")).write_bytes(z.stderr);z.check_returncode()
 events=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
 case=next(x for x in events if x["event"]=="CPU_geometry_config");case["SDK"]=[events[0],events[-1]];cases.append(case)
assert cases[0]["config_accepted"]
after=probe("after")
native_refs=[]
for f in["/vllm-workspace/vllm/vllm/v1/metrics/stats.py","/vllm-workspace/vllm/vllm/config/parallel.py"]:
 code="from pathlib import Path;import hashlib,json;p=Path("+repr(f)+");b=p.read_bytes();print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))"
 z=subprocess.run(["docker","exec","glm52-single","python3","-c",code],capture_output=True,timeout=30);z.check_returncode();native_refs.append(json.loads(z.stdout))
out=dict(at=utc(),kind="CPU_native_fullconfig_parallel_geometry",measurement_valid=True,cases=cases,before=before,after=after,source_refs=native_refs,native_timing_semantics=dict(prefill="firstSCHEDULED→firstNEW_TOKEN EngineCore monotonic",queue="firstQUEUED→firstSCHEDULED, QUEUED emittedScheduler.add_request afterengineinbox/inputprocessing",TTFT="frontendarrival_time→OutputProcessor IterationStats walltimestamp; residual includesotherbeforeQUEUED/aftercoreoutput unknown"),new_inference=0,model_operations=0,limits=["Native config/metadata resolver acceptance plus HFfullIndex boundary is necessary, not weights/modelconstruct/HCCL/Graph/STORE/fullE2E correctness/latency/fit proof","PCP expands world, nativePCP-DCP sizes1/PCP/TPxPCP; DP>1 unsupported; hardware still32, no extra node","PP38,40 differs39,39/40,38 previouslyinvalid IndexShare boundary; HFboundaryfull doesnotcertifymodel","Timingresidual notGPUbound/not assignedtotokenizer-or-outputdelivery withoutnewtimestamps"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="VALID CPUfullnativegeometry/config/resolver/HFboundary, currentnative32unchanged/noModel/NPUinference",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="geometry",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualnativefullEngineArgs/ascendresolvers/CPUonlySDK0")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps([dict(case=x["case"],accepted=x["config_accepted"],error=x.get("error"),world=x.get("world"),PCP=x.get("PCP"),DCP=x.get("DCP"))for x in cases]))
