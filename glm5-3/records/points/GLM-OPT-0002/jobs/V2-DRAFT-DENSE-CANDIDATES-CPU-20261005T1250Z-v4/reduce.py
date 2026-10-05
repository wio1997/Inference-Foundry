from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;old=j.parent/"V2-DRAFT-DENSE-CANDIDATES-CPU-20261005T1250Z-v3"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
v=json.loads((old/"probe.json").read_text());assert v["CPU_contract_VALID"]and v["NPU_tensors"]==0and not v["runtime_activation"]
events=[json.loads(l)for l in(old/"probe.stdout").read_text().splitlines()if l.startswith('{"event":')]
sdk=[x for x in events if x["event"]in["task_acl_init","task_acl_finalize"]]
assert len(sdk)==2 and[s["event"]for s in sdk]==["task_acl_init","task_acl_finalize"]and all(s["returncode"]==0for s in sdk)
assert len([x for x in events if x["event"]=="fullCLI_PP32_exactplan"])==1
v.update(at=utc(),SDKinit_finalize0=True,SDKevents=sdk,Current=None,prior_native_CPU=ref(old/"probe.json"),prior_stdout=ref(old/"probe.stdout"),prior_execute_wrapper_failed=True,CPU_reexecuted=False,reason="v3 actualCPU passed; wrapper treated fullCLI event as SDK event and looked for absentreturncode; readonlyfiltered reduction")
atomic_json(j/"reduction.json",v)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Readonly actualv3 CPU fullCLI/draft dense1..8 nativecandidate VALID, old1pad2/7-8eagerfallback, SDK0/noNPU; wrapperv3failurepreserved noCPUrerun",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="actualv3-nativeCPUdescriptors-nativeCLI-SDK0")],unknowns=v["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(CPU_VALID=True,no_reexecution=True)))
