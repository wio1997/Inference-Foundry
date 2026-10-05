from pathlib import Path
import sys,json,subprocess,shlex,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
shell="ulimit -c 0; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp222:/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; export HCCL_NPU_SOCKET_PORT_RANGE=63100-63163 HCCL_HOST_SOCKET_PORT_RANGE=62500-62563 HCCL_BUFFSIZE=768 GLM_ISSUE_BUDGET_POLICY=/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp222/issue_budget_policy.json GLM_ISSUE_BUDGET_COHORT=GLM-COHORT-0222 GLM_ISSUE_BUDGET_DEFAULT=8192 PROFILING_MODE=dynamic VLLM_PP_LAYER_PARTITION=42,36 HCCL_LOGIC_SUPERPOD_ID=1 GLM_STATIC_K=2 VLLM_USE_V2_MODEL_RUNNER=1 ASCEND_LAUNCH_BLOCKING=0; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+shlex.quote(str(j/"probe.py"))+" "+shlex.quote(json.dumps(json.loads((j/"candidate_plan.json").read_text())["argv"]))
z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=180);(j/"probe.stdout").write_bytes(z.stdout);(j/"probe.stderr").write_bytes(z.stderr);z.check_returncode()
events=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and all(x["returncode"]==0for x in events)
v=json.loads((j/"probe.json").read_text());assert v["CPU_contract_VALID"]and v["NPU_tensors"]==0and not v["runtime_activation"];v.update(at=utc(),SDKinit_finalize0=True,Current=None)
atomic_json(j/"reduction.json",v)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Native fullCLIstaticK2 guard/CPU actualdraftdecode candidate old3n padsbatch1to3 vs dense1..8 exact; SDK0/noNPU/targetunchanged",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="actualnativeCPUmethod-queryextent-fixture-SDK0")],unknowns=v["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(CPU_VALID=True,SDK0=True)))
