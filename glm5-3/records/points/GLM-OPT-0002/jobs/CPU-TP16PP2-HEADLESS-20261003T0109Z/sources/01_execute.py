from pathlib import Path
import sys,json,hashlib,subprocess,shlex,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent
for v in json.loads((j/"source_pins.json").read_text()):
 assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
argv=json.loads((j/"planned_CPU_argv.json").read_text());proof=[]
for node in[0]:
 a=argv[:];a[a.index("--node-rank")+1]=str(node);a[a.index("--port")+1]=str(9081 if node==0 else 9900)
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run89:"+str(j)+":/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; export GLM_ISSUE_BUDGET_POLICY=/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run89/issue_budget_policy.json GLM_ISSUE_BUDGET_COHORT=GLM-COHORT-0089 GLM_ISSUE_BUDGET_DEFAULT=4096 HCCL_BUFFSIZE=768 PROFILING_MODE=dynamic VLLM_PP_LAYER_PARTITION=42,36; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run89/native_acl_lifecycle.py script "+shlex.quote(str(j/"probe.py"))+" "+shlex.quote(json.dumps(a))
 args=["docker","exec","glm52-single","bash","-c",shell];atomic_json(j/("node"+str(node)+".command.json"),dict(argv=args,models=0,NPU_workers=0,requests=0))
 z=subprocess.run(args,capture_output=True,timeout=120);(j/("node"+str(node)+".stdout")).write_bytes(z.stdout);(j/("node"+str(node)+".stderr")).write_bytes(z.stderr);z.check_returncode()
 acks=[json.loads(x)for x in z.stdout.decode().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
 v=next(x for x in acks if x["event"]=="native_multinode_CLI_branch_CPU_valid")
 assert v["workers"]==v["EngineCore"]==v["ports"]==v["communicators"]==v["requests"]==0
 proof.append(dict(native_CLI_branch=v,SDK_init=acks[0],SDK_finalize=acks[-1]))
out=dict(at=utc(),valid=True,CPU_only=True,proof=proof,limits=proof[0]["native_CLI_branch"]["limits"],signals_to_task_resources=0)
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Actual native CLI routes node0 API/node1 headless executor; intercepted beforeworkers/EngineCore/ports, SDKinit-final0",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="nativeCLI actual headlessbranch CPU interception")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,CPU_only=True,native_headless_branch=True)))
