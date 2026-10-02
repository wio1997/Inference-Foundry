from pathlib import Path
import json,subprocess,hashlib,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;root=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3")
env=dict(PYTHONPATH=str(root/"runtime")+":"+str(root/"tests"))
shell="export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+env["PYTHONPATH"]+":$PYTHONPATH; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run76/native_acl_lifecycle.py script "+str(j/"inner.py")
args=["docker","exec","glm52-single","bash","-c",shell]
z=subprocess.run(args,capture_output=True,timeout=180);(j/"CPU.stdout").write_bytes(z.stdout);(j/"CPU.stderr").write_bytes(z.stderr);z.check_returncode()
acks=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
a=json.loads((j/"race_reduction.json").read_text());sources={}
for f in[root/"runtime/response_affinity_gateway_v5.py",root/"runtime/request_estimates.py",root/"runtime/prefill_work_placement.py",root/"tests/test_prefill_workload.py",root/"runtime/placement.py",root/"runtime/response_affinity.py",root/"tests/test_response_budget_v6.py",root/"records/points/GLM-OPT-0002/runs/GLM-RUN-0039/restart/responses_json.wire"]:
 sources[str(f)]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
out=dict(at=utc(),CPU_contracts=a,sources=sources,SDK_init_finalize_zero=True,native_actual_requests=0,models_started=0,model_signals=0,limits=["CPU20 route-awarebudget andownershipcontracts only; nativebackground occupancy is not represented by completedHTTPlease, no capacity/performance/GPUproof","Native JSON fixture typedvalidation preserved, native GPU services untouched"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="CPU gatewayV5 contracts and concurrentcustomID reservation regressions passed; no actualinference",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualgatewayV5ASGI/native39fixture/ownershipreservation/nativeResponsesbudget")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(out))
