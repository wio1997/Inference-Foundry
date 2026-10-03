from pathlib import Path
import sys,json,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+str(j/"cpu.py")
z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=120);(j/"cpu.stdout").write_bytes(z.stdout);(j/"cpu.stderr").write_bytes(z.stderr);z.check_returncode()
events=[json.loads(l) for l in z.stdout.decode().splitlines() if l.startswith('{"event":')]
assert events[0]["event"]=="task_acl_init" and events[-1]["event"]=="task_acl_finalize" and events[0]["returncode"]==events[-1]["returncode"]==0
o=next(x for x in events if x["event"]=="CPU_PP_empty_token_guard");o.update(at=utc(),SDKinit_finalize0=True)
atomic_json(j/"reduction.json",o);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Actual native PP branch reproduces mixed-empty IndexError using synthetic metadata; candidate guard skips empty indexing, nonempty commit cases unchanged. CPU only, SDK0; full native E2E required.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="Synthetic metadata under actual native source branch")],unknowns=["Full native PP4 functional correctness and later shape error"],decision_request=None,next_check_at=None))
print(json.dumps(o))
