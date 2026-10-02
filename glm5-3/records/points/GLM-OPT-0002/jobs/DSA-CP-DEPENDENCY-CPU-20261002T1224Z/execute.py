from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent
script="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run69:$PYTHONPATH HCCL_BUFFSIZE=4096; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run69/native_acl_lifecycle.py script "+str(j/"inner.py")
z=subprocess.run(["docker","exec","glm52-single","bash","-c",script],capture_output=True,timeout=300);(j/"CPU.stdout").write_bytes(z.stdout);(j/"CPU.stderr").write_bytes(z.stderr);z.check_returncode()
acks=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
out=json.loads((j/"dependency_reduction.json").read_text());out.update(at=utc(),SDK_init_finalize_zero=True);atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Actualnative DSACP+SP+DCP16 backend selected for2 fullCLI configs/K5capture48/Async preserved; currentworkspaceguard noteligible, keepsnative allocation; no models/GPUrequests/signals",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="nativeCLI/effectiveDSACP/SP/DSADCPclass/TPalignedK5geometry/sourcehashes")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(configs=len(out["configs"]),scope=out["limits"])))
