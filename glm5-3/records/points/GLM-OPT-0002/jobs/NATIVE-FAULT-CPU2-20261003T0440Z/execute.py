from pathlib import Path
import json,sys,subprocess,shlex,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;pins=json.loads((j/"source_pins.json").read_text())
for pin in pins:
 assert Path(pin["path"]).read_bytes()==Path(pin["snapshot"]).read_bytes()and hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+shlex.quote(str(j/"cpu_probe.py"))+" "+shlex.quote(str(j))
z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=120);(j/"cpu.stdout").write_bytes(z.stdout);(j/"cpu.stderr").write_bytes(z.stderr);z.check_returncode()
acks=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
v=next(x for x in acks if x["event"]=="native_fault_CPU_valid");assert len(v["contracts"])==18and v["native_requests"]==v["models"]==v["NPU_workers_started"]==v["signals"]==0
out=dict(at=utc(),valid=True,CPU_contracts=v,actual_SDK_init=acks[0],actual_SDK_finalize=acks[-1],source_pins=pins,live_public147_still_old_loaded_code=True,native_requests=0,models=0,signals=0,limits=v["limits"]);atomic_json(j/"reduction.json",out)
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Nativeepoch-specific durablequarantine actualV11 ASGI/18CPUcontracts/SDK0/no leaseinvention/clear/crossownerreplay; no physicaloperations",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="nativeepoch-specificfault/owner/lease/durability/HTTP/JournIO CPUproof")],unknowns=v["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(contracts=len(v["contracts"]),SDK0=True,native_requests=0)))
