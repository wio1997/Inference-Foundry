from pathlib import Path
import json,subprocess,sys,hashlib,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;g=j.parents[5]
pins=json.loads((j/"source_pins.json").read_text())
for v in pins:
 assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()
 assert hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run89/native_acl_lifecycle.py script "+shlex.quote(str(j/"cpu_probe.py"))+" "+shlex.quote(str(j))
args=["docker","exec","glm52-single","bash","-c",shell];atomic_json(j/"command.json",dict(at=utc(),argv=args,models=0,requests=0))
z=subprocess.run(args,capture_output=True,timeout=120);(j/"cpu.stdout").write_bytes(z.stdout);(j/"cpu.stderr").write_bytes(z.stderr);z.check_returncode()
acks=[json.loads(v)for v in z.stdout.decode().splitlines()if v.startswith('{"event":')]
assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
proof=next(v for v in acks if v["event"]=="standalone_service_CPU_valid");assert len(proof["contracts"])==14 and proof["native_requests"]==proof["NPU_workers_started"]==0
out=dict(at=utc(),valid=True,CPU_contracts=proof,actual_SDK_init=acks[0],actual_SDK_finalize=acks[-1],source_pins=pins,native_requests=0,models=0,signals=0,limits=["Synthetic standalone root/NPU identities with actual nativeCPUtested CLI; headless identity change isCPUfixture, notphysicalrestart","ActualserviceconfigCLI andactualV11factory used, no listener/HTTP/native inference/modeloperation; currentpublic127P89D123 retained","ExistingV11fullnativeAPI117/sourcepinned reused; no capacity/KEEP/physicalAPIstate replication claim"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Standalone service actualCLI/V11factory/coupled nativeowner/headless epoch rejection 14CPUcontracts/SDK0; no listener/inference",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualCLI/sourcepins/nativeownerCPUrestart/changedepoch/SDK0")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(contracts=14,SDK0=True,models=0,requests=0)))

