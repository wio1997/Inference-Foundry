from pathlib import Path
import sys,json,hashlib,subprocess,shlex,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1]
for v in json.loads((j/"source_pins.json").read_text()):
 assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
s=json.loads((p/"runs/GLM-RUN-0125/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
assert not re.search(r":8002\s",subprocess.check_output(["ss","-ltnp"],text=True))
shell="export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; python3 "+shlex.quote(str(j/"cpu_probe.py"))+" "+shlex.quote(str(j))
args=["docker","exec","glm52-single","bash","-c",shell];atomic_json(j/"command.json",dict(argv=args,at=utc()))
z=subprocess.run(args,capture_output=True,timeout=150);(j/"CPU.stdout").write_bytes(z.stdout);(j/"CPU.stderr").write_bytes(z.stderr);z.check_returncode()
assert not re.search(r":8002\s",subprocess.check_output(["ss","-ltnp"],text=True))
proof=json.loads((j/"CPU_proof.json").read_text());atomic_json(j/"reduction.json",dict(at=utc(),valid=True,proof=proof,source_pins=json.loads((j/"source_pins.json").read_text())));b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Actualservice_entry privatefrontend SIGTERM/nativeACL init-final0/exit0/journalclosed/currentpublic125models untouched; zero inference",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualownedchild/privateCLI/SIGTERM/SDK0/journal/nativecountersunchanged")],unknowns=proof["limits"],decision_request=None,next_check_at=None))
print(json.dumps(proof))
