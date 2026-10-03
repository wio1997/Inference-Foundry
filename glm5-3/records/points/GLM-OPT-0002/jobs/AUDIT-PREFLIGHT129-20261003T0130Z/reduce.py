from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0129"
state=json.loads((r/"state.json").read_text());assert state["status"]=="failed"and state["failure_phase"]=="prepare"and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for v in spec["stages"][0]["sources"]:
 assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
assert not(r/"deploy.phase.json").exists()and not(r/"startup_root_identities.json").exists()and not(r/"retire_public.json").exists()
raw=(r/"CPU_config_166.stdout").read_text();err=(r/"CPU_config_166.stderr").read_text()
assert "ModuleNotFoundError: No module named 'coupled_atomic_mq_worker'"in err
acks=[json.loads(x)for x in raw.splitlines()if x.startswith('{"event":')]
assert len(acks)==2and acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
out=dict(at=utc(),run_id=r.name,valid=False,verdict="INVALID",source_pins=len(spec["stages"][0]["sources"]),failure="CPU probe import dependency absent from copied task plugin",model_operations=0,native_signals=0,inference_requests=0,public_retirement_not_executed=True,SDK_init_final0=True,raw=[ref(r/n)for n in["CPU_config_166.stdout","CPU_config_166.stderr","prepare.phase.json","prepare.log"]],limits=["Native coupled32NPU feasibility/performance not measured; this is a preflight fixture failure","Current ownership must use subsequent130 actualstate, not129failed controller/old128services"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INVALID",results=out);atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nINVALID before anyGPU/model/frontend operation: CPU probe imported coupled_atomic_mq_worker absent from task plugin copy list. NativeSDKinit-final0; original84sources/failedphase/rawlogs preserved. New130 explicitly adds dependency and performs fresh preflight; no oldqueue replay.\n")
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="129INVALID CPU missingimport before model/frontend signals/inference; SDK0/rawpreserved",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="source84/rawCPUfailure/SDK0/zero modelops")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print("129 preflight INVALID recorded")
