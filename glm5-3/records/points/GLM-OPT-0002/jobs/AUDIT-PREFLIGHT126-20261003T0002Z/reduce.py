from pathlib import Path
import json,hashlib,sys,subprocess,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0126";old=p/"runs/GLM-RUN-0125"
assert not(r/"state.json").exists()and not(r/"public_retire.json").exists()
log=(r/"controller.log").read_text();assert "ValueError: invalid stage id"in log
spec=json.loads((r/"controller_spec.json").read_text());assert spec["stages"][0]["id"]=="retire_public"and not spec["stages"][0]["id"].isalnum()
for v in spec["stages"][0]["sources"]:assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
state=json.loads((old/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
proof=json.loads((p/"jobs/AUDIT-RUN125-20261002T2352Z-v2/public_service_proof.json").read_text())
assert same_process(proof["host_identity"])
argv=proof["container_owner"]["argv"];pid=proof["host_identity"]["pid"];assert[v.decode()for v in Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(bytes([0]))if v]==argv
out=dict(at=utc(),verdict="INVALID",reason="controller rejects nonalphanumeric stageid retire_public before publishingowner/executing phases",phases_executed=0,model_operations=0,prior_public125_current=True,current_owner=proof["host_identity"],controller_log=dict(path=str(r/"controller.log"),sha256=hashlib.sha256((r/"controller.log").read_bytes()).hexdigest()),source_pins=len(spec["stages"][0]["sources"]),limits=["No workload/nativeinference was executed; no capacity credit"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INVALID",results=out);atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0126\n\nINVALID before controller owner publication: nonalphanumeric stage id retire_public. No phases, public retirement or modeloperations. Existing125public remains. Preserve originalspec/log; correctedfreshRun127 required.\n")
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="126controllerpreflightINVALID nonalphanumericstageid/zerophases/noresourceoperations/current125public unchanged",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="controlleroriginalmalformedspec/current125frontend/noownerpublication")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(out))
