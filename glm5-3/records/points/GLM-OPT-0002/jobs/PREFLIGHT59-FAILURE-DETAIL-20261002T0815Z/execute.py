from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0059"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
audit=p/"jobs/REDUCE-PD59-NATIVE-20261002T0811Z-v2/reduction.json";assert ref(audit)["sha256"]=="4375a6959805a2eb19fffcf1a1f69df97f3e4c69e1870dd4f5c5e61baaa7a6cb"
a=json.loads(audit.read_text());assert a["state"]["status"]=="failed"and a["state"]["failure_phase"]=="deploy"and a["inference_attempts"]==0 and all(x["present"]for x in a["current_owners"].values())
events=json.loads((r/"deployment_events.json").read_text());last=events[-1];assert last["returncode"]==1 and 'assert os.environ.get("HCCL_BUFFSIZE")=="512"'in last["stderr_tail"]
assert not any(x["event"]in["native_role_started","exact_P58_stopped_D56_retained"]for x in events)
assert not(r/"startup_model_identities.json").exists()and not list(r.glob("CPU_fullCLI*.stdout"))
assert all(x["HCCL_BUFFSIZE"]=="512"for x in a["actual_API_HCCL_env"].values())
limits=["Only task CPU preflight asserted stale512 against intendedP416; nativeCPUconfig was built andACLfinally0; zero newworkers/models/inferences/signals","Original59source/spec/log frozen; correction requires new60 snapshot/controller, no vendor/native operator checks bypass","Auditor v1 incorrectly compared unstartedP59env to retainedP58; INVALID frozen, v2 binds actualARGV to prior58launch","58Pfailedcompute cohort stillowned/retainedD56; new60exactP58onlycleanup required, nooldqueue","P416/chunk256 candidate physicalfit/nativewindow/runtime/fullE2E pending; notGPU failure/capacity/KEEP"]
out=dict(at=utc(),run_id=r.name,verdict="INVALID",classification="GPT task CPU preflight hardcoded512MB rejected intendedP416 before cleanup/model start",audit=ref(audit),failing_event=dict(returncode=last["returncode"],stderr=last["stderr_tail"],stdout=last["stdout_tail"]),events=ref(r/"deployment_events.json"),source=ref(r/"api_config_probe.py"),signals=0,new_models=0,new_requests=0,limits=limits,next_candidate=dict(run_id="GLM-RUN-0060",P_HCCL_MB=416,D_HCCL_MB=512,P_batch=256,D_batch=1024,preflight="bind role-specific env and batch to actual planned CLI; native guards retained",actual_old_cohort="P58/D56"))
atomic_json(j/"reduction.json",out);b=json.loads((r/"reduction_brief.json").read_text());b.update(verdict="INVALID",measurement_valid=False,classification=out["classification"],CPU_preflight_failure_detail=ref(j/"reduction.json"),limits=limits);atomic_json(r/"reduction_brief.json",b)
m=json.loads((r/"manifest.json").read_text());m.update(verdict="INVALID",valid=False,CPU_preflight_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0059\n\nINVALID taskCPUpreflight stale512MB assertion rejectsintendedP416. Zero newmodel/requests/signals; actualpriorP58/D56 identities remain. 33pins audited; firstauditor wronglyexpects unstartedP59env againstP58 andINVALID, v2usesactualARGV/priorlaunchVALID. New60sameP416/chunk256/1GiB/GMU.46 withrole-specific preflight; D56retained, exactP58-only cleanup+journal/newpilot, nooldqueue. Not nativeGPU/config failure orcapacity verdict.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run59 INVALID stale512 CPUfixture before cleanup/models/requests;33pins actualP58/D56 audited; new60 role-specific P416/256 D512/1024 preflight",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="frozenfailedCPU source/event/nativeACLfinal/actualpriorARGV-env/auditv2",**ref(j/"reduction.json"))],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(ref(j/"reduction.json")))

