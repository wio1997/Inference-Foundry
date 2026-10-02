from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0055";a=p/"jobs/REDUCE-PD55-NATIVE-20261002T0611Z"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
assert ref(a/"reduction.json")["sha256"]=="4685a926755b97ad58986aee9bada341fc50bec94d00ecf19d73c8e0cd9e288f"
v=json.loads((a/"reduction.json").read_text());assert v["measurement_valid"]and v["state"]["status"]=="failed"and v["source_pins"]==20 and v["inference_attempts"]==0
owners=json.loads((r/"startup_model_identities.json").read_text());assert set(owners)=={"P0","P1"}
events=json.loads((r/"deployment_events.json").read_text());assert not any(x.get("event")=="native_role_started"and x.get("role")=="D"for x in events)
native={}
for key in owners:
 f=a/(key+".native.log");ls=f.read_text().splitlines();hits=[(i,l)for i,l in enumerate(ls)if"NEEDED_HCCL_BUFFSIZE_HIERARCHY"in l];assert hits
 i,line=hits[0];prefix=line.split(" ERROR ")[0];stack=[l for l in ls[max(0,i-120):i+15]if l.startswith(prefix)]
 assert "maxBs = 1, h = 6144"in line and "= 408MB, HCCL_BUFFSIZE=256MB"in line
 assert not any("torch.OutOfMemoryError:"in l for l in ls)
 native[key]=dict(raw=ref(f),first_window_rejection=line,first_stack=stack,torch_OOM_count=0,actual_launch_env=json.loads((r/("HCCL_env_"+key+".json")).read_text()))
limits=["408MB is observed native hierarchy minimum for maxBs1/h6144, not a bound for all batch sizes/requests","No memory OOM in frozen P logs; this rejection identifies communication window insufficiency, not physical capacity","No D model launch, Graph capture, inference, transfer or public output; no functional/capacity credit","512MB preserves native hierarchy operator/ordinary PG options; full startup and actual E2E remain unproved"]
out=dict(at=utc(),run_id=r.name,verdict="REJECT",classification="Native hierarchy MoE dispatch tiling rejects HCCL_BUFFSIZE256MB: observed conditional required408MB for maxBs1/h6144",terminal_audit=ref(a/"reduction.json"),P_native=native,D_started=False,current_API_owners=v["current_owners"],source_pins=20,inference_attempts=0,outputs=0,signals=0,new_models=0,new_requests=0,limits=limits,next_candidate=dict(run_id="GLM-RUN-0056",HCCL_BUFFSIZE_MB=512,same_native_operators=True,ordinary_PG_options_unchanged=True))
atomic_json(j/"reduction.json",out)
b=json.loads((r/"reduction_brief.json").read_text());b.update(verdict="REJECT",classification=out["classification"],hierarchy_window_failure_detail=ref(j/"reduction.json"),limits=limits);atomic_json(r/"reduction_brief.json",b)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="REJECT",hierarchy_window_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0055\n\nREJECT native hierarchy MoE dispatch tiling: both P ranks firstreject06:05:30, required408MB atmaxBs1/h6144 vs HCCL_BUFFSIZE256MB. No torch OOM in frozen P logs, no D launch/Graph/inference/output. Terminal20pin auditVALID, P API roots absent; allworker attribution checked before nextcohort. Nativewindow conditional bound only, not hardware/allworkload bound. Next56native512MB, ordinaryPG200/DPformula/operators/samePD geometry preserved.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run55 REJECT hierarchy communicationwindow256<408MB atmaxBs1/h6144; bothP/noOOM/noD/0requests; next512native candidate",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="frozenfirstnative tilingstack/actualenv/noD",**ref(j/"reduction.json"))],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(ref(j/"reduction.json")))

