from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0056"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
audit=p/"jobs/REDUCE-PD56-NATIVE-20261002T0639Z/reduction.json";assert ref(audit)["sha256"]=="128201966aee1463248f7aa5da3f20295f37c313ca4309e983941427a3b17891"
a=json.loads(audit.read_text());assert a["measurement_valid"]and a["state"]["status"]=="failed"and a["state"]["failure_phase"]=="pilot"and a["inference_attempts"]==3 and a["completed_native_requests"]==2 and a["effective_public_output_tokens"]==64 and all(x["present"]for x in a["current_owners"].values())
cpu=p/"jobs/P56-KV-ADMISSION-CPU-20261002T0655Z-v2/reduction.json";assert ref(cpu)["sha256"]=="41a55680b3975a68245cbea8aebdb5b7ed9eba5628e18e469cd1c4df91f24db6";c=json.loads(cpu.read_text())
metrics=p/"jobs/PD56-ACTIVE-METRICS-20261002T0644Z/reduction.json";assert ref(metrics)["sha256"]=="3f9629234e5747d77eb2d0180dd8ca8529aef9e843d6041f53a365c4d7ae9e63"
m=json.loads(metrics.read_text());P=m["native_metrics"]["P0"]["selected"];assert any('num_requests_waiting_by_reason{'in l and'reason="capacity"'in l and l.endswith(" 1.0")for l in P)
counts={};graph={}
for key in["P0","P1","D0","D1"]:
 f=audit.parent/(key+".native.log");ls=f.read_text().splitlines();skips=[json.loads(l.split("GLM_UNUSED_SFA_WORKSPACE_SKIPPED ",1)[1])for l in ls if"GLM_UNUSED_SFA_WORKSPACE_SKIPPED "in l]
 assert len(skips)==32 and len({x["pid"]for x in skips})==16 and all(x["actual_elements"]==0 for x in skips)
 assert not any("torch.OutOfMemoryError:"in l or"GLM_UNUSED_SFA_WORKSPACE_READ:"in l for l in ls)
 counts[key]=dict(worker_count=16,zero_builders=32,native=ref(f))
 if key.startswith("D"):
  graph[key]=[l for l in ls if"Graph capturing finished"in l];assert graph[key]
limits=["NativeCPUactualspec/manager/41vs42positive-negative reproduces emptyP fullsequenceadmission failure; actualscheduler BlockPool counter not directly exported","Run56 64workers/128zeroSFA builders andbothD K5FULLGraph ready/small32outputs each; notfullPD/fallback/stability/capacity","P firstlonghelper timeout haszero wire/output/processedprompt; no KVtransfer attempt or capacitycredit, no oldqueue replay","NativeGPU KVcache capacity81933/1.00x planner display doesnotaccount nullblock admission; physicalHBM or operator failure not claimed","NextP1GiB nativeplanner46blocks45free is CPUcandidate; GPU P reload whileD retained/new E2E required","RegisteredP IDs include native UUID/DPsuffix; exactnative handshake/rootworker identity supersedesCLIseed equality, do not rewritemetadata"]
out=dict(at=utc(),run_id=r.name,verdict="REJECT",classification="Functionalpilot fails beforefirstP compute: emptyP capacitywait with41pool/40free, canonical81931/32 needs41effective2048-token blocks; nativeCPUadmission reproduces",terminal_audit=ref(audit),admission_CPU=ref(cpu),active_metrics=ref(metrics),workspace_counts=counts,D_Graph_completion=graph,actual_inferences=3,completed_native_requests=2,effective_public_outputs=64,failed_P_credit=0,signals=0,new_inference=0,models=0,limits=limits,next_candidate=dict(run_id="GLM-RUN-0057",P_KV_bytes=1073741824,native_planned_P_blocks=c["plans"]["1073741824"]["num_blocks"],retain_D_existing_API_Graph=True,HCCL_MB=512,native_operators_unchanged=True))
atomic_json(j/"reduction.json",out);b=json.loads((r/"reduction_brief.json").read_text());b.update(verdict="REJECT",classification=out["classification"],admission_failure_detail=ref(j/"reduction.json"),limits=limits);atomic_json(r/"reduction_brief.json",b)
manifest=json.loads((r/"manifest.json").read_text());manifest.update(verdict="REJECT",valid=False,admission_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",manifest)
(r/"summary.md").write_text("# GLM-RUN-0056\n\nREJECT pilot: P first81932-token helper nevercomputes, capacitywait atemptyKV pool then240s timeout/zero wire. NativeP pool41/null1/free40, DCP16 effectiveblock2048, input81931/32 needs41blocks; nativeactualspec/KVManager CPUpositive-negative reproduces41fail42pass, P1GiBplans46blocks. BothP/D64workers ready,128SFA builders0, bothD K5FULLGraph complete,2Dlocal requests each32output=64effective tokens retained. NoOOM/poison/PDtransfer/fullinputfallback/capacityproof. Actualnative registeredPIDs use UUID/DPrank suffix andmustnot be replaced byCLIseed. Next57 P1GiB/reloadP-only/retainverifiedidleD/same512PGpolicy/nativeoperators, newrequests only.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run56 REJECT emptyP nativeKVadmission41/40free vsrequired41;64workers/128zeroSFA/bothDGraphready/2Dlocal64outputs; nextP1GiBretainD",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="nativephase/metrics/actualspec BlockPool KVManager CPUcounterexample/Graph/rawtokens",**ref(j/"reduction.json"))],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(ref(j/"reduction.json")))

