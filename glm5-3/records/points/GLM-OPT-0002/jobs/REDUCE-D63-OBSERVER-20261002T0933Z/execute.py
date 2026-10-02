from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0063";raw=(r/"full_D0.wire").read_bytes()
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and s["failure_phase"]=="pilot"
for p in json.loads((r/"controller_spec.json").read_text())["stages"][0]["sources"]:assert hashlib.sha256(Path(p["snapshot"]).read_bytes()).hexdigest()==p["sha256"]
frames=[]
for b in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
 d=b"\n".join(l[5:].removeprefix(b" ")for l in b.splitlines()if l.startswith(b"data:"))
 if d and d!=b"[DONE]":frames.append(json.loads(d))
assert all(not x.get("error")for x in frames)
small=NativeSSEObserver(collect_contract=True);small.feed(raw);big=NativeSSEObserver(max_bytes=2097152,collect_contract=True)
for n in range(0,len(raw),317):big.feed(raw[n:n+317])
c=big.contract();assert c["done"]and not c["unknown"]and not c["native_error"]and c["finish_reasons"]=={"0":"length"}
assert c["usage"]==dict(prompt_tokens=81932,total_tokens=81996,completion_tokens=64)
prompts=[x["prompt_token_ids"]for x in frames if isinstance(x.get("prompt_token_ids"),list)];ids=[t for x in frames for ch in x.get("choices",[])for t in ch.get("token_ids")or[]]
assert len(prompts)==1 and len(prompts[0])==81932 and len(ids)==64 and all(type(x)is int for x in prompts[0]+ids)
cleanup={}
for node in["166","167"]:
 a=json.loads((r/("cleanup_"+node+".json")).read_text());assert a[-1]["D_every_identity_retained"]and all(a[-1][k]==0 for k in["signals_to_D","signals_to_unattributed","signals_to_inactive_native"])
 cleanup[node]=dict(D_retained=len(a[0]["retained_D_targets"]),P_was_active=a[-1]["P_was_active"],sent=[v for v in a if v.get("event")=="signal_sent"])
assert sum(x["D_retained"]for x in cleanup.values())==647
out=dict(at=utc(),run_status="failed/pilot",verdict="INVALID",failure_kind="diagnosticobserver default64KiB discards requested prompt-idframe",source_pins=30,old_default_contract=small.contract(),bounded2MiB_contract=c,original_native_response_valid=True,audited_completed_requests=1,audited_committed_native_output_tokens=64,effective_public_output_tokens=64,token_id_count=64,prompt_id_count=81932,wire=dict(path=str(r/"full_D0.wire"),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),cleanup=cleanup,TTFT_s=None,wall_s=None,limits=["FreshD0 full81932→64 nativecompletion proves one finite D-only cold-salt window only; nativecoldcounters/postidle/realTTFT notfullycaptured","Originalfailedattemptledger preserved withzero automatedcredit; retrospective raw token audit explicitlyseparate","SameD56all647physicalidentities retained; P61stopped,resourcecomposition changes, no isolatedgain/capacity/KEEP","2MiB observer onlydiagnosticcontract opted toreturnpromptids; defaultboundedforwardingtelemetryunchanged"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run63 INVALID boundedobserver fixture, actualraw D0 full81932→64 completion audited; exactP61cleanup/D56all647retained",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="originalnativewire/source/cleanup")],unknowns=out["limits"],decision_request=None,next_check_at=None))
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",verdict="INVALID",valid=False,posthoc_audited_native_outputs=64,posthoc_reduction=str(j/"reduction.json"),failure_kind=out["failure_kind"]);atomic_json(r/"manifest.json",m);(r/"summary.md").write_text("# "+r.name+"\n\nINVALID observerfixture: requested prompt-ID frame exceeds64KiB defaulttelemetry bound. Raw nativeSSE audit withbounded2MiB parser yields81932prompt/64commit IDs/length/DONE/noerror,one finiteD0 completion; TTFT/wall/coldpostcounter unknown. Both exactP61 APIroots receivedoneSIGTERM each; D56all647/NPU32 retained, noD/unknown/Zsignals. No secondD/window/performance/capacity/KEEP claim.\n");print(json.dumps(out))

