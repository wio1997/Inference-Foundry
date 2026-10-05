from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0209"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
audit=p/"jobs/AUDIT-RUN209-20261005TPOSTLATE/reduction.json";v=json.loads(audit.read_text());assert v["measurement_valid"]and v["effective_outputs"]==14208
samples=[json.loads(l)for l in(r/"native_samples.jsonl").read_text().splitlines()];rows=json.loads((r/"dynamic_summary.json").read_text())["requests"]
def empty_busy(s):
 a,b=s["native"]["D0"],s["native"]["D1"];public={x["id"]:x for x in s["placement"]["replicas"]}
 return a["num_requests_running"]>0and b["num_requests_running"]==b["num_requests_waiting"]==b["kv_cache_usage_perc"]==0and public["D1"]["active_requests"]==public["D1"]["prefilling_requests"]==0and not public["D1"]["draining"]and not public["D1"]["group_faulted"]and not public["D1"]["temporarily_unhealthy"]
observed=[s for s in samples if empty_busy(s)];late=[]
for row in rows:
 if not row["id"].startswith("late"):continue
 t=row["actual_dispatch_s"];left=max([s for s in samples if s["elapsed_s"]<=t],key=lambda s:s["elapsed_s"]);right=min([s for s in samples if s["elapsed_s"]>=t],key=lambda s:s["elapsed_s"])
 assert empty_busy(left)and empty_busy(right);late.append(dict(id=row["id"],dispatch_s=t,native_owner=row["native_owner"],observed_TTFT_s=row["ttft_s"],observed_output_interval_s=row["mean_after_first_output_s"],surrounding_samples=[dict(elapsed_s=s["elapsed_s"],native=s["native"],public_D1=next(x for x in s["placement"]["replicas"]if x["id"]=="D1"))for s in[left,right]]))
out=dict(at=datetime.now(timezone.utc).isoformat(),valid=True,source=ref(r/"native_samples.jsonl"),audit=ref(audit),n_samples=len(samples),idle_peer_busy_D0_samples=len(observed),sample_range_s=[observed[0]["elapsed_s"],observed[-1]["elapsed_s"]],late=late,Current=None,limits=["Eachobserved sample hasownedD1nativeRunning-Waiting-KV0/publiclease0 versusD0Running>0; samples do not certifycontinuousGPUidle/hardwareutilization", "No newrequests/models/policies/publicedits; original209fullcontractvalid, late16 arrivalworkload differs original12", "Lease-empty shortspill prototype at currenttwoheterogeneous engines needsrealcandidateE2E; nativebackground/cancel tails notinferable fromleasealone"])
f=j/"reduction.json";f.write_text(json.dumps(out,ensure_ascii=False,indent=2)+chr(10));result=dict(schema_version=1,job_id=j.name,status="completed",summary="Run209 fourlatearrivals bracketedbyownedD1Running-Waiting-KV0/publiclease0 whileD0busy; sourceaudit valid, sampledopportunity only",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(f),locator="fourlatearrivals surroundingnative-publicpressure samples")],unknowns=out["limits"],decision_request=None,next_check_at=None);(j/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10));print(json.dumps(dict(evidence=ref(f),samples=len(observed))))
