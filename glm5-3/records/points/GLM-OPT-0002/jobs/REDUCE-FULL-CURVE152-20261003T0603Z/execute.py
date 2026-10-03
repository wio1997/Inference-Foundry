from pathlib import Path
import json,hashlib,statistics,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0152";m=p/"jobs/FULL-CURVE152-20261003T0539Z"
s=json.loads((r/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-RUN152V2-20261003T0550Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="8afc89d4dd68c8d9039a863bfd59935c6e5c3362b0235acf363092326a9324b6";a=json.loads(f.read_text());assert a["measurement_valid"]and a["complete_outputs"]==245760
rows=[json.loads(x)for x in(m/"samples.jsonl").read_text().splitlines()];assert rows and not rows[-1]["controller_alive"]and all(not x["errors"]for x in rows)
raw_index=[]
for row in rows:
 for key in ["D0","D1"]:
  info=row["nodes"][key]["raw"];b=Path(info["path"]).read_bytes();assert len(b)==info["bytes"]and hashlib.sha256(b).hexdigest()==info["sha256"];raw_index.append(info)
assert all(rows[-1]["nodes"][k]["full_generation_so_far"]==122880for k in["D0","D1"])
windows=[]
for before,after in zip(rows,rows[1:]):
 dt=after["monotonic_s"]-before["monotonic_s"];assert dt>0
 if all(before["nodes"][k]["metrics"]["num_requests_running"]==2and after["nodes"][k]["metrics"]["num_requests_running"]==2for k in["D0","D1"]):
  rates={k:(after["nodes"][k]["full_generation_so_far"]-before["nodes"][k]["full_generation_so_far"])/dt for k in["D0","D1"]}
  assert all(abs(rates[k]-after["native_committed_tps"][k])<1e-8for k in rates)
  windows.append(dict(start=before["at"],end=after["at"],seconds=dt,tokens=sum(rates.values())*dt,total_TPS=sum(rates.values()),perdomain_TPS=rates,contexts={k:[before["nodes"][k]["mean_context_tokens_estimate"],after["nodes"][k]["mean_context_tokens_estimate"]]for k in rates}))
assert windows
weighted=sum(w["tokens"]for w in windows)/sum(w["seconds"]for w in windows)
out=dict(at=utc(),kind="readonly_native_counter_curve",run_id=r.name,measurement_valid=True,source_samples=len(rows),same_run_full_E2E_reused=True,full_output_credit_reused=245760,new_inference=0,model_operations=0,balanced_two_running_windows=len(windows),windows=windows,balanced_window_total_TPS_weighted=weighted,total_TPS_min=min(x["total_TPS"]for x in windows),total_TPS_max=max(x["total_TPS"]for x in windows),context_range={k:[min(x["contexts"][k][0]for x in windows),max(x["contexts"][k][1]for x in windows)]for k in["D0","D1"]},terminal_full_generation={k:rows[-1]["nodes"][k]["full_generation_so_far"]for k in["D0","D1"]},limits=["Observer started during full decode; does not cover early context bins","Context is estimate from domain total/actual2requests, individual sequences differ","Committed metric intervalTPS differs fullCLI finiteE2E316.5465TPS, no overhead subtraction/no stablecapacity/globalhardwarebound","At most4fullrequests, not robust tail latency or repeatKEEP"])
atomic_json(j/"raw_index.json",raw_index);atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="VALID readonly152 high-context committed countercurve; no newinference/outputcredit",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="curve",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="balancednative2runningwindows")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps({k:v for k,v in out.items()if k not in["windows","limits"]}))
