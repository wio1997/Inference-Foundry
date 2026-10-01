import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0041";s=json.loads((r/"state.json").read_text());events=json.loads((r/"deployment_events.json").read_text());phase=json.loads((r/"deploy.phase.json").read_text())
assert s["status"]=="failed" and not same_process(s["owner"]) and s["completed_stages"]==[]
assert not any(e["event"]=="physical_node_started"for e in events)
assert "ModuleNotFoundError: No module named 'acl'"in events[-1]["stderr"]
assert not list(r.glob("**/load_result.json"))
spec=json.loads((r/"controller_spec.json").read_text())
for x in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]
def ref(p):
 b=p.read_bytes();return {"path":str(p),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
out={"run_id":r.name,"at":utc(),"measurement_valid":False,"diagnostic_evidence_valid":True,"verdict":"INVALID","inference_attempts":0,"model_signals":0,"new_native_engines":0,"state":s,"deploy_phase":phase,"sourcepins_verified":len(spec["stages"][0]["sources"]),"last_command":events[-1],"root_cause":"CPU preflight set PYTHONPATH to task scripts only without sourcing native CANN environment; importing NPUWorker requires acl module path from common environment. Native launch config already sources common env; startup was never executed.","remedy":"New Run preserves native CANN environment and appends task scripts in preflight and native launch, no oldqueue replay"}
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction.json",out);atomic_json(r/"reduction_brief.json",out)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INVALID",diagnostic_evidence_valid=True,completed_at=phase["finished_at"],new_client_attempts=0,new_completed_inference_requests=0,new_effective_output_tokens=0,reduction=ref(r/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nINVALID CPU preflight before any native start/signal/inference: PYTHONPATH replaced CANN acl path, NPUWorker import failed. CPU EngineArgs config accepted; atomic MessageQueue contract independently valid. Native config already sources common env, but native launch never reached. Exact frozen sources/spec preserved. New Run will source CANN env and append task scripts during preflight; no replay.\n")
job=json.loads((j/"job.json").read_text());atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Run41 INVALID preflight environment, zero native start/signal/inference, exact source audit retained","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[{"id":"reduction",**ref(j/"reduction.json"),"locator":"exact CPU preflight failure/no physical_node_started/source hashes"}],"unknowns":["Crosshost startup and native model behavior still untested"],"decision_request":None,"next_check_at":None});print(json.dumps({"verdict":"INVALID","model_signals":0,"inference_attempts":0}))
