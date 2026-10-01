import json,sys,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process
j=Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=Path(job["inputs"][0]["path"]);old=r.parent/"GLM-RUN-0038"
state=json.loads((old/"state.json").read_text());assert state["status"]=="completed" and not same_process(state["owner"])
audit=json.loads((old/"reduction.json").read_text());assert audit["measurement_valid"] and audit["functional_acceptance"] and audit["new_completed_inference_requests"]==37
spec=json.loads((r/"controller_spec.json").read_text());assert [x["id"]for x in spec["stages"]]==["prepare","capability","restart","finalize"]
for pin in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
p=subprocess.run(["/usr/bin/python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/controller.py","start",str(r/"controller_spec.json")],capture_output=True,timeout=30)
(j/"launch.stdout").write_bytes(p.stdout);(j/"launch.stderr").write_bytes(p.stderr);p.check_returncode()
state=json.loads((r/"state.json").read_text());assert state["status"]=="running" and same_process(state["owner"]);atomic_json(j/"launch_state.json",state)
m=json.loads((r/"manifest.json").read_text());m.update(status="running",controller_state_ref=str(r/"state.json"));atomic_json(r/"manifest.json",m)
def ref(path,key):
 raw=path.read_bytes();return {"id":key,"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"immutable newcontroller launch identity"}
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"running","summary":"New unique Run39 controller for persistentgateway healthygroup capability and cleanrestart, exactRun38cohort validation pendingprepare; no native reload","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[{"pid":state["owner"]["pid"],"host":state["owner"]["host"],"readiness":"unready","evidence_ids":["state"],"boot_id":state["owner"]["boot_id"],"start_ticks":state["owner"]["start_ticks"]}]},"findings":[{"kind":"fact","text":"Run38completed/dead and independent all39nativeattempts auditvalid; newcontrolspec/sourcepins verified. Nativeowners/idle/epoch retained assertions run inprepare before gateway admission.","scope":{"launch_only":True},"evidence_ids":["state"]}],"evidence":[ref(j/"launch_state.json","state"),ref(j/"launch.stdout","stdout")],"unknowns":["Persistent native E2E pending; CPU42failure and uncleanjournal tests separate; no nativefault/recovery/capacity/KEEP"],"decision_request":None,"next_check_at":None})
print(json.dumps({"run":r.name,"owner":state["owner"],"stage":state["active_stage"]}))
