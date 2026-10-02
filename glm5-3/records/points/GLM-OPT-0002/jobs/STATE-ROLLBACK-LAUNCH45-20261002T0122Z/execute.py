import json,sys,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process
j=Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=Path(job["inputs"][0]["path"]);old=r.parent/"GLM-RUN-0044"
state=json.loads((old/"state.json").read_text());assert state["status"]=="failed"and not same_process(state["owner"])
a=json.loads((old/"reduction.json").read_text());assert a["diagnostic_evidence_valid"]and a["verdict"]=="REJECT"and a["inference_attempts"]==2 and a["model_signals"]==0
spec=json.loads((r/"controller_spec.json").read_text());assert [s["id"]for s in spec["stages"]]==["deploy","prepare","tools","responses","finalize"]
for pin in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"] and Path(pin["path"]).read_bytes()==Path(pin["snapshot"]).read_bytes()
p=subprocess.run(["/usr/bin/python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/controller.py","start",str(r/"controller_spec.json")],capture_output=True,timeout=30);(j/"launch.stdout").write_bytes(p.stdout);(j/"launch.stderr").write_bytes(p.stderr);p.check_returncode()
s=json.loads((r/"state.json").read_text());assert s["status"]=="running"and same_process(s["owner"]);atomic_json(j/"launch_state.json",s)
m=json.loads((r/"manifest.json").read_text());m.update(status="running",controller_state_ref=str(r/"state.json"));atomic_json(r/"manifest.json",m)
def ref(path):
 raw=path.read_bytes();return {"id":path.name,"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"actual immutable uniquecontroller launch"}
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"running","summary":"Run45 uniquecontroller started after audited44native toolregression REJECT: configured reduce-samplingFalse/STORE1 rebuild then4tool rollbackcontrol andactual nativeResponsesstate/restart/backgroundcontracts","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[{"host":s["owner"]["host"],"pid":s["owner"]["pid"],"boot_id":s["owner"]["boot_id"],"start_ticks":s["owner"]["start_ticks"],"readiness":"unready","evidence_ids":["launch_state.json"]}]},"findings":[{"kind":"fact","text":"Run44failed/dead2attempts audited, all23prepared45sourcepins match; exactnative44idle/currentowner/jointcleanup runsindeploy before45start. No oldqueue/operatorchanges.","scope":{"launch_only":True},"evidence_ids":["launch_state.json"]}],"evidence":[ref(j/"launch_state.json"),ref(j/"launch.stdout")],"unknowns":["Newnativeepochfit/4tool/nativeenabledstate/logprob/background/restartE2E pending","No performance/KEEP/capacity claim"],"decision_request":None,"next_check_at":None})
print(json.dumps({"controller":s["owner"],"spec":s["spec_sha256"],"actual":"launch_only"}))
