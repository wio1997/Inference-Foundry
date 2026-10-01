import json,sys,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process
j=Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=Path(job["inputs"][0]["path"]);old=r.parent/"GLM-RUN-0036"
s=json.loads((old/"state.json").read_text());assert s["status"]=="failed" and s["failure_phase"]=="deploy" and s["completed_stages"]==[] and not same_process(s["owner"])

assert not list(old.glob("**/load_result.json")) and not list(old.glob("**/*.wire"))
spec=json.loads((r/"controller_spec.json").read_text());assert spec["run_id"]==r.name
import re
assert all(all(ident==r.name for ident in re.findall(r"GLM-RUN-[0-9]{4}",v)) for stage in spec["stages"] for v in stage["argv"])
assert [v["id"]for v in spec["stages"]]==["deploy","identity","cold","tools","capability","pilot","prefix","finalize"]
for pin in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
argv=["/usr/bin/python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/controller.py","start",str(r/"controller_spec.json")]
p=subprocess.run(argv,capture_output=True,timeout=30);(j/"launch.stdout").write_bytes(p.stdout);(j/"launch.stderr").write_bytes(p.stderr);assert p.returncode==0
state=json.loads((r/"state.json").read_text());assert state["status"]=="running" and same_process(state["owner"])
atomic_json(j/"launch_state.json",state)
m=json.loads((r/"manifest.json").read_text());m["status"]="running";m["controller_state_ref"]=str(r/"state.json");atomic_json(r/"manifest.json",m)
def ref(p,id):
 raw=p.read_bytes();return {"id":id,"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"immutable launch identity"}
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"running","summary":"New unique Run37 controller native DP4 TP8 DCP8 EP32 disjoint8device slices rebuild and bounded real E2E","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[{"pid":state["owner"]["pid"],"host":state["owner"]["host"],"readiness":"unready","evidence_ids":["state"],"boot_id":state["owner"]["boot_id"],"start_ticks":state["owner"]["start_ticks"]}]},"findings":[{"kind":"fact","text":"Run36deploy deadline/dead/zero inference; actualCPU4rank config/2host oldroot preflight before jointlycleanup; native4API readiness pending.","scope":{"run_id":r.name,"launch_only":True},"evidence_ids":["state","stdout"]}],"evidence":[ref(j/"launch_state.json","state"),ref(j/"launch.stdout","stdout")],"unknowns":["Native Worker metadata install/multirank semantics/capabilities/dynamic/prefix outcomes pending; no KEEP/stablecapacity"],"decision_request":None,"next_check_at":None})
print(json.dumps({"run":r.name,"owner":state["owner"],"stage":state["active_stage"]}))
