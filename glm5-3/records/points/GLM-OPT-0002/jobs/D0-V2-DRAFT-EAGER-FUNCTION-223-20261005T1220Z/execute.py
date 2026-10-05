from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process
j=Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=Path(job["inputs"][0]["path"]);s=json.loads((r.parent/"GLM-RUN-0222/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
spec=json.loads((r/"controller_spec.json").read_text());assert all(st["id"].isalnum()for st in spec["stages"])
for x in spec["stages"][0]["sources"]:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]
z=subprocess.run(["/usr/bin/python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/controller.py","start",str(r/"controller_spec.json")],capture_output=True,timeout=30);(j/"launch.stdout").write_bytes(z.stdout);(j/"launch.stderr").write_bytes(z.stderr);z.check_returncode()
s=json.loads((r/"state.json").read_text());assert s["status"]=="running"and same_process(s["owner"]);atomic_json(j/"launch_state.json",s)
b=(j/"launch_state.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="running",summary="Unique223 resident222 draftNONE targetFULL multistepfunction controller started; actualE2E pending",execution=dict(inner_exit_code=0,acceptance="passed",processes=[dict(host=s["owner"]["host"],pid=s["owner"]["pid"],boot_id=s["owner"]["boot_id"],start_ticks=s["owner"]["start_ticks"],readiness="unready",evidence_ids=["launch_state"])]),findings=[],evidence=[dict(id="launch_state",path=str(j/"launch_state.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualuniquecontroller")],unknowns=["D0 V2-K2sameGraph3n prototypefit/capture8/SDK/fullrank-wire-typedSTORE/retainedD1; fullrequestequivalenceunknown pending; noKEEP/capacity"],decision_request=None,next_check_at=None));print(json.dumps(s))
