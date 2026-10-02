from pathlib import Path
import json,sys,hashlib,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0078";state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
a=json.loads((r/"semantic_summary.json").read_text());assert a["transport_valid"]and a["semantic_cases"]==6 and len(a["requests"])==6
spec=json.loads((r/"controller_spec.json").read_text())
for x in spec["stages"][0]["sources"]:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]
for x in a["requests"]:
 for k in["body","wire"]:
  f=Path(x[k]["path"]);raw=f.read_bytes();assert len(raw)==x[k]["bytes"]and hashlib.sha256(raw).hexdigest()==x[k]["sha256"]
 b=json.loads(Path(x["body"]["path"]).read_text());v=json.loads(Path(x["wire"]["path"]).read_text());assert b["chat_template_kwargs"]["enable_thinking"]is False and not b["ignore_eos"]and b["cache_salt"].startswith(r.name)
 assert v["usage"]==x["usage"]and v["choices"][0]["message"].get("content")==x["content"]and x["semantic_pass"]==(x["content"].strip()==x["expected"])
assert a["semantic_pass_cases"]==sum(x["semantic_pass"]for x in a["requests"])
a.update(source_pins=len(spec["stages"][0]["sources"]),controller_dead=True,epochs_same_and_idle=True,models_started=0,model_signals=0)
atomic_json(j/"reduction.json",a);atomic_json(r/"reduction_brief.json",a);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="SameD77 semantic raw/source/epoch/usage audited: "+str(a["semantic_pass_cases"])+"/6 exactanswers; verdict="+a["verdict"],execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="sixactualnative semanticanswers/body/raw/nativeusage/epochs/pins")],unknowns=a["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(semantic_pass_cases=a["semantic_pass_cases"],verdict=a["verdict"])))
