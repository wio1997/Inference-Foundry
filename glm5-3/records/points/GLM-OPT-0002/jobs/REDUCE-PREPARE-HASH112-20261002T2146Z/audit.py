from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0112";s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])and s["completed_stages"]==[]
a=json.loads((r/"prepare.phase.json").read_text());assert a["exit_code"]==1 and not a["timed_out"]
text=(r/"prepare.log").read_text();assert "AssertionError"in text and "hashlib.sha256(f.read_bytes())"in text
assert not(r/"epoch_prepare.stdout").exists()and not(r/"native_dynamic.stdout").exists()and not(r/"arrival_plan.json").exists()
sp=json.loads((r/"controller_spec.json").read_text())
for x in sp["stages"][0]["sources"]:
 b=Path(x["path"]).read_bytes();assert b==Path(x["snapshot"]).read_bytes()and hashlib.sha256(b).hexdigest()==x["sha256"]
f=r.parents[1]/"jobs/NATIVE-PD-GEOMETRY-CPU-20261002T2124Z-v2/reduction.json";actual=hashlib.sha256(f.read_bytes()).hexdigest();bad="2e93975a6fc642b087675cbc73306b4e4ec485491a43dd8995a065e0e5adb17"
assert bad in(r/"prepare.py").read_text()and actual=="2e93975a6fc642b087675cbc73306b4e4eec485491a43dd8995a065e0e5adb17"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
out=dict(at=utc(),verdict="INVALID",functional_acceptance=False,effective_public_output_tokens=0,models_started=0,inference_launched=False,state=s,phase=a,source_pins=len(sp["stages"][0]["sources"]),failure="Handcopied CPU22 evidence SHA omitted one e; precondition rejects before epochcheck/dynamic/inference",expected_in_script=bad,actual_cpu22_sha=actual,artifacts=[ref(r/n)for n in["prepare.py","prepare.log","controller_spec.json","state.json"]],limits=["No inference or performance evidence from112; launch bridge only proved detached controller start","Do not rewrite executed source/raw;113 is fresh candidate with verified hash; no actualSDK invoked by112"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run112 INVALID zeroinference: CPU22 expectedhash transcription mismatch, source/raw frozen; readonly diagnosis valid",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="source32/phase/log/zeroexecution")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(verdict="INVALID",output=0,actual_cpu22_sha=actual)))
