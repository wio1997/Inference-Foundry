import json,subprocess,hashlib,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;g=j.parents[4]
sources=[g/"runtime/response_affinity.py",g/"runtime/response_affinity_gateway.py",g/"tests/test_response_affinity.py"]
pins=[{"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}for p in sources]
native_paths=["/vllm-workspace/vllm/vllm/entrypoints/openai/responses/serving.py","/vllm-workspace/vllm/vllm/entrypoints/openai/responses/api_router.py","/vllm-workspace/vllm/vllm/entrypoints/openai/responses/protocol.py"]
native=[]
for path in native_paths:
 raw=subprocess.check_output(["docker","exec","glm52-single","cat",path],timeout=30);(j/Path(path).name).write_bytes(raw);native.append({"path":path,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
p=subprocess.run(["docker","exec","-e","PYTHONPATH="+str(g/"runtime"),"glm52-single","python3","-m","unittest","discover","-s",str(g/"tests"),"-p","test_response_affinity.py","-v"],capture_output=True,timeout=120)
(j/"test.stdout").write_bytes(p.stdout);(j/"test.stderr").write_bytes(p.stderr);p.check_returncode()
assert "Ran 6 tests"in p.stderr.decode()and "OK"in p.stderr.decode()
out={"at":utc(),"cpu_only":True,"model_calls":0,"weights_loaded":0,"tests_passed":6,"sourcepins":pins,"native_source_index":native,"fixture_native_source_run":"GLM-RUN-0039 actual ResponsesJSON/SSE bytes, CPU transport/store simulation only","contracts":["Native request bodies/query/duplicate headers/output SSE bytes preserved","JSON/SSE owner registered before nativeID delivered; split gzip typedSSE actual39 fixture","GET/cancel/previous_response_id/caller request_id route to exact known native owner, no duplicate request reroute","Clean gateway restart owner restore and sameepoch logicaldrain/readd preserve owner","Native group fault/changedphysicalepoch forbid crossowner state replay; unknown IDs retain native404","600 atomic journal singlewriter/corruption/replaceError conservative behavior","Default disabled state feature retains stateless transparent gateway behavior"],"limits":["No native enabled-store/background/continuation E2E or model work; CPU transport cannot establish native frontend behavior","Background native jobs outlive HTTP leases; background capacity must use native counters, not lease counts","Owner journal references native API memory state; it does not persist or recreate model/response state after API epoch reset","One API frontend process per mapped endpoint required in current controlled deployment; not a multiworker shared store","No OS/powerloss/disaster/stablecapacity/KEEP proof"]}
atomic_json(j/"reduction.json",out)
def ref(path):
 raw=path.read_bytes();return {"id":path.name,"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"CPU native fixture affinity/wire/journal contracts and installed Responses source snapshots"}
job=json.loads((j/"job.json").read_text());atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"6 CPU Responsesowner/wire/persistence contracts passed; native39 fixtures/source reused, zero model calls, enabled-store E2E pending","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[ref(j/"reduction.json"),ref(j/"test.stderr")],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"passed":6,"cpu_only":True,"model_calls":0,"reduction":ref(j/"reduction.json")}))
