from pathlib import Path
import json,hashlib,sys,subprocess,urllib.request,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0046";state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
pins=json.loads((r/"controller_spec.json").read_text())["stages"][0]["sources"]
for pin in pins:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]and Path(pin["path"]).read_bytes()==Path(pin["snapshot"]).read_bytes()
source=(r/"adopt_models.py").read_text().replace('atomic_json(root/"adopted_model_identities.json",identities)','atomic_json(j/"current_model_identities.json",identities)').replace('atomic_json(root/"execution_groups.json",','atomic_json(j/"current_execution_groups.json",')
ns={"__file__":str(r/"adopt_models.py"),"j":j};exec(compile(source,"current46 exactreadonlyidentity","exec"),ns);assert ns["identities"]==json.loads((r/"adopted_model_identities.json").read_text())
p=subprocess.run(["docker","exec","-e","PYTHONPATH=/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime","glm52-single","python3",str(j/"verify.py")],capture_output=True,timeout=120);(j/"verify.stdout").write_bytes(p.stdout);(j/"verify.stderr").write_bytes(p.stderr);p.check_returncode()
out=json.loads((j/"protocol_reduction.json").read_text());out.update(generated_at=utc(),verdict="INCONCLUSIVE",source_pins=len(pins),physical_owners=ns["identities"])
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open("http://172.16.10.166:9081/metrics",timeout=10)as res:raw=res.read();(j/"final_native.metrics").write_bytes(raw)
def metric(path,key):
 vals=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",path.read_text(),re.M);return sum(float(x)for x in vals)if vals else None
for key in ["num_requests_running","num_requests_waiting"]:assert metric(j/"final_native.metrics",key)==0
begin=r/"initial_0.metrics";end=j/"final_native.metrics"
out["state_native_counter_delta"]={k:metric(end,k)-metric(begin,k)for k in ["generation_tokens_total","num_preemptions_total","prefix_cache_queries_total","prefix_cache_hits_total","spec_decode_num_drafts_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total"]}
assert out["state_native_counter_delta"]["generation_tokens_total"]>=192
out["native_cancelled_or_overshoot_output_no_credit"]=out["state_native_counter_delta"]["generation_tokens_total"]-192
for entry in json.loads((r/"artifact_index.json").read_text()):
 f=Path(entry["path"]);assert f.stat().st_size==entry["bytes"]and hashlib.sha256(f.read_bytes()).hexdigest()==entry["sha256"],str(f)
def ref(f):
 raw=f.read_bytes();return{"id":f.name,"path":str(f),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"independent complete native state/raw/source/owner/counter audit"}
atomic_json(r/"reduction.json",out);brief={k:v for k,v in out.items()if k!="Responses_state"};brief["Responses_state"]={k:v for k,v in out["Responses_state"].items()if k not in ["attempts","native_unique_commit_ledger"]};brief["full_reduction"]=ref(r/"reduction.json");atomic_json(r/"reduction_brief.json",brief)
m=json.loads((r/"manifest.json").read_text());m["reduction"]=ref(r/"reduction.json");atomic_json(r/"manifest.json",m)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Run46 fullnative STORE1/toolrollback/state/restart/background/cancel/SSEreplay/rawbyte/logprob audit passed;10nativecomplete/"+str(out["new_effective_output_tokens"])+"outputs, canceled0credit, exact23pins/physicalowner/nativeidle;no performanceKEEP","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[ref(r/"reduction.json")],"unknowns":out["limits"],"decision_request":None,"next_check_at":None});print(json.dumps({"valid":True,"outputs":out["new_effective_output_tokens"],"native_state_delta":out["state_native_counter_delta"],"reduction":ref(r/"reduction.json")}))
