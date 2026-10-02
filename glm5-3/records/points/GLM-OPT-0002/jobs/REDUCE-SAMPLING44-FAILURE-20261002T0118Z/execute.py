import json,hashlib,re,subprocess,shlex,sys,urllib.request
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0044";job=json.loads((j/"job.json").read_text())
def ref(p):
 raw=p.read_bytes();return {"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
state=json.loads((r/"state.json").read_text());assert state["status"]=="failed"and state["failure_phase"]=="tools"and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for x in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]and Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()
body=r/"control_TP0/request.body.json";assert ref(body)["sha256"]=="349de8820c328303ac846a72906ecf385589b2fb6b65866dfb4b8951251cba9b"
cold=json.loads((r/"control_TP0/load_result.json").read_text());assert cold["valid"]and len(cold["requests"])==1;row=cold["requests"][0];assert row["done"]and row["usage"]["prompt_tokens"]==81932 and row["usage"]["completion_tokens"]==2048 and row["finish_reasons"]=={"0":"length"}
obs=NativeSSEObserver(collect_contract=True)
for x in (r/"control_TP0/cold_TP0.response.jsonl").read_text().splitlines():obs.feed(("data: "+json.loads(x)["data"]+"\n\n").encode())
c=obs.contract();assert c["done"]and not c["unknown"]and not c["native_error"]and c["usage"]==row["usage"]
tp=r/"tools_TP32/auto.wire";raw=tp.read_bytes();o=NativeSSEObserver(collect_contract=True);o.feed(raw);tc=o.contract();assert tc["done"]and not tc["native_error"]and not tc["unknown"]and tc["usage"]=={"prompt_tokens":160,"completion_tokens":4,"total_tokens":164}
calls={}
for line in raw.decode().splitlines():
 if line.startswith("data: ")and line!="data: [DONE]":
  value=json.loads(line[6:])
  for choice in value.get("choices",[]):
   for item in choice.get("delta",{}).get("tool_calls")or[]:
    c=calls.setdefault(item.get("index",0),{"name":"","arguments":""})
    for key in c:c[key]+=(item.get("function")or{}).get(key)or""
assert calls=={0:{"name":"get_weather","arguments":"{}"}}
expected=json.loads((r/"tools_TP32/auto.body.json").read_text());assert expected["messages"][0]["content"]=="Call get_weather for Shanghai."and expected["tools"][0]["function"]["parameters"]["required"]==["city"]
cleanup=json.loads((r/"tools_TP32/cleanup.json").read_text());assert cleanup["native_idle"]
source=(r/"adopt_models.py").read_text().replace('atomic_json(root/"adopted_model_identities.json",identities)','atomic_json(j/"current_model_identities.json",identities)').replace('atomic_json(root/"execution_groups.json",','atomic_json(j/"current_execution_groups.json",')
ns={"__file__":str(r/"adopt_models.py"),"j":j};exec(compile(source,"current44 readonlyidentity","exec"),ns);assert ns["identities"]==json.loads((r/"adopted_model_identities.json").read_text())
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open("http://172.16.10.166:9081/metrics",timeout=10)as res:native=res.read();(j/"final_native.metrics").write_bytes(native)
def metric(p,key):
 vals=re.findall(r"^vllm:"+re.escape(key)+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",p.read_text(),re.M);return sum(float(x)for x in vals)if vals else None
assert metric(j/"final_native.metrics","num_requests_running")==metric(j/"final_native.metrics","num_requests_waiting")==0
names=["generation_tokens_total","spec_decode_num_drafts_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total","num_preemptions_total","prefix_cache_queries_total","prefix_cache_hits_total"]
delta={k:metric(r/"control_TP0/final.metrics",k)-metric(r/"control_TP0/initial.metrics",k)for k in names}
assert delta["generation_tokens_total"]==2048
logs={}
for rank,node in [(0,"166"),(1,"167")]:
 argv=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/TP_run44_"+str(rank)+".log"]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
 data=subprocess.check_output(argv,timeout=30);path=j/("native_"+node+".log");path.write_bytes(data);logs[node]=ref(path)
# Source-only mechanism snapshot, no native operator changes.
upstream="/vllm-workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py";data=subprocess.check_output(["docker","exec","glm52-single","cat",upstream],timeout=30);(j/"upstream_llm_base_proposer.py").write_bytes(data)
out={"run_id":r.name,"diagnostic_evidence_valid":True,"functional_acceptance":False,"verdict":"REJECT","generated_at":utc(),"inference_attempts":2,"completed_protocol_streams":2,"native_committed_outputs":2052,"functionally_valid_completed_requests":1,"functionally_valid_output_tokens":2048,"invalid_tool_output_credit":0,"cold":{"row":row,"events":ref(r/"control_TP0/cold_TP0.response.jsonl"),"native_counter_deltas":delta},"tool":{"request":ref(r/"tools_TP32/auto.body.json"),"wire":ref(tp),"native_contract":tc,"expected":{"name":"get_weather","arguments":{"city":"Shanghai"}},"actual":calls,"function_contract":False},"physical_owners":ns["identities"],"source_pins":len(spec["stages"][0]["sources"]),"native_logs":logs,"source_mechanism":{"native_upstream":ref(j/"upstream_llm_base_proposer.py"),"finding":"Installed enable_reduce_sample suppresses vocabgather. Ascend MTP compute_draft_token_ids uses local model.compute_logits then inherited _sample_from_logits which returns local argmax without globaloffset/allgather; targetgreedy uses TPglobalargmax. Source-supported namespace asymmetry aligns withactual zero MTPacceptance, not full live draft-ID capture proof.","operator_edits":0},"model_signals":0,"remaining_stages_executed":False,"limits":["Failednativefunction contract candidate, no formal/dynamic/prefix/Responses state proof","RawcoldUsage2048valid, tool4rawnativecommit invalidfunction so0functionalcredit","No kernel/profile/qualitycomparison or hardwarebound","Zeroactualacceptance and source namespace asymmetry do not prove all per-rank runtime tokenIDs without livecapture","Native STORE1configured butstatefulnativeAPI notyettested"]}
atomic_json(r/"reduction.json",out);atomic_json(r/"reduction_brief.json",out)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="REJECT",failure_phase="tools",reduction=ref(r/"reduction.json"),results={k:out[k]for k in ["inference_attempts","completed_protocol_streams","native_committed_outputs","functionally_valid_completed_requests","functionally_valid_output_tokens","invalid_tool_output_credit"]});atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nREJECT native enable_reduce_sample=True for current GLM MTP/TP32 configuration. Twoactualattempts: cold81932→2048 protocolvalid TTFT34.504575s/TPOT85.832638ms; firstnativeauto-tool completes4tokens/DONE/Usage but get_weather arguments{} lacksrequiredShanghai city. No furtherstages ran. Functionvalidcredit2048 only; failedtool4excluded. NativecoldMTPacceptedcounter recorded0; source-supported local/globaldraftID asymmetry, actualliveIDs notcaptured. NativeAPIhealthy/idle exactcohort retained; no operator/vendor edits, no native STORE-state proof/performanceKEEP/hardwarebound. Realterminalaudit referencesall26pins/raw/source/owners.\n")
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Run44 actual native tool requiredcityregression REJECT; cold2048 valid, native tool4 invalidfunction0credit, no furtherphases; readonlyowner/idle/source audit","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[{"id":"reduction",**ref(r/"reduction.json"),"locator":"cold native commit/counters and failedtool raw/source/currentowner/idle"}],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"verdict":"REJECT","native_delta":delta,"tool":calls,"output_credit":2048,"reduction":ref(r/"reduction.json")}))
