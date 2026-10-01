import pathlib
import json,hashlib,re,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent; job=json.loads((j/"job.json").read_text()); r=Path(job["inputs"][0]["path"])
def ref(p,ident):
 return {"id":ident,"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"locator":"entire curated artifact"}
def metric(p,key):
 values=[]
 for line in p.read_text().splitlines():
  if re.match(r"^vllm:"+re.escape(key)+r"(?:\{|\s)",line):
   values.append(float(line.rsplit(" ",1)[1]))
 return sum(values) if values else None
from sse_observer import NativeSSEObserver
from phase_runner import same_process
state=json.loads((r/"state.json").read_text());assert state["status"]=="completed" and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for pin in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
cap=r/"capability";results=list(sorted(cap.glob("*/load_result.json")))+[r/"pilot/dynamic/load_result.json"]+list(sorted(r.glob("control_*/load_result.json")))+list(sorted((r/"prefix_study").glob("*/load_result.json")))
audits=[];negatives=[]
for rp in results:
 data=json.loads(rp.read_text());assert data["valid"]
 plan=json.loads((rp.parent/"plan.json").read_text());assert len(data["requests"])==len(plan["requests"])
 for row in data["requests"]:
  assert row["valid"] and not row["error"]
  body=Path(row["body_path"]).read_bytes();assert hashlib.sha256(body).hexdigest()==row["body_sha256"]
  raw=rp.parent/(str(row["id"])+".response.jsonl");events=[json.loads(line) for line in raw.read_text().splitlines()]
  if row["outcome"]=="expected_rejection":
   import base64
   assert row["http_status"]==row["expected"]["http_status"]==400 and len(events)==1
   payload=base64.b64decode(events[0]["error_body_base64"]);assert len(payload)==events[0]["error_body_bytes"] and hashlib.sha256(payload).hexdigest()==events[0]["error_body_sha256"]
   negatives.append({"body":ref(Path(row["body_path"]),"negative-body"),"response":ref(raw,"negative-response"),"status":400,"output_credit":0});continue
  assert row["outcome"]=="completed" and row["done"] and row["finish_reasons"]
  digest=hashlib.sha256();bp=json.loads(body)
  if bp.get("stream"):
   observer=NativeSSEObserver(collect_contract=True)
   for event in events:
    val=event["data"];assert isinstance(val,str);observer.feed(("data: "+val+"\n\n").encode())
    if val!="[DONE]":
     for c in json.loads(val).get("choices",[]):
      delta=c.get("delta",{});text=delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or c.get("text") or ""
      if text or delta.get("tool_calls") or delta.get("function_call"):digest.update(json.dumps(delta,sort_keys=True,ensure_ascii=False).encode())
   c=observer.contract();assert c["done"] and not c["unknown"] and not c["native_error"] and c["usage"]==row["usage"] and c["finish_reasons"]==row["finish_reasons"]
   assert digest.hexdigest()==row["delta_stream_sha256"]
  else:
   assert len(events)==1;value=events[0]["data"];assert not value.get("error") and value["usage"]==row["usage"]
   assert {str(c.get("index",0)):c["finish_reason"] for c in value["choices"]}==row["finish_reasons"]
  assert len(row["finish_reasons"])==bp.get("n",1) and row["usage"]["completion_tokens"]==row["expected"]["output_tokens"]
  audits.append({"result":str(rp),"request_id":row["id"],"body_sha256":row["body_sha256"],"usage":row["usage"],"done":True,"finish_reasons":row["finish_reasons"],"response_events":len(events),"response":ref(raw,"response-"+str(len(audits)))})
assert len(audits)==33 and len(negatives)==1 and sum(a["usage"]["completion_tokens"] for a in audits)==35476

identity=json.loads((r/"adopted_model_identities.json").read_text())
startup=json.loads((r/"startup_model_identities.json").read_text())
for k,v in identity.items():assert all(v[f]==startup[k][f]for f in ["host","pid","identity","argv"])
assert set(identity)=={"DP0","DP1","DP2","DP3"}
group=json.loads((r/"execution_groups.json").read_text())[0]
assert group["members"]==["DP0","DP1","DP2","DP3"]
assert group["epoch"]==hashlib.sha256(json.dumps({k:{f:v[f]for f in ["host","pid","identity","argv"]}for k,v in identity.items()},sort_keys=True).encode()).hexdigest()
wire_receipts=[];all_bindings={}
for scope,count in [("capability",9),("pilot",6)]:
 trace=[json.loads(x)for x in (r/scope/"router_trace.jsonl").read_text().splitlines()]
 leases=[x for x in trace if x["event"]=="lease_acquired"];assert len(leases)==count
 by_id={};by_body={}
 for lease in leases:
  key=lease["lease_id"];assert key not in all_bindings
  seq=[x for x in trace if x.get("lease_id")==key]
  rel=[x for x in seq if x["event"]=="lease_released"];assert len(rel)==1 and rel[0]["released"] and not rel[0]["backend_failure"]
  hdr=[x for x in seq if x["event"]=="upstream_headers"];assert len(hdr)==1
  contracts=[x for x in seq if x["event"]=="upstream_stream_contract"];assert len(contracts)==1
  contract=contracts[0];assert contract["audit_error"] is None
  raw=Path(contract["wire_path"]).read_bytes();assert len(raw)==contract["wire_bytes"] and hashlib.sha256(raw).hexdigest()==contract["wire_sha256"]
  val={"scope":scope,"lease":lease,"status":hdr[0]["status"],"contract":contract,"release":rel[0]}
  all_bindings[key]=val
  if lease["request_header_id"] is not None:
   assert lease["request_header_id"] not in by_id;by_id[lease["request_header_id"]]=val
  by_body.setdefault(lease["body_sha256"],[]).append(val)
  wire_receipts.append(val)
 result_paths=[x for x in sorted((r/"capability").glob("*/load_result.json")) if not x.parent.name.startswith("local_")] if scope=="capability" else [r/"pilot/dynamic/load_result.json"]
 matches=set()
 for result_path in result_paths:
  data=json.loads(result_path.read_text())
  for row in data["requests"]:
   rid=r.name+"-pilot-"+row["id"]
   match=by_id[rid] if scope=="pilot" else by_body[row["body_sha256"]][0]
   if scope=="capability":assert len(by_body[row["body_sha256"]])==1
   assert match["lease"]["body_sha256"]==row["body_sha256"] and match["status"]==row["http_status"]
   key=match["lease"]["lease_id"];assert key not in matches;matches.add(key)
   raw=Path(match["contract"]["wire_path"]).read_bytes()
   events=[json.loads(x)for x in (result_path.parent/(row["id"]+".response.jsonl")).read_text().splitlines()]
   bp=json.loads(Path(row["body_path"]).read_bytes())
   if row["outcome"]=="expected_rejection":
    import base64
    assert raw==base64.b64decode(events[0]["error_body_base64"]);continue
   if bp["stream"]:
    obs=NativeSSEObserver(collect_contract=True);obs.feed(raw);c=obs.contract()
    assert c==match["contract"]["contract"] and c["done"] and not c["native_error"] and not c["unknown"] and c["usage"]==row["usage"] and c["finish_reasons"]==row["finish_reasons"]
    assert [x[5:].strip()for x in raw.decode().splitlines()if x.startswith("data:")]==[x["data"]for x in events]
   else:assert json.loads(raw)==events[0]["data"]
 if scope=="capability":
  ev=json.loads((r/"capability/capability_events.json").read_text())
  cancel=[x for x in ev if x["event"]=="client_closed_after_first_output"];assert len(cancel)==1
  cp=json.loads((r/"capability/cancel.body.json").read_text());ch=hashlib.sha256(json.dumps(cp,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
  match=by_body[ch][0];assert not match["contract"]["contract"]["done"] and match["contract"]["contract"]["usage"] is None
  assert match["lease"]["lease_id"] not in matches;matches.add(match["lease"]["lease_id"])
  assert cancel[0]["first_event"]in [x[5:].strip()for x in Path(match["contract"]["wire_path"]).read_text().splitlines()if x.startswith("data:")]
  assert len([x for x in ev if x["event"]=="cancel_lease_released"])==1
  for e in ev:
   if "snapshot"in e:
    for item in e["snapshot"]["replicas"]:assert item["execution_group"]==group["id"]and item["native_owner_epoch"]==group["epoch"]and not item["group_faulted"]
  assert all(v==0 for vals in [x for x in ev if x["event"]=="final"][-1]["metrics"].values()for v in vals.values())
 else:
  ev=json.loads((r/"pilot/pilot_events.json").read_text())
  assert all(item["active_requests"]==0 and not item["group_faulted"]for item in json.loads((r/"pilot/final_placement.json").read_text())["replicas"])
 assert ev[-1]["event"]=="gateway_stopped" and ev[-1]["exit_code"]==-15
 assert len(matches)==len(leases)
# Independently validate all four native tool wire contracts using frozen validation code.
tool_rows=[]
tools_source=(r/"tools.py").read_text();validate=tools_source[tools_source.index("  calls={};"):tools_source.index("  attempt.update(",tools_source.index("  calls={};"))]
tools=json.loads((r/"tools_DP0/summary.json").read_text());assert tools["valid"] and len(tools["requests"])==4 and tools["cleanup"]["native_idle"]
for row in tools["requests"]:
 raw=Path(row["wire"]["path"]).read_bytes();assert len(raw)==row["wire"]["bytes"]and hashlib.sha256(raw).hexdigest()==row["wire"]["sha256"]
 ns={"raw":raw,"stream":row["stream"],"name":row["name"],"json":json,"NativeSSEObserver":NativeSSEObserver}
 exec(compile("if True:\n"+validate,"offline native tool validation","exec"),ns)
 assert ns["usage"]==row["usage"] and ns["finish"]==row["finish_reasons"] and list(ns["calls"].values())==row["tools"]
 tool_rows.append({"name":row["name"],"usage":row["usage"],"wire":row["wire"]})
# Metric deltas include actual cache reuse and startup memory; unknown stays unknown.
keys=["prefix_cache_queries_total","prefix_cache_hits_total","external_prefix_cache_queries_total","external_prefix_cache_hits_total","request_prefill_time_seconds_sum","request_queue_time_seconds_sum","request_decode_time_seconds_sum","request_generation_tokens_sum","num_preemptions_total","spec_decode_num_drafts_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total"]
def delta(before,after):
 out={}
 for key in keys:
  a=metric(before,key);b=metric(after,key);out[key]=None if a is None or b is None else b-a
 for key in ["num_requests_running","num_requests_waiting"]:assert metric(before,key)==metric(after,key)==0
 return out
prefix=[]
for rank in range(4):
 for i in range(3):
  p=r/"prefix_study"/("DP"+str(rank)+"_"+str(i));row=json.loads((p/"load_result.json").read_text())["requests"][0]
  original=r.parent/"GLM-RUN-0023/prefix_study"/("canonical_"+str(i)+".body.json")
  assert original.read_bytes()==(p/"request.body.json").read_bytes()==(r/"prefix_study"/original.name).read_bytes()
  counters=delta(r/"prefix_study"/("DP"+str(rank)+"_"+str(i)+"_before.metrics"),r/"prefix_study"/("DP"+str(rank)+"_"+str(i)+"_after.metrics"))
  assert counters["prefix_cache_queries_total"]==row["usage"]["prompt_tokens"]
  prefix.append({"rank":rank,"index":i,"usage":row["usage"],"ttft_s":row["ttft_s"],"tpot_s":row["tpot_s"],"body_sha256":row["body_sha256"],"native_counter_deltas":counters})
dyn=json.loads((r/"pilot/dynamic/load_result.json").read_text())
dynamic={"effective_output_tokens":dyn["effective_output_tokens"],"wall_s":dyn["elapsed_s"],"effective_tps":dyn["effective_tps"],"requests":dyn["requests"],"native_counter_deltas":{n:delta(r/"pilot"/("initial_"+n+".metrics"),r/"pilot"/("final_"+n+".metrics"))for n in identity}}
dynamic["max_sampled_gauges"]={}
for n in identity:
 dynamic["max_sampled_gauges"][n]={}
 for key in ["num_requests_running","num_requests_waiting"]:
  vals=[metric(f,key)for f in(r/"pilot").glob("sample*_"+n+".metrics")]
  dynamic["max_sampled_gauges"][n][key]=max(v for v in vals if v is not None)if any(v is not None for v in vals)else None
cold={"requests":json.loads((r/"cold_summary.json").read_text())["requests"],"native_counter_deltas":{},"limits":"Four cold concurrent vs priorone/two"}
for n in identity:
 assert hashlib.sha256((r/("control_"+n)/"request.body.json").read_bytes()).hexdigest()=="349de8820c328303ac846a72906ecf385589b2fb6b65866dfb4b8951251cba9b"
 cold["native_counter_deltas"][n]=delta(r/("control_"+n)/"initial.metrics",r/("control_"+n)/"final.metrics")
memory={}
pattern=r"Actual usage: ([0-9.]+) GiB for weights, ([0-9.]+) GiB for peak activation, ([0-9.]+) GiB for non-torch memory, ([0-9.]+) GiB for NPU graph memory.*Current KV cache memory: ([0-9.]+) GiB"
for rank in range(4):
 p=r/("DP_run38_"+str(rank)+".log");lines=p.read_text(errors="replace").splitlines();rows=[]
 for i,line in enumerate(lines):
  m=re.search(pattern,line)
  if m:rows.append({"line":i+1,"text":line,"weights_GiB":float(m[1]),"activation_GiB":float(m[2]),"non_torch_GiB":float(m[3]),"graph_GiB":float(m[4]),"KV_GiB":float(m[5])})
 # Native headless logging may differ; telemetry missing is not substituted with zero.
 memory["DP"+str(rank)]={"workers":rows,"ranges":{k:[min(v[k]for v in rows),max(v[k]for v in rows)]if rows else None for k in ["weights_GiB","activation_GiB","non_torch_GiB","graph_GiB","KV_GiB"]},"KV_capacity_lines":[{"line":i+1,"text":l}for i,l in enumerate(lines)if "GPU KV cache size"in l],"log":ref(p,"log"+str(rank))}
installation={}
for rank in range(4):
 lines=(r/("DP_run38_"+str(rank)+".log")).read_text(errors="replace").splitlines()
 installs=[l for l in lines if "GLM_DP_METADATA_INSTALLED" in l]
 shapes=[l for l in lines if "GLM_DP_METADATA " in l]
 installation["DP"+str(rank)]={"install_lines":installs,"shape_lines":shapes}
 assert len(installs)==8,"nativeworker installation receipt missing"
source=(r/"adopt_models.py").read_text().replace('atomic_json(root/"adopted_model_identities.json",identities)','atomic_json(j/"current_model_identities.json",identities)').replace('atomic_json(root/"execution_groups.json",','atomic_json(j/"current_execution_groups.json",')
ns={"__file__":str(r/"adopt_models.py"),"j":j};exec(compile(source,"read-only exact current TP32 cohort","exec"),ns);assert ns["identities"]==identity
import urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
idle={}
for n,node,port in [("DP0","166",9081),("DP1","166",9082),("DP2","167",9900),("DP3","167",9901)]:
 with opener.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=10)as response:text=response.read().decode()
 f=j/("final_"+n+".metrics");f.write_text(text);idle[n]={k:metric(f,k)for k in ["num_requests_running","num_requests_waiting"]};assert all(v==0 for v in idle[n].values())
for entry in json.loads((r/"artifact_index.json").read_text()):
 p=Path(entry["path"]);assert p.stat().st_size==entry["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==entry["sha256"],str(p)
outputs=35476+sum(row["usage"]["completion_tokens"]for row in tool_rows)
m=json.loads((r/"manifest.json").read_text());assert m["valid"] and m["new_client_attempts"]==39 and m["new_completed_inference_requests"]==37 and m["new_effective_output_tokens"]==outputs
out={"run_id":r.name,"measurement_valid":True,"functional_acceptance":True,"verdict":"INCONCLUSIVE","generated_at":utc(),"new_client_attempts":39,"new_completed_inference_requests":37,"new_effective_output_tokens":outputs,"cancelled_output_credit":0,"expected_native400":1,"native_group":group,"native_identities":identity,"final_native_idle":idle,"cold":cold,"prefix_pairs":prefix,"dynamic":dynamic,"native_memory":memory,"native_control_receipts":installation,"offline_response_receipts":audits,"tools_receipts":tool_rows,"negative_receipts":negatives,"proxy_wire_receipts":wire_receipts,"limits":["Finite four scheduler DP4 TP8 DCP8 EP32 package, no full61440/stablecapacity/KEEP","Cold concurrencyfour vsone/two in priorcontrols, aggregate maxseq32 vsprior8/16, caches/MTP/native topology differ; not isolated speedup","Native upstream wire exact hashed, client event-level sequences compared; no byte-exact client wire claim","Actual counters and host startup memory not kernel durations/runtime peaks","No live fault/recovery/persistent gateway crash proof"]}
p=r/"reduction.json";atomic_json(p,out);m["reduction"]=ref(p,"reduction");atomic_json(r/"manifest.json",m)
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"All39 native attempts verified:37 complete"+str(outputs)+"outputs; one400/cancelzero credit, four DP4 functional/nativecounter/memory/lease/source/physicalowner audit","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[{"kind":"fact","text":"Bounded actual native DP4/TP8/DCP8/EP32 four schedulers E2E valid; performance and reuse scope explicit","scope":{"run_id":r.name},"evidence_ids":["reduction"]}],"evidence":[ref(p,"reduction")],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps(ref(p,"reduction")))
