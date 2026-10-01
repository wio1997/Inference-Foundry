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
evidence=[];waves=[]
for wave in [{"name":"dynamic","policy":"active_count"}]:
 w=r/"pilot";d=json.loads((w/"dynamic/load_result.json").read_text())
 t=[json.loads(x) for x in (w/"router_trace.jsonl").read_text().splitlines()]
 bindings={};active={};progress=set();events={}
 for event in t:
  key=event.get("lease_id");kind=event["event"]
  if kind=="lease_acquired":
   rid=event["request_header_id"]
   assert rid not in bindings and key not in active
   before={node:{"active":sum(v["replica"]==node for v in active.values()),"pending_first_output":sum(v["replica"]==node and k not in progress for k,v in active.items())} for node in ["DP0","DP1"]}
   bindings[rid]={**event,"before":before};active[key]=event;events[key]=[event]
  elif key in events:
   events[key].append(event)
   if kind=="upstream_first_output" and event.get("prefill_transitioned"):progress.add(key)
   if kind=="lease_released":
    assert key in active;active.pop(key);progress.discard(key)
 assert not active
 rows=[]
 for req in d["requests"]:
  rid=r.name+"-"+w.name+"-"+req["id"];bind=bindings[rid]
  assert bind["body_sha256"]==req["body_sha256"] and req["valid"] and req["done"] and req["finish_reasons"]
  seq=events[bind["lease_id"]];output=[x for x in seq if x["event"]=="upstream_first_output"];release=[x for x in seq if x["event"]=="lease_released"]
  assert len(output)==1 and len(release)==1 and output[0]["prefill_transitioned"]
  rows.append({"id":req["id"],"replica":bind["replica"],"request_header_id":rid,"body_sha256":req["body_sha256"],"lease_id":bind["lease_id"],"before_placement":bind["before"],"ttft_s":req["ttft_s"],"tpot_ms":1000*req["tpot_s"],"wall_s":req["elapsed_s"],"usage":req["usage"],"valid":req["valid"],"first_output_from_epoch_s":(output[0]["monotonic_ns"]-d["started_monotonic_ns"])/1e9,"release_from_epoch_s":(release[0]["monotonic_ns"]-d["started_monotonic_ns"])/1e9})
 deltas={}
 for node in ["DP0","DP1"]:
  keys=["prefix_cache_queries_total","prefix_cache_hits_total","external_prefix_cache_queries_total","external_prefix_cache_hits_total","num_preemptions_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total"]
  vals={}
  for key in keys:
   a=metric(w/("initial_"+node+".metrics"),key);b=metric(w/("final_"+node+".metrics"),key)
   vals[key]=b-a if a is not None and b is not None else None
  vals["acceptance_ratio"]=vals["spec_decode_num_accepted_tokens_total"]/vals["spec_decode_num_draft_tokens_total"] if vals["spec_decode_num_draft_tokens_total"] else None
  vals["prefix_hit_ratio"]=vals["prefix_cache_hits_total"]/vals["prefix_cache_queries_total"] if vals["prefix_cache_queries_total"] else None
  for key in ["num_requests_running","num_requests_waiting"]:
   samples=[metric(p,key) for p in w.glob("sample*_"+node+".metrics")]
   vals["max_sampled_"+key]=max(x for x in samples if x is not None) if samples else None
   vals["final_"+key]=metric(w/("final_"+node+".metrics"),key)
   assert vals["final_"+key]==0
  deltas[node]=vals
 for name in ["dynamic/load_result.json","router_trace.jsonl","initial_DP0.metrics","final_DP0.metrics","initial_DP1.metrics","final_DP1.metrics"]:
  evidence.append(ref(w/name,wave["name"]+"-"+name.replace("/","-")))
 waves.append({"name":wave["name"],"policy":wave["policy"],"effective_output_tokens":d["effective_output_tokens"],"wall_s":d["elapsed_s"],"effective_tps":d["effective_tps"],"requests":rows,"native_counter_deltas":deltas})
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
assert len(audits)==23 and len(negatives)==1 and sum(a["usage"]["completion_tokens"] for a in audits)==22930
# Verify every gateway lease including negative/cancelled attempts. Direct local requests audited above.
trace=[json.loads(x) for x in (cap/"router_trace.jsonl").read_text().splitlines()];leases=[x for x in trace if x["event"]=="lease_acquired"];assert len(leases)==9
for lease in leases:
 seq=[x for x in trace if x.get("lease_id")==lease["lease_id"]]
 releases=[x for x in seq if x["event"]=="lease_released"];assert len(releases)==1 and releases[0]["released"]
ev=json.loads((cap/"capability_events.json").read_text())
cancel=[x for x in ev if x["event"]=="client_closed_after_first_output"];assert len(cancel)==1 and json.loads(cancel[0]["first_event"])["choices"]
assert len([x for x in ev if x["event"]=="cancel_lease_released"])==1
assert ev[-1]["event"]=="gateway_stopped" and ev[-1]["exit_code"]==-15
assert all(v==0 for vals in [x for x in ev if x["event"]=="final"][-1]["metrics"].values() for v in vals.values())
pilot_ev=json.loads((r/"pilot/pilot_events.json").read_text());assert pilot_ev[-1]["event"]=="gateway_stopped" and pilot_ev[-1]["exit_code"]==-15
assert all(x["active_requests"]==0 for x in json.loads((r/"pilot/final_placement.json").read_text())["replicas"])
prefix=[]
keys=["prefix_cache_queries_total","prefix_cache_hits_total","external_prefix_cache_queries_total","external_prefix_cache_hits_total","request_prefill_time_seconds_sum","request_queue_time_seconds_sum","request_decode_time_seconds_sum","request_generation_tokens_sum","num_preemptions_total","spec_decode_num_drafts_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total"]
for node in ["DP0","DP1"]:
 for i in range(3):
  p=r/"prefix_study"/(node+"_"+str(i));row=json.loads((p/"load_result.json").read_text())["requests"][0]
  before=r/"prefix_study"/(node+"_"+str(i)+"_before.metrics");after=r/"prefix_study"/(node+"_"+str(i)+"_after.metrics")
  deltas={}
  for k in keys:
   a=metric(before,k);b=metric(after,k);deltas[k]=b-a if a is not None and b is not None else None
  assert deltas["prefix_cache_queries_total"]==row["usage"]["prompt_tokens"]
  for k in ["num_requests_running","num_requests_waiting"]:assert metric(before,k)==metric(after,k)==0
  prefix.append({"replica":node,"index":i,"body_sha256":row["body_sha256"],"usage":row["usage"],"ttft_s":row["ttft_s"],"tpot_s":row["tpot_s"],"native_counter_deltas":deltas,"before":ref(before,node+str(i)+"-before"),"after":ref(after,node+str(i)+"-after")})
from phase_runner import same_process
assert not same_process(json.loads((r/"state.json").read_text())["owner"])
prior=r.parent/"GLM-RUN-0023/prefix_study";record=json.loads((prior/"dataset_identity.json").read_text())
for i in range(3):
 raw=(r/"prefix_study"/("canonical_"+str(i)+".body.json")).read_bytes()
 assert raw==(prior/("canonical_"+str(i)+".body.json")).read_bytes() and hashlib.sha256(raw).hexdigest()==record["payload_hashes"][i]
 for node in ["DP0","DP1"]:assert raw==(r/"prefix_study"/(node+"_"+str(i))/"request.body.json").read_bytes()

# Position acceptance counters and native histogram sums are host metrics, not device kernel timings.
for row in prefix:
 node=row["replica"];i=row["index"];before=r/"prefix_study"/(node+"_"+str(i)+"_before.metrics");after=r/"prefix_study"/(node+"_"+str(i)+"_after.metrics")
 def positions(p):
  vals={}
  for line in p.read_text().splitlines():
   if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
    m=re.search(r'position="([^"]+)"',line)
    if m:vals[m[1]]=float(line.rsplit(" ",1)[1])
  return vals
 a=positions(before);b=positions(after);row["position_accepted_delta"]={k:b[k]-a[k] for k in a if k in b}
 drafts=row["native_counter_deltas"]["spec_decode_num_drafts_total"];accepted=row["native_counter_deltas"]["spec_decode_num_accepted_tokens_total"]
 row["expected_emitted_per_draft_sequence"]=1+accepted/drafts if drafts else None
 row["source_notes"]="MTP drafts count logical sequences, not physical engine rounds; addedbonus/overshoot not committed output credit"
tool_rows=[]
tools_source=(r/"tools.py").read_text();validate=tools_source[tools_source.index("  calls={};"):tools_source.index("  attempt.update(",tools_source.index("  calls={};"))]
for rank in ["DP0","DP1"]:
 data=json.loads((r/("tools_"+rank)/"summary.json").read_text());assert data["valid"] and len(data["requests"])==4 and data["cleanup"]["native_idle"]
 for row in data["requests"]:
  raw=pathlib.Path(row["wire"]["path"]).read_bytes();assert len(raw)==row["wire"]["bytes"] and hashlib.sha256(raw).hexdigest()==row["wire"]["sha256"]
  ns={"raw":raw,"stream":row["stream"],"name":row["name"],"json":json,"NativeSSEObserver":NativeSSEObserver}
  exec(compile("if True:\n"+validate,"offline native tool validation","exec"),ns)
  assert ns["usage"]==row["usage"] and ns["finish"]==row["finish_reasons"] and list(ns["calls"].values())==row["tools"]
  tool_rows.append({"rank":rank,"kind":"tool_"+row["name"],"usage":row["usage"],"wire":row["wire"]})
memory={}
pattern=r"Actual usage: ([0-9.]+) GiB for weights, ([0-9.]+) GiB for peak activation, ([0-9.]+) GiB for non-torch memory, ([0-9.]+) GiB for NPU graph memory.*Current KV cache memory: ([0-9.]+) GiB"
for rank in [0,1]:
 p=r/("DP_run28_"+str(rank)+".log");lines=p.read_text(errors="replace").splitlines();rows=[]
 for i,line in enumerate(lines):
  m=re.search(pattern,line)
  if m:rows.append({"line":i+1,"text":line,"weights_GiB":float(m[1]),"activation_GiB":float(m[2]),"non_torch_GiB":float(m[3]),"graph_GiB":float(m[4]),"KV_GiB":float(m[5])})
 assert len(rows)==16
 memory["DP"+str(rank)]={"workers":rows,"ranges":{k:[min(v[k] for v in rows),max(v[k] for v in rows)] for k in ["weights_GiB","activation_GiB","non_torch_GiB","graph_GiB","KV_GiB"]},"KV_capacity_lines":[{"line":i+1,"text":l} for i,l in enumerate(lines) if "GPU KV cache size" in l],"log":ref(p,"native-log-"+str(rank))}
identity=json.loads((r/"adopted_model_identities.json").read_text());old=json.loads((r.parent/"GLM-RUN-0028/startup_model_identities.json").read_text())
for key,item in identity.items():assert all(item[k]==old[key][k]for k in ["host","pid","identity","argv"])
# Re-verify live identities/health without restarting or sending inference.
source=(r/"adopt_models.py").read_text().replace('atomic_json(root/"adopted_model_identities.json",identities)','atomic_json(j/"current_model_identities.json",identities)')
ns={"__file__":str(r/"adopt_models.py"),"j":j};exec(compile(source,"read-only exact current cohort","exec"),ns);assert ns["identities"]==identity
import urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));idle={}
for rank,node,port in [(0,"166",9081),(1,"167",9900)]:
 with opener.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=10) as reply:text=reply.read().decode()
 p=j/("final_DP"+str(rank)+".metrics");p.write_text(text)
 idle["DP"+str(rank)]={k:metric(p,k)for k in ["num_requests_running","num_requests_waiting"]}
 assert all(v==0 for v in idle["DP"+str(rank)].values())
 for k in ["num_requests_running","num_requests_waiting"]:
  assert metric(r/("control_DP"+str(rank)+"/initial.metrics"),k)==metric(r/("control_DP"+str(rank)+"/final.metrics"),k)==0
for row in audits:
 if "/control_" in row["result"]:assert row["body_sha256"]=="349de8820c328303ac846a72906ecf385589b2fb6b65866dfb4b8951251cba9b"
tool_outputs=sum(v["usage"]["completion_tokens"]for v in tool_rows);assert len(tool_rows)==8
outputs=22930+tool_outputs
manifest=json.loads((r/"manifest.json").read_text());assert manifest["valid"] and manifest["new_client_attempts"]==33 and manifest["new_completed_inference_requests"]==31 and manifest["new_effective_output_tokens"]==outputs
for row in json.loads((r/"artifact_index.json").read_text()):
 p=Path(row["path"]);assert p.stat().st_size==row["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"],str(p)
out={"run_id":r.name,"measurement_valid":True,"functional_acceptance":True,"verdict":"INCONCLUSIVE","generated_at":utc(),"waves":waves,"prefix_pairs":prefix,"offline_response_receipts":audits,"tools_receipts":tool_rows,"negative_receipts":negatives,"native_memory":memory,"final_native_idle":idle,"native_identities":identity,"new_client_attempts":33,"new_completed_inference_requests":31,"new_effective_output_tokens":outputs,"cancelled_attempts":1,"cancelled_output_credit":0,"capability_gateway_leases":9,"cold":json.loads((r/"cold_summary.json").read_text()),"installation":{"source_conditioned":True,"explicitWorker_nativeInit_guarded":True,"CPU_external_INFO_filtered":True,"live_patched_object_or_shape_observed":False},"limits":["Finite diagnostic, no full61440/current KEEP/stable capacity","Native Worker installation inferred from explicit argv/import path/source guards/virtual init flow and healthy loaded engine, not direct live methods or shape receipt","MTP trajectories/tool stop outputs differ; one dynamic window slower than baseline26, no causal gain/rejection of exercised padding mechanism","Native memory startup reports are not runtime peaks","Response jsonl evidence is event-level, exact original wire only tools","Coupled physical ranks both retained during HTTP logical drain; no physical elasticity claim"],"evidence":evidence}
p=r/"reduction.json";atomic_json(p,out)
manifest["reduction"]=ref(p,"reduction");atomic_json(r/"manifest.json",manifest)
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Run29 all33 attempts verified:31 completed"+str(outputs)+" outputs,one400 andonecancelled excluded; metadata Worker cohort finite E2E valid, no gain/KEEP","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[{"kind":"fact","text":"23 committed native response event receipts plus8 tool wires validated, body/Usage/finish/DONE/errors/unknown/canonical hashes and every lease/native idle/source pin/native startup32workers verified","scope":{"run_id":r.name},"evidence_ids":["reduction"]}],"evidence":[ref(p,"reduction")],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"requests":31,"outputs":outputs,"dynamic_tps":waves[0]["effective_tps"],"memory_ranges":{k:v["ranges"]for k,v in memory.items()},"prefix_pairs":[{"rank":v["replica"],"index":v["index"],"ttft_s":v["ttft_s"],"tpot_s":v["tpot_s"],"native":v["native_counter_deltas"]}for v in prefix]}))
