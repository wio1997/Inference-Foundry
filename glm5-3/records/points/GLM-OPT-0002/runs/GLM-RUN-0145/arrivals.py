from pathlib import Path
import json,sys,subprocess,os,signal,time,urllib.request,re,hashlib,statistics
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
from dynamic_load_probe import run_probe
start_watchdog();r=Path(__file__).parent;http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
BASE="http://127.0.0.1:8000";NATIVE={"D0":"http://172.16.10.166:9081","D1":"http://172.16.10.167:9900"}
config=[dict(id=k,url=v)for k,v in NATIVE.items()];groups=json.loads((r/"service_config.json").read_text())["native_domains"];assert len(groups)==2 and {g["members"][0]for g in groups}==set(NATIVE)
trace_path=r.parent/"GLM-RUN-0125/router_trace.jsonl"
def fetch(url,body=None):
 req=urllib.request.Request(url,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={"Content-Type":"application/json"})
 with http.open(req,timeout=15)as reply:raw=reply.read();assert reply.status==200;return raw
def metrics(label=None):
 out={"native":{}}
 for key,url in NATIVE.items():
  raw=fetch(url+"/metrics")
  if label:(r/(label+"_"+key+".metrics")).write_bytes(raw)
  vals={}
  for name in["num_requests_running","num_requests_waiting","kv_cache_usage_perc","generation_tokens_total","prompt_tokens_total","num_preemptions_total"]:
   values=re.findall(r"^vllm:"+name+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",raw.decode(),re.M);assert values;vals[name]=sum(float(v)for v in values)
  out["native"][key]=vals
 out["placement"]=json.loads(fetch(BASE+"/control/replicas"))
 return out
def idle(label):
 for n in range(120):
  out=metrics(label+"_"+str(n))
  if all(v["num_requests_running"]==v["num_requests_waiting"]==v["kv_cache_usage_perc"]==0 for v in out["native"].values()):return out
  time.sleep(.5)
 raise RuntimeError("native queues/KV did not drain")
try:
 guard();assert json.loads(fetch(BASE+"/healthcheck"))["status"]=="ok"
 invalid=dict(model="glm-52",messages=[dict(role="user",content="Long invalid sampling request. "*2048)],max_tokens=32,temperature=-1,stream=False,cache_salt=r.name+"-invalid")
 raw=json.dumps(invalid,ensure_ascii=False).encode();(r/"validation_temperature_negative.body").write_bytes(raw)
 req=urllib.request.Request(BASE+"/v1/chat/completions",data=raw,headers={"Content-Type":"application/json","X-Request-ID":r.name+"-validation-negative-temperature"})
 try:
  with http.open(req,timeout=30)as reply:status=reply.status;wire=reply.read()
 except urllib.error.HTTPError as error:status=error.code;wire=error.read()
 (r/"validation_temperature_negative.wire").write_bytes(wire);assert status==400 and isinstance(json.loads(wire),dict)
 atomic_json(r/"validation_native400.json",dict(status=status,body_sha256=hashlib.sha256(raw).hexdigest(),wire_sha256=hashlib.sha256(wire).hexdigest(),public_credit=0))
 canonical=json.loads((r/"canonical_81932.body.json").read_text());content=canonical["messages"][0]["content"]
 inputs={"long81932":canonical["messages"],"medium8k":[dict(role="user",content=content[:40000])],"medium32k":[dict(role="user",content=content[:160000])],"short":[dict(role="user",content="List three properties of a correct inference service.")]}
 tokenized={}
 for key,messages in inputs.items():
  body=dict(model="glm-52",messages=messages);atomic_json(r/("tokenize_"+key+".body.json"),body);raw=fetch(BASE+"/tokenize",body);(r/("tokenize_"+key+".wire")).write_bytes(raw)
  obj=json.loads(raw);assert type(obj["count"])is int and obj["count"]==len(obj["tokens"]);tokenized[key]=obj["count"]
 assert tokenized["long81932"]==81932 and 0<tokenized["short"]<tokenized["medium8k"]<tokenized["medium32k"]<81932
 atomic_json(r/"native_tokenized_counts.json",tokenized)
 cases=[]
 for case in json.loads((r/"workload_contract.json").read_text())["cases"]:
  ident,kind,arrival,outputs=[case[k]for k in["id","input_kind","arrival_s","expected_outputs"]]
  body=dict(canonical,messages=inputs[kind],max_tokens=outputs,stream=True,stream_options=dict(include_usage=True),return_token_ids=True,cache_salt=r.name+"-"+ident);body.pop("kv_transfer_params",None)
  name=ident+".body.json";atomic_json(r/name,body);cases.append(dict(id=ident,input_kind=kind,arrival_s=arrival,expected_outputs=outputs,expected_prompt_tokens=tokenized[kind],body=name))
 plan=dict(run_id=r.name,kind="diagnostic",cases=cases,request_timeout_s=900,contract="NEW sixteen equal short boundedburst/input21/output4096 each/65536effective/arrivals0..1.5 by0.1; requestseed20260930/nativeargvseed1024/temperature0/ignoreEOS/newcoldsalts. Same137twoindependentlocalTP8PP2DCP8/noEPnativeAllGather/noKV/42,36/K3noSPGraph32/maxseq8/all32same/cohort137CPU4096t1024c1serial1/STORE1/public141V11nativeenginesentry/active_count/state125/no model/frontend/privatepolicyops. Reuse144VALID32768/8complete/124.039956556s/264.172939993TPS/cache0/helper0/preempt0/SDK0/source79/peakRunning4each/outputP5027.852983914msFAIL18. Doubles eachengine concurrency4to8 atcurrentnative maxseq8; eightlocalrequests timesuniformdecode4 equalsGraph32. Actual8requests32768outputs/peakRunning8each/nativecost/full16IDs-counts-usage-length-DONE/native-clientwire/cache0/helper0/source/SDK/all32HOSTepochs/policy/referenceSLO toverify. Changedtotalworkload from144; boundedburst no samepackagegain/steadycapacity/globalbound/KEEP; nativeguards/operators/mathunchanged/oldqueueforbidden.")
 atomic_json(r/"arrival_plan.json",plan);before=idle("before");atomic_json(r/"before_state.json",before)
 out=run_probe(plan,r,BASE,guard,metrics);after=idle("after");atomic_json(r/"after_state.json",after)
 trace=[json.loads(x)for x in trace_path.read_text().splitlines()]
 lease_ids={x["lease_id"]for x in trace if x["event"]=="lease_acquired"and(x.get("request_header_id")or"").startswith(r.name+"-")};trace=[x for x in trace if x.get("lease_id")in lease_ids]
 for a in out["requests"]:
  leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==r.name+"-"+a["id"]];assert len(leases)==1;l=leases[0];a["native_owner"]=l["replica"];assert l["body_sha256"]==a["body_sha256"]
  ss=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==l["lease_id"]];assert len(ss)==1;s=ss[0];assert s["audit_error"]is None and s["wire_sha256"]==a["wire"]["sha256"]and Path(s["wire_path"]).read_bytes()==Path(a["wire"]["path"]).read_bytes()
  releases=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==l["lease_id"]];assert len(releases)==1 and releases[0]["released"]and not releases[0]["backend_failure"]
 def percentile(values,p):
  vals=sorted(values);index=(len(vals)-1)*p/100;lo=int(index);hi=min(lo+1,len(vals)-1);return vals[lo]+(vals[hi]-vals[lo])*(index-lo)
 latency={}
 for kind in["all","short"]:
  rows=out["requests"]if kind=="all"else[x for x in out["requests"]if x["input_kind"]==kind]
  latency[kind]=dict(n=len(rows),ttft_ms={str(p):1000*percentile([x["ttft_s"]for x in rows],p)for p in[50,75,90,99]},mean_after_first_output_ms={str(p):1000*percentile([x["mean_after_first_output_s"]for x in rows],p)for p in[50,90]},max_chunk_gap_s=max(x["max_native_chunk_gap_s"]for x in rows))
 PD=[x for x in trace if x["event"].startswith("pd_")]
 helpers=[x for x in PD if x["event"]=="pd_helper_completed"];assert not helpers and not any(x["event"].startswith("pd_helper_")for x in PD)
 fallback=[x for x in PD if x["event"]=="pd_geometry_native_fallback"];assert not fallback
 assert {v["native_owner"]for v in out["requests"]}=={"D0","D1"}
 for event in PD:
  for key in ["original_body_artifact","helper_body_artifact","artifact","native_body_artifact"]:
   if event.get(key):
    z=event[key];assert not z.get("audit_error");raw=Path(z["path"]).read_bytes();assert len(raw)==z["bytes"]and hashlib.sha256(raw).hexdigest()==z["sha256"]
 out.update(internal_helper_commits=0,validation_native400=True,total_native_commits=65536,PD_true_long_medium5=False,native_geometry_fallback=False,native_helper_raw_complete=True)
 out.update(latency=latency,before=before,after=after,native_tokenized_counts=tokenized,model_operations=0)
 out["diagnostic_reference_SLO"]={key:latency["all"][metric][str(p)]<bound for key,metric,p,bound in[
  ("TTFT_P50_lt_4000ms","ttft_ms",50,4000),("TTFT_P75_lt_8000ms","ttft_ms",75,8000),("TTFT_P90_lt_12000ms","ttft_ms",90,12000),("TTFT_P99_lt_30000ms","ttft_ms",99,30000),("mean_after_first_output_P50_lt_18ms","mean_after_first_output_ms",50,18),("mean_after_first_output_P90_lt_40ms","mean_after_first_output_ms",90,40)]}
 out["limits"]+=["N16 equal4096 short boundedburst/no robusttail/globalcapacity orsteadyarrival certificate","mean_after_first_output=(lastHTTPstreamread-firstcontentarrival)/(Q-1), distinct from nativeGPUper-token time and originalaisbenchformalmeasurement"]
 atomic_json(r/"dynamic_summary.json",out)
finally:
 guard();atomic_json(r/"public_retained_final.json",dict(at=utc(),health=json.loads(fetch(BASE+"/healthcheck")),model_operations=0,frontend_operations=0))
print(json.dumps(dict(outputs=out["effective_public_output_tokens"],elapsed=out["elapsed_s"],latency=out["latency"],SLOreference=out["diagnostic_reference_SLO"])))
