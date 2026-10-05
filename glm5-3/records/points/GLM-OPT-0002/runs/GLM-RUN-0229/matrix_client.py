from pathlib import Path
import json,sys,re,time,hashlib,urllib.request,urllib.error
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
from dynamic_load_probe import run_probe
start_watchdog();r=Path(__file__).parent;p=r.parents[1];BASE="http://127.0.0.1:8000";tracepath=p/"runs/GLM-RUN-0125/router_trace.jsonl";policy=Path("/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp228/batch_queue_policy.json")
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def fetch(url,method="GET",body=None):
 guard();req=urllib.request.Request(url,data=None if body is None else json.dumps(body).encode(),headers={"Content-Type":"application/json"},method=method)
 with http.open(req,timeout=20)as res:assert res.status==200;return res.read()
def metrics(label=None):
 out={}
 for key,ip,port in[("D0","166",9081),("D1","167",9900)]:
  raw=fetch("http://172.16.10."+ip+":"+str(port)+"/metrics")
  if label:(r/(label+"_"+key+".metrics")).write_bytes(raw)
  v={}
  for name in["num_requests_running","num_requests_waiting","generation_tokens_total","prompt_tokens_total","num_preemptions_total"]:
   vals=re.findall(r"^vllm:"+name+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",raw.decode(),re.M);assert vals;v[name]=sum(float(x)for x in vals)
  out[key]=v
 return out
def idle(label):
 for n in range(120):
  v=metrics(label if n==0 else None)
  if all(x["num_requests_running"]==x["num_requests_waiting"]==0for x in v.values())and all(not x["active_requests"]for x in json.loads(fetch(BASE+"/control/replicas"))["replicas"]):return v
  time.sleep(.5)
 raise RuntimeError("native queues not drained")
def setcap(cap,serial,diagnostic):
 prior=json.loads(policy.read_text());assert prior["cohort_id"]=="GLM-COHORT-0228"
 idle("policy"+str(serial)+"before")
 v=dict(schema_version=1,cohort_id="GLM-COHORT-0228",cap=cap,serial=serial,diagnostic=diagnostic,max_records=64 if diagnostic else 0)
 atomic_json(policy,v);atomic_json(r/("queue_policy_serial"+str(serial)+".json"),dict(at=utc(),prior=prior,next=v,model_operations=0,transition_native_idle=True))
 return v
messages=[dict(role="user",content="List three properties of a correct inference service.")]
tokenized=json.loads(fetch(BASE+"/tokenize","POST",dict(model="glm-52",messages=messages)));assert tokenized["count"]==len(tokenized["tokens"])and tokenized["count"]==21
config=json.loads((r/"restored/service_config.json").read_text());peers=json.loads(config["environment"]["GLM_REPLICAS"]);D1=next(x for x in peers if x["id"]=="D1")
before=idle("matrix_before");results=[];serial=1;removed=False
try:
 fetch(BASE+"/control/replicas/D1","DELETE");removed=True
 for n in[1,2,4]:
  for cap in[3,2]:
   serial+=1;selection=setcap(cap,serial,True);label="cap"+str(cap)+"N"+str(n);root=r/label;root.mkdir();cases=[]
   for i in range(n):
    ident=label+"_"+str(i);body=dict(model="glm-52",messages=messages,max_tokens=256,temperature=0,seed=20260930,ignore_eos=True,stream=True,stream_options=dict(include_usage=True),return_token_ids=True,cache_salt=r.name+"-"+ident,chat_template_kwargs=dict(enable_thinking=False))
    atomic_json(root/(ident+".body.json"),body);cases.append(dict(id=ident,input_kind="short21",arrival_s=0,expected_outputs=256,expected_prompt_tokens=21,body=ident+".body.json"))
   plan=dict(run_id=r.name,kind="diagnostic",cases=cases,request_timeout_s=180,contract="Native queueadmission samephysical228/capture/pool3; boundedcap3or2; CPUstepobserver enabled, noGPUmetadata sync, notformalperformance/fullAPI/capacity")
   atomic_json(root/"plan.json",plan);b=idle(label+"_before");out=run_probe(plan,root,BASE,guard);a=idle(label+"_after")
   assert out["functional_acceptance"]and out["effective_public_output_tokens"]==256*n
   delta={key:{k:a[key][k]-b[key][k]for k in["generation_tokens_total","prompt_tokens_total","num_preemptions_total"]}for key in b}
   assert delta["D0"]["generation_tokens_total"]==256*n and delta["D0"]["prompt_tokens_total"]==21*n and delta["D0"]["num_preemptions_total"]==0 and all(x==0for x in delta["D1"].values())
   trace=[json.loads(l)for l in tracepath.read_text().splitlines()]
   for row in out["requests"]:
    leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==r.name+"-"+row["id"]];assert len(leases)==1 and leases[0]["replica"]=="D0"and leases[0]["body_sha256"]==row["body_sha256"]
    lease=leases[0];contracts=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];releases=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]];assert len(contracts)==len(releases)==1and releases[0]["released"]and not releases[0]["backend_failure"]
    wire=contracts[0];assert wire["audit_error"]is None and wire["wire_sha256"]==row["wire"]["sha256"]and wire["wire_bytes"]==row["wire"]["bytes"]and Path(wire["wire_path"]).read_bytes()==Path(row["wire"]["path"]).read_bytes()
   value=dict(label=label,cap=cap,N=n,serial=serial,selection=selection,summary=ref(root/"arrival_summary.json"),native_delta=delta,elapsed_s=out["elapsed_s"],finite_TPS=out["finite_effective_output_tps"],outputs=out["effective_public_output_tokens"],requests=[dict(id=x["id"],TTFT_s=x["ttft_s"],wall_s=x["wall_s"],mean_after_first_output_ms=1000*x["mean_after_first_output_s"],token_ids_sha256=hashlib.sha256(json.dumps(x["committed_token_ids"]).encode()).hexdigest(),token_ids=x["committed_token_ids"])for x in out["requests"]])
   results.append(value);atomic_json(r/"matrix_partial.json",results)
 after=idle("matrix_after");assert after["D0"]["generation_tokens_total"]-before["D0"]["generation_tokens_total"]==3584and after["D1"]["generation_tokens_total"]==before["D1"]["generation_tokens_total"]
 setcap(2,8,False)
 out=dict(at=utc(),valid=True,outputs=3584,new_native_completed=14,matrix=results,CPU_step_observer=True,GPU_metadata_sync=False,native_resource_capacity=3,queue_admission_cap_after=2,policy_serial_after=8,native_operator_changes=0,model_operations=0,public_restarts=0,limits=["Bounded sameepoch N1/2/4 cap3-vs2 withhostCPUsteprecords diagnostic; not formalperformance/fullAPI/KEEP/stablecapacity","Native token trajectories may change across cap/salt/order; only identical vectors support stronger conditionalcomparison"])
 atomic_json(r/"matrix_summary.json",out);print(json.dumps(dict(valid=True,outputs=3584,new_completed=14,cap_after=2)),flush=True)
except BaseException as e:
 atomic_json(r/"matrix_failure.json",dict(at=utc(),error_type=type(e).__name__,error=str(e),partial=results))
 try:idle("failure_native_drain");setcap(3,100,False)
 except BaseException:pass
 raise
finally:
 if removed:fetch(BASE+"/control/replicas","POST",D1)
