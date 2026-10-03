from pathlib import Path
import json,hashlib,sys
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1]
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def metrics(f):
 a={}
 for l in f.read_text().splitlines():
  if not l or l.startswith("#"):continue
  name=l.split("{")[0].split()[0]
  if name.endswith("_created")or name.endswith("_bucket"):continue
  a[name]=a.get(name,0)+float(l.rsplit(" ",1)[1])
 return a
out={}
for n,job,sha in [(170,"AUDIT-RUN170-20261003T1049Z","35e00c609cca466fc7ecbce54c973f038df3bb6ded9b8057e39316c11e124f92"),(171,"AUDIT-RUN171-20261003T1056Z","95ea7c3094b0c182a0e8580a7c8a64ed10c9b46067f4755547bae43fc083b41c")]:
 r=p/"runs"/("GLM-RUN-%04d"%n);f=p/"jobs"/job/"reduction.json";assert ref(f)["sha256"]==sha
 audit=json.loads(f.read_text());assert audit["measurement_valid"];state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
 formal=r/"formal";final=metrics(formal/"final_D1.metrics");before=metrics(formal/"after_cache_condition_D1.metrics")
 warmfile=Path(audit["native_full_histogram_means"]["D1"]["warm_excluded_sample"]["path"]);warm=metrics(warmfile)
 delta={k:final[k]-v for k,v in warm.items()if k.endswith("_total")and k.startswith("vllm:spec_decode_")}
 drafted=delta["vllm:spec_decode_num_draft_tokens_total"];accepted=delta["vllm:spec_decode_num_accepted_tokens_total"];drafts=delta["vllm:spec_decode_num_drafts_total"]
 assert drafted>0and 0<=accepted<=drafted and drafts>0
 events=json.loads((formal/"formal_events.json").read_text());samples=[x for x in events if x["event"]=="metrics"and (x["label"].startswith("sample")or x["label"]=="final")]
 samples=sorted(samples,key=lambda x:datetime.fromisoformat(x["at"]))
 intervals=[];weighted=[]
 for left,right in zip(samples,samples[1:]):
  a=metrics(formal/(left["label"]+"_D1.metrics"));b=metrics(formal/(right["label"]+"_D1.metrics"));dt=(datetime.fromisoformat(right["at"])-datetime.fromisoformat(left["at"])).total_seconds()
  if dt<=0:continue
  generation=b["vllm:generation_tokens_total"]-a["vllm:generation_tokens_total"]
  prompt=b["vllm:prompt_tokens_total"]-a["vllm:prompt_tokens_total"]
  maxrun=4 if n==170 else 2
  selected=a["vllm:num_requests_running"]==b["vllm:num_requests_running"]==maxrun and prompt==0
  row=dict(start=left["at"],end=right["at"],seconds=dt,generation=generation,prompt=prompt,start_running=a["vllm:num_requests_running"],end_running=b["vllm:num_requests_running"],finite_interval_TPS=generation/dt,selected_sustained_observed_running=selected)
  intervals.append(row)
  if selected:weighted.append(row)
 trace=[json.loads(l)for l in (formal/"router_trace.jsonl").read_text().splitlines()if l.strip()]
 req=[]
 for lease in [x for x in trace if x["event"]=="lease_acquired"and x.get("method")=="POST"and x.get("output_budget")==4096]:
  same=[x for x in trace if x.get("lease_id")==lease["lease_id"]]
  first=next(x for x in same if x["event"]=="upstream_first_output");release=next(x for x in same if x["event"]=="lease_released")
  req.append(dict(lease_id=lease["lease_id"],request_header_id=lease.get("request_header_id"),body_sha256=lease["body_sha256"],start_ns=lease["monotonic_ns"],TTFT_s=(first["monotonic_ns"]-lease["monotonic_ns"])/1e9,lease_wall_s=(release["monotonic_ns"]-lease["monotonic_ns"])/1e9))
 assert len(req)==4
 d=sum(x["seconds"]for x in weighted);g=sum(x["generation"]for x in weighted)
 out[str(n)]=dict(audit=ref(f),warm_excluded_metrics=ref(warmfile),full_native_speculation_counters=delta,accepted_fraction=accepted/drafted,draft_tokens_per_draft= drafted/drafts,accepted_tokens_per_draft=accepted/drafts,limits=["Counters count native speculative requests/drafts, not physical GPU steps; ratios are aggregate across four requests","Observed maxrunning at sample endpoints does not certify continuous per-step occupancy; prompt counter does not timestamp GPU work"],intervals=intervals,selected_window=dict(seconds=d,generation=g,weighted_TPS=g/d if d else None),native_router_requests=sorted(req,key=lambda x:x["start_ns"]))
atomic_json(j/"reduction.json",dict(at=utc(),kind="completed_pair_native_counter_timeline_reduction",runs=out,new_inference=0,service_operations=0,limits=["Single ordered sameconfiguration all4inputs/output4096 c4-c2 pair, fresh salts; no global or stable capacity certificate","Different MTP trajectory can affect output-token latency; no causal attribution from throughput alone"]))
f=j/"reduction.json";a=ref(f);atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="170171 completed-only native MTP acceptance and observed Running/time windows reduced; inference0/serviceops0",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**a,locator="warmexcluded full4 native counters and raw router timings")],unknowns=["No GPU-step/continuous/globalbound inferred"],decision_request=None,next_check_at=None))
print(json.dumps({n:dict(acceptance=v["accepted_fraction"],accepted_per_draft=v["accepted_tokens_per_draft"],selected=v["selected_window"])for n,v in out.items()}))
