from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1]
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
out={}
for num in[143,144,145]:
 r=p/"runs"/("GLM-RUN-%04d"%num);a=json.loads((r/"arrival_summary.json").read_text());rows=a["requests"]
 assert a["functional_acceptance"]and all(x["completed"]and x["effective_public_output_credit"]==4096for x in rows)
 samples=[json.loads(l)for l in(r/"native_samples.jsonl").read_text().splitlines()]
 points=[(t["origin_s"],t["chunk_commits"])for x in rows for t in x["native_token_chunk_points"]]
 assert sum(n for t,n in points)==a["effective_public_output_tokens"]
 bins=[]
 for start in range(0,int(a["elapsed_s"])+1,20):
  end=min(start+20,a["elapsed_s"])
  if end<=start:continue
  chosen=[s for s in samples if start<=s["elapsed_s"]<end];commits=sum(n for t,n in points if start<=t<end)
  bins.append(dict(start_s=start,end_s=end,HTTP_commits=commits,HTTP_commit_tps=commits/(end-start),arrivals=sum(start<=x["actual_dispatch_s"]<end for x in rows),completions=sum(start<=x["finished_origin_s"]<end for x in rows),sample_n=len(chosen),native_pressure={key:{k:dict(mean=sum(s["native"][key][k]for s in chosen)/len(chosen),max=max(s["native"][key][k]for s in chosen))for k in["num_requests_running","num_requests_waiting","kv_cache_usage_perc"]}for key in["D0","D1"]}if chosen else {}))
 assert sum(x["HTTP_commits"]for x in bins)==a["effective_public_output_tokens"]
 out[r.name]=dict(requests=len(rows),outputs=a["effective_public_output_tokens"],elapsed_s=a["elapsed_s"],finite_TPS=a["finite_effective_output_tps"],bins=bins,sources=[ref(r/"arrival_summary.json"),ref(r/"native_samples.jsonl")],request_wall_s=[x["wall_s"]for x in rows],limits=["20s HTTParrival bins include MTP and transport; native pressure sampled at1s, not kernel throughput/hardwareupperbound","EachN andtotaloutputs differ; finite burst includes initial ramp andterminal drain, notstablearrival capacity"])
v=dict(at=utc(),curve=out,script=ref(Path(__file__)));atomic_json(j/"reduction.json",v)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Read-only balancednative shortburst4/8/16 complete-token arrival bins and native pressure; no GPU/resourceoperations",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="complete committed HTTPchunk counts and samplednativepressure")],unknowns=["Finiteburst notstablecapacity orhardwarebound"],decision_request=None,next_check_at=None))
print(json.dumps({k:dict(TPS=v["finite_TPS"],bins=[dict(start=x["start_s"],TPS=x["HTTP_commit_tps"],pressure=x["native_pressure"])for x in v["bins"]])for k,v in out.items()}))
