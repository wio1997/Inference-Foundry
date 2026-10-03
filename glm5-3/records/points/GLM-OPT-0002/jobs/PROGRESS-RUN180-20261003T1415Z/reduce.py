from pathlib import Path
import json,hashlib,time
from datetime import datetime,timezone
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0180"
state=json.loads((r/"state.json").read_text());assert state["status"]=="running"
now=datetime.now(timezone.utc);event=json.loads((r/"formal/formal_events.json").read_text())[-1]
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def metric(f):
 d={}
 for l in f.read_text().splitlines():
  if l and not l.startswith("#"):
   k=l.split("{")[0].split()[0];d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
 return d
values={}
for key in["D0","D1"]:
 afterfile=r/"formal"/(event["label"]+"_"+key+".metrics");beforefile=r/"formal"/("after_cache_condition_"+key+".metrics")
 a=metric(afterfile);b=metric(beforefile);delta={k:a[k]-v for k,v in b.items()if k.endswith("_total")}
 files=[f for f in(r/"formal").glob("sample*_"+key+".metrics")if int(f.name.split("_")[0][6:])<=int(event["label"][6:])]
 recent=sorted(files,key=lambda f:int(f.name.split("_")[0][6:]))[-7:]
 first=recent[0];last=recent[-1];x=metric(first);y=metric(last)
 n=(int(last.name.split("_")[0][6:])-int(first.name.split("_")[0][6:]))*10
 draft=y["vllm:spec_decode_num_draft_tokens_total"]-x["vllm:spec_decode_num_draft_tokens_total"]
 accept=y["vllm:spec_decode_num_accepted_tokens_total"]-x["vllm:spec_decode_num_accepted_tokens_total"]
 gen=y["vllm:generation_tokens_total"]-x["vllm:generation_tokens_total"]
 values[key]=dict(metrics=ref(afterfile),native_delta=delta,expected_outputs_including_warm=122881,
  native_remaining_output_commit_target=122881-delta["vllm:generation_tokens_total"],
  observed_running=a["vllm:num_requests_running"],observed_waiting=a["vllm:num_requests_waiting"],KV=a["vllm:kv_cache_usage_perc"],
  recent_sample_count=len(recent),approx_endpoint_period_s=n,observed_aggregate_native_TPS=gen/n,
  MTP_accepted_fraction=accept/draft if draft else None,recent_endpoints=[ref(first),ref(last)],
  limits=["Native totals/endpoint approx10s periods; notexactperrequest tokenprogress/GPU step; remaining target is nativeaggregate not deadline guarantee"])
tracecursor=json.loads((r/"formal/trace_cursor.json").read_text());raw=Path(tracecursor["path"]).read_bytes()
assert hashlib.sha256(raw[:tracecursor["bytes"]]).hexdigest()==tracecursor["sha256"]
rows=[json.loads(l)for l in raw[tracecursor["bytes"]:].decode().splitlines()]
leases=[x for x in rows if x["event"]=="lease_acquired"and x.get("output_budget")==61440]
wire=[]
for lease in leases:
 f=p/"runs/GLM-RUN-0125/router_wire"/(lease["lease_id"]+".sse")
 size=f.stat().st_size
 with f.open("rb")as inp:inp.seek(max(0,size-16384));tail=inp.read()
 choices=[]
 for line in tail.splitlines():
  if line.startswith(b"data: {"):
   try:v=json.loads(line[6:])
   except ValueError:continue
   choices+=v.get("choices",[])
 wire.append(dict(lease_id=lease["lease_id"],replica=lease["replica"],path=str(f),observed_bytes=size,
  tail_choices=choices[-3:],tail_DONE=b"data: [DONE]"in tail,complete_wire_SHA_deferred_until_close=True))
out=dict(at=now.isoformat(),kind="ongoing_partial_native_progress_not_terminal_performance",state=state,values=values,partial_wires=wire,
 guards=["No resource/model operations; unique180controller remains in charge","OrdinarybenchmarkSSE token_ids=null, notclaimedfulloutputIDproof; semantic179JSON nativeIDs proof separate",
 "Native lateMTP zero-ish and forcedignoreEOS repetitivecontent observed; no accuracy/quantization-quality claim",
 "Frozenformal.py subprocess1700s/frozenexperiment1750s/phase2200s; short4K wrapper budget inherited. If deadline preventsfull61440, markINVALID/truncation notperformanceupperbound; do not rewritefrozen sources/replayoldqueue",
 "Partial progress/output totals are not completefullSLO/KEEP/stablecapacity; actualcompleted requests musthaveusage/length/DONE/nativecounters"])
(j/"reduction.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n");e=ref(j/"reduction.json")
result=dict(schema_version=1,job_id=j.name,status="completed",summary="Readonly ongoing180 partialnative outputs/lateMTP/zero-wait/forcedlongcontent andfrozenharnessdeadline tracked; no terminalSLO/KEEP or fulloutputID claim",
 execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="ongoingnativepartialprogress/matchedsamplemetrics/SSEtail/budget")],unknowns=out["guards"],decision_request=None,next_check_at=None)
(j/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(dict(evidence=e,values={k:dict(outputs=v["native_delta"]["vllm:generation_tokens_total"],remaining=v["native_remaining_output_commit_target"],TPS=v["observed_aggregate_native_TPS"],MTP=v["MTP_accepted_fraction"])for k,v in values.items()})))
