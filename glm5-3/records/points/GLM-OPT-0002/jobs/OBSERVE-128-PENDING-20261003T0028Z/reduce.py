from pathlib import Path
import json,hashlib,sys,re,statistics
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0128";f=r/"formal"
state=json.loads((r/"state.json").read_text());assert state["status"]=="running"and same_process(state["owner"])
events=json.loads((f/"formal_events.json").read_text());rows=[x for x in events if x["event"]=="metrics"and x["label"].startswith("sample")]
def metrics(path):
 out={}
 for line in path.read_text().splitlines():
  if line.startswith("vllm:")and any(k in line for k in["generation_tokens_total","spec_decode_num_drafts_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total","num_requests_running","num_requests_waiting","kv_cache_usage_perc","prefix_cache_hits_total","prefix_cache_queries_total"]):
   name=line.split("{")[0].split()[0];out[name]=out.get(name,0)+float(line.rsplit(" ",1)[1])
 return out
steady=[x for x in rows if x["gauges"]["D167"]["num_requests_running"]==2and x["gauges"]["D167"]["num_requests_waiting"]==0][-13:]
assert len(steady)>=3
measures=[]
for x in steady:
 raw=f/(x["label"]+"_D167.metrics");measures.append(dict(at=x["at"],label=x["label"],metrics=metrics(raw),ref=dict(path=str(raw),bytes=raw.stat().st_size,sha256=hashlib.sha256(raw.read_bytes()).hexdigest())))
intervals=[]
for a,b in zip(measures,measures[1:]):
 duration=(datetime.fromisoformat(b["at"])-datetime.fromisoformat(a["at"])).total_seconds();delta={k:b["metrics"][k]-v for k,v in a["metrics"].items()if k.endswith("_total")}
 assert duration>0and delta["vllm:generation_tokens_total"]>=0
 intervals.append(dict(start=a["at"],end=b["at"],seconds=duration,native_generation_tokens=delta["vllm:generation_tokens_total"],provisional_native_generation_rate=delta["vllm:generation_tokens_total"]/duration,counters=delta))
cursor=json.loads((f/"trace_cursor.json").read_text());raw=Path(cursor["path"]).read_bytes();assert hashlib.sha256(raw[:cursor["bytes"]]).hexdigest()==cursor["sha256"];trace=[json.loads(x)for x in raw[cursor["bytes"]:].splitlines()if x]
ls=[x for x in trace if x["event"]=="lease_acquired"and x.get("output_budget")==61440];assert len(ls)==2
first=[]
for lease in ls:
 es=[x for x in trace if x["event"]=="upstream_first_output"and x.get("lease_id")==lease["lease_id"]];assert len(es)==1and es[0]["prefill_transitioned"]
 first.append(dict(lease_id=lease["lease_id"],replica=lease["replica"],actual_first_output_from_gateway_acquire_s=(es[0]["monotonic_ns"]-lease["monotonic_ns"])/1e9,input_bytes=lease["input_bytes"],output_budget=61440,body_sha256=lease["body_sha256"]))
assert all(x["replica"]=="D1"for x in first)
period=(datetime.fromisoformat(measures[-1]["at"])-datetime.fromisoformat(measures[0]["at"])).total_seconds();delta={k:measures[-1]["metrics"][k]-measures[0]["metrics"][k]for k in measures[0]["metrics"]if k.endswith("_total")}
drafts=delta["vllm:spec_decode_num_drafts_total"]
source=Path("/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run89/issue_budget_scheduler_v3.py");assert hashlib.sha256(source.read_bytes()).hexdigest()=="4ac2c4f76bbbf778e3919090d18338cafe0a19061bc903961429ebd3707f531c"
out=dict(at=utc(),kind="readonly_pending_native_observation",measurement_status="fullrequests_incomplete",run_id=r.name,first_full_request_outputs=first,native_running=2,native_waiting=0,P_idle=True,sample_refs=measures,intervals=intervals,window_seconds=period,window_native_generation_rate=delta["vllm:generation_tokens_total"]/period,native_request_drafts=drafts,accepted_per_request_draft=delta["vllm:spec_decode_num_accepted_tokens_total"]/drafts if drafts else None,metrics_delta=delta,valid_effective_full_output_credit=0,model_operations=0,signals=0,network_requests=0,provisional_optimization_question="Native perrequest prefill threshold1024 caps2prefills to2048 despiteissuebudget4096; increasingt2048 underremainingnative4096allocation mightreducepairedwarmtailTTFT; reuse116finitepositive/tradeoff but full61440 currentD123 classuntested. Do notalter128frozenconfig orprequeue followup beforeterminal evidence",limits=["Incompletefullrequests; nativegenerationrate isnot committed effectiveTPS/SLO/capacity","Window hostpoll walltime includesRPC/metrics and not GPUtime orphysicaliterations","Draftcounters areper-requestsequence drafts, notengine/GPUrounds; acceptance/trajectory/parallelbatchcost coupled","Two firstTTFT observations do not certifyfourrequestpercentiles/61440completion orrobusttail","Prior116threshold2048 differentload/coldmixedclass giveshypothesis notformalcausalgain"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="128pending2Dfullrequests actualfirstoutputs~6s/native2Running0Waiting/Pidle/windowprovisionalnativegeneration only/zeroeffectivecompletedfullcredit; threshold2048question notqueued",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="immutablepollmetrics/source/actualacquire-firstoutput/partialnotTPS")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(first_output=first,window=period,rate=out["window_native_generation_rate"],accepted_per_draft=out["accepted_per_request_draft"])))
