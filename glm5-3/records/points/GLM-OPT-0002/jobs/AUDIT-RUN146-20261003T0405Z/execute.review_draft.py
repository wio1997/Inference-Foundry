from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0146";old=r.parent/"GLM-RUN-0131"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
owner=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text());assert owner["run_id"]==r.name and owner["status"]=="completed"
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"]
for pin in pins:
 assert Path(pin["path"]).read_bytes()==Path(pin["snapshot"]).read_bytes()and ref(Path(pin["path"]))["sha256"]==pin["sha256"]
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());live={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=100);(j/(key+".stdout")).write_bytes(z.stdout);(j/(key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 live[key]=dict(root=v["root"],NPU16_same=True)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def values(b):
 return {k:sum(float(x)for x in re.findall(r"^vllm:"+k+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M))for k in ["num_requests_running","num_requests_waiting","generation_tokens_total","prompt_tokens_total","request_success_total","prefix_cache_queries_total","prefix_cache_hits_total","num_preemptions_total"]}
deltas={}
for key,node,port in[("D0","166",9081),("D1","167",9900)]:
 with http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=10)as response:raw=response.read()
 (j/(node+"_final.metrics")).write_bytes(raw);end=values(raw);assert end["num_requests_running"]==end["num_requests_waiting"]==0
 initial=values((r/("before_0_"+key+".metrics")).read_bytes());deltas[key]={k:end[k]-initial[k]for k in end if k not in["num_requests_running","num_requests_waiting"]}
delta={k:sum(v[k]for v in deltas.values())for k in deltas["D0"]}
summary=json.loads((r/"arrival_summary.json").read_text());assert summary["functional_acceptance"]and summary["actual_new_requests"]==48and summary["effective_public_output_tokens"]==196608and not summary["errors"]
assert not (r/"native_dynamic.stderr").read_text()
assert json.loads((r/"dynamic_summary.json").read_text())["functional_acceptance"]
plan=json.loads((r/"arrival_plan.json").read_text());prior_plan=json.loads((r/"workload_contract.json").read_text());cases=plan["cases"];rows=summary["requests"]
assert [{k:x[k]for k in["id","input_kind","arrival_s","expected_outputs","expected_prompt_tokens"]}for x in cases]==[{k:x[k]for k in["id","input_kind","arrival_s","expected_outputs","expected_prompt_tokens"]}for x in prior_plan["cases"]]
trace_path=r.parent/"GLM-RUN-0125/router_trace.jsonl";trace=[json.loads(l)for l in trace_path.read_text().splitlines()]
headers={r.name+"-"+x["id"]for x in rows};leases=[e for e in trace if e["event"]=="lease_acquired"and e.get("request_header_id")in headers];assert len(leases)==48
ids={e["lease_id"]for e in leases};owned=[e for e in trace if e.get("lease_id")in ids]
PD=[e for e in owned if e["event"].startswith("pd_")];assert not PD
foreign_PD=[e for e in trace if e["event"].startswith("pd_")and e.get("lease_id")not in ids];assert foreign_PD
compact=[]
for row,case in zip(rows,cases):
 assert row["id"]==case["id"]and row["completed"]and row["http_status"]==200and row["effective_public_output_credit"]==case["expected_outputs"]
 assert row["usage"]==dict(prompt_tokens=case["expected_prompt_tokens"],completion_tokens=case["expected_outputs"],total_tokens=case["expected_prompt_tokens"]+case["expected_outputs"])
 assert row["contract"]["done"]and not row["contract"]["native_error"]and not row["contract"]["unknown"]and row["contract"]["finish_reasons"]=={"0":"length"}
 tokens=row["committed_token_ids"];assert len(tokens)==case["expected_outputs"]and all(type(v)is int and v>=0for v in tokens)
 wire=Path(row["wire"]["path"]);assert ref(wire)==row["wire"]
 body=r/case["body"];assert ref(body)["sha256"]==row["body_sha256"]
 actual=json.loads(body.read_text());assert actual["cache_salt"]==r.name+"-"+row["id"]and "kv_transfer_params"not in actual
 l=next(x for x in leases if x["request_header_id"]==r.name+"-"+row["id"]);assert l["replica"]in["D0","D1"]and l["body_sha256"]==row["body_sha256"]
 stream=[e for e in owned if e["event"]=="upstream_stream_contract"and e.get("lease_id")==l["lease_id"]];release=[e for e in owned if e["event"]=="lease_released"and e.get("lease_id")==l["lease_id"]]
 assert len(stream)==len(release)==1and stream[0]["audit_error"]is None and stream[0]["wire_sha256"]==row["wire"]["sha256"]and Path(stream[0]["wire_path"]).read_bytes()==wire.read_bytes()
 assert release[0]["released"]and not release[0]["backend_failure"]
 compact.append({k:v for k,v in row.items()if k not in["committed_token_ids","native_token_chunk_points","headers"]}|dict(native_owner=l["replica"],committed_token_ids_count=len(tokens),committed_token_ids_sha256=hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest()))
acks=[json.loads(l)for l in(r/"native_dynamic.stdout").read_text().splitlines()if l.startswith('{"event":')];assert [x["event"]for x in acks]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in acks)
assert delta["generation_tokens_total"]==196608and delta["request_success_total"]==48and delta["prefix_cache_hits_total"]==delta["num_preemptions_total"]==0
assert json.loads((r/"validation_native400.json").read_text())["status"]==400
samples=[json.loads(l)for l in(r/"native_samples.jsonl").read_text().splitlines()];assert len(samples)==summary["metrics_samples"]
peak={key:{k:max(x["native"][key][k]for x in samples)for k in["num_requests_running","num_requests_waiting","kv_cache_usage_perc"]}for key in["D0","D1"]}
assert {x["native_owner"]for x in compact}=={"D0","D1"}
for key in["D0","D1"]:
 mine=[x for x in compact if x["native_owner"]==key]
 assert mine and deltas[key]["generation_tokens_total"]==sum(x["effective_public_output_credit"]for x in mine)and deltas[key]["request_success_total"]==len(mine)
 assert 0<peak[key]["num_requests_running"]<=8
assert not(r/"sample_error.json").exists()
assert max(abs(x["actual_dispatch_s"]-x["scheduled_arrival_s"])for x in rows)<1
points=[(t["origin_s"],t["chunk_commits"])for x in rows for t in x["native_token_chunk_points"]]
assert sum(n for t,n in points)==196608
arrival_end=max(x["scheduled_arrival_s"]for x in rows)
assert arrival_end==470 and [x["scheduled_arrival_s"]for x in rows]==list(range(0,480,10))
def backlog(t):
 return sum(x["actual_dispatch_s"]<=t<x["finished_origin_s"]for x in rows)
bins=[]
boundaries=list(range(0,421,60))+[470,summary["elapsed_s"]]
for start,end in zip(boundaries,boundaries[1:]):
 assert end>start
 chosen=[x for x in samples if start<=x["elapsed_s"]<end];commits=sum(n for t,n in points if start<=t<end)
 bins.append(dict(start_s=start,end_s=end,HTTP_committed_tokens=commits,HTTP_committed_tps=commits/(end-start),actual_arrivals=sum(start<=x["actual_dispatch_s"]<end for x in rows),full_completed_requests=sum(start<=x["finished_origin_s"]<end for x in rows),request_backlog_start=backlog(start),request_backlog_end=backlog(end),sample_n=len(chosen),native_pressure={key:{k:dict(mean=sum(x["native"][key][k]for x in chosen)/len(chosen),max=max(x["native"][key][k]for x in chosen))for k in["num_requests_running","num_requests_waiting","kv_cache_usage_perc"]}for key in["D0","D1"]}if chosen else {}))
assert sum(x["HTTP_committed_tokens"]for x in bins)==196608
flow=dict(offered_requests_per_s=.1,nominal_offered_output_tokens_per_s=409.6,last_scheduled_arrival_s=arrival_end,full_drain_elapsed_s=summary["elapsed_s"],backlog_at_last_scheduled_arrival=backlog(arrival_end),max_dispatch_lateness_s=max(x["actual_dispatch_s"]-x["scheduled_arrival_s"]for x in rows),bins=bins,limits=["Original146 NEWfinite regularopen48 completed; actual48NEWcold input21/output4096 each/196608/arrivals0..470 period10/fullIDs-promptcounts-usage-length-DONE/body-native-clientwire/source/helper0/cache0/preempt0/clientSDK0/same32epochs verified","Twoindependentlocal native16NPUeach/active_count/same141cohort/public/state; actualperownercounts-outputs-deltas/nativeRunning<=8 audited; nativeWaiting reported notassumedzero; no model/frontend/privatepolicyops","Nominal0.1requests-per-second/409.6offered-output-tokens-per-second; totalwindow finiteTPS includesrampanddrain, flow_bins separateobservedarrivals/commits/fullcompletion/backlog/samplednativepressure; notinfinite stability/globalbound/KEEP/robustP99","Mean after first HTTPclientarrival distinct nativeGPUtime and originalAISBench61440 formalTPOT","Nativeargvseed1024/requestseed20260930 inheritedcanonical/hashedbody/temperature0/ignoreEOS; no outputamendment/qualitycomparison; inheritedprepared_state source114_workpackage_reused label doesnotdescribeNEW146frozenarrivalcontract","CurrentNone; nativeSTORE not replicated/concurrentnativecustomID/automaticphysicalfault stillunknown"])
atomic_json(j/"reduction.json",out)
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="needs_decision",summary="Run146 valid196608 outputs/48complete/regularopenperiod10/twoindependentnative/active_count/cache0/helper0/SDK0/NPU32same; measuredconditionalSLO/noKEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="complete diagnostic and retained native/public identity audit")],unknowns=out["limits"],decision_request="Proceed fullAPI and discriminating variableload research",next_check_at=None));print(json.dumps(dict(valid=True,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),effective=effective,elapsed=summary["elapsed_s"],TPS=summary["finite_effective_output_tps"],SLO=SLO,peak=peak,delta=delta,public=ident)))
