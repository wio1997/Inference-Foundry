from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from standalone_service_config import checked_config
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0134";old=r.parent/"GLM-RUN-0131"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
state=json.loads((r/"state.json").read_text());assert state["status"]=="failed"and not same_process(state["owner"])
owner=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text());assert owner["run_id"]==r.name and owner["status"]=="failed"
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
with http.open("http://172.16.10.166:9081/metrics",timeout=10)as response:raw=response.read()
(j/"final.metrics").write_bytes(raw)
def values(b):
 return {k:sum(float(x)for x in re.findall(r"^vllm:"+k+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M))for k in ["num_requests_running","num_requests_waiting","generation_tokens_total","prompt_tokens_total","request_success_total","prefix_cache_queries_total","prefix_cache_hits_total","num_preemptions_total"]}
end=values(raw);assert end["num_requests_running"]==end["num_requests_waiting"]==0
initial=values((r/"before_0_D0.metrics").read_bytes());delta={k:end[k]-initial[k]for k in end if k not in["num_requests_running","num_requests_waiting"]}
summary=json.loads((r/"arrival_summary.json").read_text());assert summary["functional_acceptance"]and summary["actual_new_requests"]==12and summary["effective_public_output_tokens"]==10112and not summary["errors"]
assert "assert not fallback"in(r/"native_dynamic.stderr").read_text()and "AssertionError"in(r/"native_dynamic.stderr").read_text()
assert not(r/"dynamic_summary.json").exists()
plan=json.loads((r/"arrival_plan.json").read_text());prior_plan=json.loads((r.parent/"GLM-RUN-0114/arrival_plan.json").read_text());cases=plan["cases"];rows=summary["requests"]
assert [{k:x[k]for k in["id","input_kind","arrival_s","expected_outputs","expected_prompt_tokens"]}for x in cases]==[{k:x[k]for k in["id","input_kind","arrival_s","expected_outputs","expected_prompt_tokens"]}for x in prior_plan["cases"]]
trace_path=r.parent/"GLM-RUN-0125/router_trace.jsonl";trace=[json.loads(l)for l in trace_path.read_text().splitlines()]
headers={r.name+"-"+x["id"]for x in rows};leases=[e for e in trace if e["event"]=="lease_acquired"and e.get("request_header_id")in headers];assert len(leases)==12
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
 l=next(x for x in leases if x["request_header_id"]==r.name+"-"+row["id"]);assert l["replica"]=="D0"and l["body_sha256"]==row["body_sha256"]
 stream=[e for e in owned if e["event"]=="upstream_stream_contract"and e.get("lease_id")==l["lease_id"]];release=[e for e in owned if e["event"]=="lease_released"and e.get("lease_id")==l["lease_id"]]
 assert len(stream)==len(release)==1and stream[0]["audit_error"]is None and stream[0]["wire_sha256"]==row["wire"]["sha256"]and Path(stream[0]["wire_path"]).read_bytes()==wire.read_bytes()
 assert release[0]["released"]and not release[0]["backend_failure"]
 compact.append({k:v for k,v in row.items()if k not in["committed_token_ids","native_token_chunk_points","headers"]}|dict(native_owner="D0",committed_token_ids_count=len(tokens),committed_token_ids_sha256=hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest()))
acks=[json.loads(l)for l in(r/"native_dynamic.stdout").read_text().splitlines()if l.startswith('{"event":')];assert [x["event"]for x in acks]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in acks)
assert delta["generation_tokens_total"]==10112and delta["request_success_total"]==12and delta["prefix_cache_hits_total"]==delta["num_preemptions_total"]==0
assert json.loads((r/"validation_native400.json").read_text())["status"]==400
samples=[json.loads(l)for l in(r/"native_samples.jsonl").read_text().splitlines()];assert len(samples)==summary["metrics_samples"]
peak={k:max(x["native"]["D0"][k]for x in samples)for k in["num_requests_running","num_requests_waiting","kv_cache_usage_perc"]}
def percentile(values,p):
 vals=sorted(values);index=(len(vals)-1)*p/100;lo=int(index);hi=min(lo+1,len(vals)-1);return vals[lo]+(vals[hi]-vals[lo])*(index-lo)
latency={}
for kind in ["all","long81932","medium8k","medium32k","short"]:
 selected=rows if kind=="all"else[x for x in rows if x["input_kind"]==kind]
 latency[kind]=dict(n=len(selected),ttft_ms={str(p):1000*percentile([x["ttft_s"]for x in selected],p)for p in[50,75,90,99]},mean_after_first_output_ms={str(p):1000*percentile([x["mean_after_first_output_s"]for x in selected],p)for p in[50,90]},max_chunk_gap_s=max(x["max_native_chunk_gap_s"]for x in selected))
SLO={key:latency["all"][metric][str(p)]<bound for key,metric,p,bound in[
 ("TTFT_P50_lt_4000ms","ttft_ms",50,4000),("TTFT_P75_lt_8000ms","ttft_ms",75,8000),("TTFT_P90_lt_12000ms","ttft_ms",90,12000),("TTFT_P99_lt_30000ms","ttft_ms",99,30000),("mean_after_first_output_P50_lt_18ms","mean_after_first_output_ms",50,18),("mean_after_first_output_P90_lt_40ms","mean_after_first_output_ms",90,40)]}
effective=10112
config,_=checked_config(r/"service_config.json");public_run=r.parent/"GLM-RUN-0133";container=json.loads((public_run/"public_service_owner.json").read_text())
top=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,stat,comm,args"],text=True)
matches=[int(l.split()[0])for l in top.splitlines()[1:]if "standalone_service_entry.py --config "+str(public_run/"service_config.json")in l];assert len(matches)==1
pid=matches[0];stat=Path("/proc/"+str(pid)+"/stat").read_text();ident=dict(pid=pid,boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),start_ticks=stat[stat.rfind(")")+2:].split()[19])
assert same_process(ident)and {k:ident[k]for k in ["boot_id","start_ticks"]}=={k:container["identity"][k]for k in ["boot_id","start_ticks"]}
ids=[int(v)for v in next(l for l in Path("/proc/"+str(pid)+"/status").read_text().splitlines()if l.startswith("NSpid:")).split()[1:]];assert ids==[pid,container["pid"]]
assert [v.decode()for v in Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(bytes([0]))if v]==container["argv"]
ports=subprocess.check_output(["ss","-ltnp"],text=True);line=next(l for l in ports.splitlines()if re.search(r":8000\s",l));assert "pid="+str(pid)+","in line
with http.open("http://127.0.0.1:8000/healthcheck",timeout=10)as response:h=json.loads(response.read())
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as response:placement=json.loads(response.read())
assert h["status"]=="ok"and h["request_num"]==0and len(placement["replicas"])==1and placement["replicas"][0]["active_requests"]==0and not placement["replicas"][0]["group_faulted"]
marker=next(json.loads(l.split("GLM_SERVICE_ENTRY_INSTALLED ",1)[1])for l in(public_run/"restart.gateway.log").read_text().splitlines()if"GLM_SERVICE_ENTRY_INSTALLED "in l)
assert marker["native_domains"]==config["native_domains"]==container["native_domains"]and marker["pid"]==container["pid"]
state_dir=r.parent/"GLM-RUN-0125";fault=json.loads((state_dir/"response_owners.json.fault").read_text());assert fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
owners=json.loads((state_dir/"response_owners.json").read_text())["owners"]
for key in ["resp_glm_run132_base","resp_glm_run132_child","resp_glm_run133_base","resp_glm_run133_other"]:
 assert owners[key]["group"]=="standalone-native"and owners[key]["epoch"]==config["native_domains"][0]["epoch"]
assert json.loads((public_run/"retire_public.json").read_text())["SDKinit_finalize0"]
atomic_json(j/"public_service_proof.json",dict(at=utc(),host=ident,container=container,argv=container["argv"],marker=marker,listener=line,placement=placement,SDK_init0=True,retained=True,state_dir=str(state_dir)))
index=[ref(f)for f in sorted(r.iterdir())if f.is_file()]
atomic_json(j/"artifact_index.json",index)
out=dict(at=utc(),run_id=r.name,controller_status="failed",controller_failure="Postexecution assertion consumed historicalsharedtrace PDfallbacks",frozen_driver_verdict="INVALID",readonly_recovered_E2E_valid=True,measurement_valid=True,functional_acceptance=True,verdict="REJECT",effective_output_tokens=10112,completed_native_requests=12,helper_count=0,PD_events_current_window=0,historical_PD_events_excluded=len(foreign_PD),SDK_client_init_finalize0=True,source_count=len(pins),source_unchanged=True,native_epochs=live,NPU32_same=True,API_count=1,geometry="DP1TP16PP2DCP16/world32/local16/42,36/noEP_AllGather/noKV/K3noSPGraph32",native_cohort="GLM-COHORT-0131",elapsed_s=summary["elapsed_s"],finite_effective_output_tps=summary["finite_effective_output_tps"],requests=compact,latency=latency,diagnostic_reference_SLO=SLO,native_counter_delta=delta,peak_native_pressure=peak,public_host=ident,public8000_retained=True,idle=True,artifact_index=ref(j/"artifact_index.json"),public_proof=ref(j/"public_service_proof.json"),limits=["Originaldriver failed after all12complete; raw/frozen source preserved; read-only recovered newlease-only audit/noGPUrerun","All10112 output IDs/usage/DONE/wire/nativecounters/coldhits0/source/epochs/clientSDK0 match; controller status isfailed, notcompleted","Sameworkpackage versus114joint topology/TP/DCP/locality/resources/placement/shortMTP5to3/trajectory; 70.275992464finiteTPS vs114111.364025 and11363.127445 isnotisolatedgain","Allsix diagnosticreferenceSLO failed isconditionalconfiguration REJECT, not stablecapacity/globalbound orpermanentrejection","Mean after first clientarrival distinct nativeGPUtime and originalAISBench61440 formalTPOT;12samples notrobustP99","CurrentNone/noKEEP; nativeSTORE not replicated/concurrentnativecustomID/automaticphysicalfault stillunknown"])
atomic_json(j/"reduction.json",out)
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="needs_decision",summary="Run134 readonlyrecoveredvalid10112 outputs/12complete/cold0/SDK0; originaldriverINVALIDhistoricaltrace; allsixSLOFAIL/configREJECT",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="complete diagnostic and retained native/public identity audit")],unknowns=out["limits"],decision_request="Proceed fullAPI and discriminating variableload research",next_check_at=None));print(json.dumps(dict(valid=True,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),effective=effective,elapsed=summary["elapsed_s"],TPS=summary["finite_effective_output_tps"],SLO=SLO,peak=peak,delta=delta,public=ident)))
