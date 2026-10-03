from pathlib import Path
import sys,json,hashlib,subprocess,re,time,urllib.request,shlex
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process,process_identity
from protocol_receipts import verify_protocol_receipts
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0171";formal=r/"formal";bench=formal/"benchmark";deadline=time.monotonic()+2300
while True:
 state=json.loads((r/"state.json").read_text())
 if state["status"]!="running":break
 assert same_process(state["owner"])and time.monotonic()<deadline;time.sleep(10)
assert state["status"]=="completed"and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for v in spec["stages"][0]["sources"]:
 assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
for stage in["prepare","experiment"]:
 a=json.loads((r/(stage+".phase.json")).read_text());assert a["status"]=="succeeded"and a["exit_code"]==0and not a["timed_out"]
assert json.loads((r/"formal_summary.json").read_text())["functional_acceptance"]
full=json.loads((bench/"full_execution.json").read_text());warm=json.loads((bench/"warmup_execution.json").read_text());assert full["status"]==warm["status"]=="succeeded"and full["exit_code"]==warm["exit_code"]==0
checks={}
for phase,count,inputt,outputt in[("warmup",1,73740,1),("full",4,81932,4096)]:
 a=json.loads((bench/(phase+"_acceptance.json")).read_text());assert a["valid"]and a["actual_count"]==count
 original=json.loads((bench/(phase+"_wire_acceptance.json")).read_text())
 receipt=verify_protocol_receipts(formal/"router_trace.jsonl",count,inputt,outputt,["D1"]);assert receipt==original
 rows=[json.loads(x)for x in Path(a["details_path"]).read_text().splitlines()if x.strip()]
 assert len(rows)==count and len({x["id"]for x in rows})==count and all(x["success"]is True and x["output_tokens"]==outputt for x in rows)
 checks[phase]=dict(receipts=receipt,details_path=a["details_path"],all_actual_requests_valid=True)
ds=json.loads((bench/"dataset_validation.json").read_text());assert ds["valid"]and ds["prefix_text_matches"]and ds["raw_prompt_token_lengths"]==[81920]*4and ds["warmup_prefix_token_lengths"]==[73728]
prior=p/"runs/GLM-RUN-0152/formal/benchmark"
data=[("prefix-GSM8K-in73728-num2-GLM-5.2-w8a8.jsonl",943912,"2df32b5ebf0e9d58c1d5cd3a39463b7b47911874b72910dcaa70d8b4b7e93d6d"),("GSM8K-in81920-num4-GLM-5.2-w8a8-repeatRate0.9.jsonl",2097102,"c8cbc3f51fbc452882f031497d65ca6163fd1bd774f17e9d9106ec1e48d7fb4a")]
for name,size,sha in data:
 f=prior/"dataset"/name;assert f.stat().st_size==size and hashlib.sha256(f.read_bytes()).hexdigest()==sha
reuse=json.loads((bench/"dataset_reuse.json").read_text());assert reuse["source_run"]=="GLM-RUN-0152"and reuse["tokenizer_validation_preserved"]and reuse["first_source_lines_only"]
for row in reuse["files"]:
 b=Path(row["source_path"]).read_bytes();selected=b"".join(b.splitlines(keepends=True)[:row["selected_first_lines"]]);assert hashlib.sha256(b).hexdigest()==row["source_sha256"]and Path(row["path"]).read_bytes()==selected and len(selected)==row["bytes"]and hashlib.sha256(selected).hexdigest()==row["sha256"]
SALT="GLM-RUN-0171-D1-cached-four-4096-c2-20261003T1053Z"
for phase in["warmup","full"]:
 code=(bench/(phase+"_temp_api.py")).read_text();assert "cache_salt="+repr(SALT)in code and "ignore_eos=True"in code and "temperature=0"in code
restore=json.loads((r/"restore_summary.json").read_text());assert restore["private_policy_writes"]==0and restore["native32same_idle"]and restore["D0_policy_unchanged"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=11)and restore["D1_policy_unchanged"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=4096,prefill_cadence=1,serial=1)
events=json.loads((formal/"formal_events.json").read_text());assert len([x for x in events if x["event"]=="controlled_D1_only"])==1and len([x for x in events if x["event"]=="D0_full_peer_restored"])==1
for phase,each in[("warmup",1),("full",4)]:assert {k:sum(x["replica"]==k for x in checks[phase]["receipts"]["requests"])for k in["D0","D1"]}==dict(D0=0,D1=each)
command=json.loads((formal/"benchmark_command.json").read_text())["argv"]
for option,value in[("--input_len","81920"),("--output_len","4096"),("--data_num","4"),("--concurrency","2"),("--request_rate","0"),("--repeat_rate","0.9"),("--prefix_num","1"),("--seed","20260930"),("--host_port","8000"),("--npu_num","16"),("--dp","1")]:assert command[command.index(option)+1]==value
assert command[command.index("--pod_info")+1:command.index("--pod_info")+2]==["172.16.10.167:9900"]
assert not list(Path(checks["full"]["details_path"]).parent.glob("request_scope_amendment.json"))
details=Path(checks["full"]["details_path"]);perf=details.with_name("gsm8k.csv");assert perf.exists()
argv=["python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/aisbench_slo.py","--perf_csv",str(perf),"--details_jsonl",str(details),"--expect_output_len","4096","--concurrency","2","--expected_requests","4","--out_json",str(j/"slo.json"),"--out_md",str(j/"slo.md")]
z=subprocess.run(argv,capture_output=True,timeout=30);(j/"slo.stdout").write_bytes(z.stdout);(j/"slo.stderr").write_bytes(z.stderr);z.check_returncode();slo=json.loads((j/"slo.json").read_text());assert slo["all_requests_succeeded"]and slo["output_len_ok"]and slo["n_success"]==4
acks=[json.loads(x)for x in(r/"native_formal.stdout").read_text().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
trace=[json.loads(x)for x in(formal/"router_trace.jsonl").read_text().splitlines()if x.strip()]
leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("method")=="POST"and x.get("path")=="/v1/chat/completions"]
assert len(leases)==5and sum(x["replica"]=="D0"for x in leases)==0and sum(x["replica"]=="D1"for x in leases)==5
assert not any(x["event"].startswith("pd_")for x in trace)
def metric(f):
 out={}
 for line in f.read_text().splitlines():
  if not line or line.startswith("#"):continue
  name=line.split("{")[0].split()[0]
  if name.startswith("vllm:")and any(v in name for v in["generation_tokens_total","prompt_tokens_total","request_success_total","num_requests_running","num_requests_waiting","prefix_cache_hits_total","prefix_cache_queries_total","num_preemptions_total","kv_cache_usage_perc"]):
   out[name]=out.get(name,0)+float(line.rsplit(" ",1)[1])
 return out
counters={}
for node in["D0","D1"]:
 before=metric(formal/("after_cache_condition_"+node+".metrics"));after=metric(formal/("final_"+node+".metrics"));delta={k:after[k]-v for k,v in before.items()if k.endswith("_total")}
 assert after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==0and delta["vllm:num_preemptions_total"]==0
 if node=="D1":
  assert delta["vllm:request_success_total"]==5and 0<=delta["vllm:prefix_cache_hits_total"]<=401468and delta["vllm:prefix_cache_queries_total"]==401468and delta["vllm:generation_tokens_total"]==16385and delta["vllm:prompt_tokens_total"]==401468
 else:assert all(x==0for x in delta.values())
 counters[node]=dict(delta=delta,idle=True,uncredited_output_cost=0)
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
config,_=checked_config(r/"service_config.json");roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());physical={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 ids={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);physical[key]=dict(root_same=True,actual_NPU16=True,root=v["root"])
 assert v["env"]["VLLM_PP_LAYER_PARTITION"]==("42,36"if o["host"]=="166"else"22,20,20,16")
 plugin="/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_engines137"if o["host"]=="166"else"local_pp168")
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(plugin)+");print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),source_sha256=hashlib.sha256((p/'issue_budget_scheduler_v3.py').read_bytes()).hexdigest())))"
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=30);z.check_returncode();v=json.loads(z.stdout);assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096 if o["host"]=="166"else 8192,prefill_threshold_tokens=1024 if o["host"]=="166"else 4096,prefill_cadence=1,serial=11 if o["host"]=="166"else 1);assert v["source_sha256"]==hashlib.sha256((p.parents[2]/"runtime/issue_budget_scheduler_v3.py").read_bytes()).hexdigest()
public=json.loads((p/"jobs/AUDIT-RUN168V2-20261003T1030Z/public_service_proof.json").read_text());host=public["host"];service=public["container"]
assert same_process(host)and [v.decode()for v in Path("/proc/"+str(host["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==public["argv"]
assert hashlib.sha256(Path(service["config"]["path"]).read_bytes()).hexdigest()==service["config"]["sha256"]and service["native_domains"]==config["native_domains"]
ports=subprocess.check_output(["ss","-ltnp"],text=True);rows=[x for x in ports.splitlines()if re.search(r":8000\s",x)];assert len(rows)==1and "pid="+str(host["pid"])+","in rows[0]
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as res:placement=json.loads(res.read())
assert len(placement["replicas"])==2and all(not x["temporarily_unhealthy"]and not x["group_faulted"]and not x["draining"]and x["active_requests"]==0and x["work_ranking_calibrated"]for x in placement["replicas"])
fault=json.loads((p/"runs/GLM-RUN-0125/response_owners.json.fault").read_text());assert fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
observer=observe(Path(service["config"]["path"]));assert all(x["status"]=="healthy"for x in observer["groups"])
obsowner=json.loads((p/"runs/GLM-RUN-0168/restored/identity_observer/process_owner.json").read_text());assert owner_alive(obsowner)
atomic_json(j/"public_service_proof.json",dict(public,at=utc(),placement=placement))
peaks={}
for node in["D0","D1"]:
 files=list(formal.glob("sample*_"+node+".metrics"));assert files
 vals=[metric(f)for f in files];peaks[node]={k:max(v.get(k,0)for v in vals)for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]}
 # 10s samples only observed pressure, no GPU per-step occupancy proof.
wall=(datetime.fromisoformat(full["finished_at"])-datetime.fromisoformat(full["started_at"])).total_seconds()
warmwall=(datetime.fromisoformat(warm["finished_at"])-datetime.fromisoformat(warm["started_at"])).total_seconds()
def ref(f):
 h=hashlib.sha256();size=0
 with f.open("rb")as stream:
  while True:
   b=stream.read(1048576)
   if not b:break
   h.update(b);size+=len(b)
 return dict(path=str(f),bytes=size,sha256=h.hexdigest())
artifact_files=[f for f in sorted(r.rglob("*"))if f.is_file()and f.name not in ["manifest.json","state.json","controller.log","artifact_index.json","reduction_brief.json"]]
artifacts=[ref(f)for f in artifact_files];atomic_json(j/"artifact_index.json",artifacts)
observer_sources=[ref(Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/aisbench_slo.py")),ref(Path("/data/tiankuan/wio/glm52-pd/deploy/scripts/analyze_slo.py"))]
verdict="INCONCLUSIVE"if slo["slo_all_pass"]else"REJECT"
observer_sources=[ref(Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/aisbench_slo.py")),ref(Path("/data/tiankuan/wio/glm52-pd/deploy/scripts/analyze_slo.py"))]
verdict="INCONCLUSIVE"if slo["slo_all_pass"]else"REJECT"
prediction=json.loads((r/"cache_prediction.json").read_text())
guard=json.loads((p/"runs/GLM-RUN-0168/restored/PP_empty_guard_workers.json").read_text());assert guard["workers"]==16 and guard["operator_math_edits"]==0
out=dict(at=utc(),run_id=r.name,kind="PP4_c2_matched170_cached_4096_finite_decode_window",functional_acceptance=True,measurement_valid=True,verdict=verdict,source_pins=len(spec["stages"][0]["sources"]),observer_sources=observer_sources,complete_outputs=16384,completed_full_requests=4,warmup_outputs=1,warmup_requests=1,internal_helper_commits=0,total_minimum_native_commits=16385,uncredited_native_output_tokens=sum(x["uncredited_output_cost"]for x in counters.values()),full_cli_phase_wall_s=wall,effective_TPS_full_cli=16384/wall,warmup_cli_wall_s=warmwall,actual_active_domains=["D1"],active_NPU_ranks=16,resident_NPU_ranks=32,actualfull_requests_perdomain=dict(D0=0,D1=4),native_counters=counters,native_pressure=peaks,physical_same=physical,public_frontend_same=True,cache_prediction=prediction,cache_prediction_matched=counters["D1"]["delta"]["vllm:prefix_cache_hits_total"]==prediction["predicted_pair_hits"],public_listener=rows[0],public_HTTP_leases_zero=True,checks=checks,SLO=slo,actual_client_SDK_init_finalize=[acks[0],acks[-1]],artifact_index=ref(j/"artifact_index.json"),model_operations=0,gateway_operations=0,dataset_exact152=True,cache_salt=SALT,actual_native_D1_threshold4096=True,D1_allocation8192=True,D0_allocation4096=True,D1_prior168_marker_reused_unchanged_policy=True,D0_unchanged_policy=restore["D0_policy_unchanged"],D1_unchanged_policy=restore["D1_policy_unchanged"],calibrated_work_seconds=True,limits=["2short-output64 D1controlleddiagnostics cannotbe full61440capacity/KEEP/globalbound/robustP99","Exactfirst2full152JSONlinebytes/prefixfirst1/closedc2/onlyD1active16/current168TP4PP4DCP4native32/public/epochs/hints; D0logicaldrain/fullpeerreadd/noD0inference; newepoch/nativeMTP/cache/HBM histories preclude isolatedprior155gain","Freshsharedsalt configactualAPI metadata; nativeD1 actualcachehits recorded/query237604/effectiveDCPblock512/prediction73216each conditional, D0cachequery-hits0/newinference0; nooldfullcache reuse","D0private4096t1024serial11 unchanged/zeroD0inference; D1private8192t4096serial1 unchanged/sourceperstepread/prior168selection; no policywrites/model/publicprocessoperations","NativeallIDs/bodyhash-wire/usage-length-DONE/cost/SDK0 ratherthanCLI0; original6SLOthresholds unchanged"])
def hist(f):
 out={}
 for l in f.read_text().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{")[0].split()[0];out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
out["native_full_histogram_means"]={}
for key in["D1"]:
 before=hist(formal/("after_cache_condition_"+key+".metrics"));after=hist(formal/("final_"+key+".metrics"))
 samples=[f for f in formal.glob("sample*_"+key+".metrics")if hist(f)["vllm:request_success_total"]==before["vllm:request_success_total"]+1and hist(f)["vllm:generation_tokens_total"]==before["vllm:generation_tokens_total"]+1];assert samples
 f=sorted(samples,key=lambda f:int(f.name.split("_")[0][6:]))[-1];warm=hist(f);values={}
 for name in["request_queue_time_seconds","request_prefill_time_seconds","request_decode_time_seconds","time_to_first_token_seconds"]:
  k="vllm:"+name;assert after[k+"_count"]-warm[k+"_count"]==4;values[name]=(after[k+"_sum"]-warm[k+"_sum"])/4
 out["native_full_histogram_means"][key]=dict(warm_excluded_sample=ref(f),means=values,limits=["Reportednativehistograms notGPU-onlykernelbound"])
previous=p/"jobs/AUDIT-RUN170-20261003T1049Z/reduction.json";assert ref(previous)["sha256"]=="35e00c609cca466fc7ecbce54c973f038df3bb6ded9b8057e39316c11e124f92"
prior=json.loads(previous.read_text());out["matched170_c4_comparison"]=dict(audit=ref(previous),prior_concurrency=4,current_concurrency=2,exact_workload_same=True,cache_hits_prior=prior["native_counters"]["D1"]["delta"]["vllm:prefix_cache_hits_total"],cache_hits_current=counters["D1"]["delta"]["vllm:prefix_cache_hits_total"],TTFT_ms=prior["SLO"]["ttft_ms"],TPOT_ms=prior["SLO"]["tpot_ms"],native_histograms=prior["native_full_histogram_means"],full_cli_TPS_prior=prior["effective_TPS_full_cli"],TTFT_P50_delta_ms=slo["ttft_ms"]["p50"]-prior["SLO"]["ttft_ms"]["p50"],TPOT_P50_delta_ms=slo["tpot_ms"]["p50"]-prior["SLO"]["tpot_ms"]["p50"],limits=["Same168native process configuration/exact all4inputs/prefix1/output4096; only c4 to c2 requested plus freshsalt, actual counter-MTP trajectories recorded; single ordered pair, not repeated causal/stable/global capacity proof"])
out["limits"]=["Finite c2 all4inputs output4096 does not prove full61440 workload/global/stable capacity or robust P99","Exact all4 full152 JSONline bytes and prefixfirst1, fresh salt/native Chat untouched; current168 PP4TP4DCP4/native32/public168, D0logicaldrain-finallyfullpeerreadd/noD0inference","Actual cache hit counts authoritative; conditional512-block prediction does not gate native correctness","No native/privatepolicy/model/publicprocess ops; guard16 reused source pinned; SDKinit-final0/full native IDs-wire-bodyhash-usage-length-DONE; six original SLO thresholds retained","Native Running/iteration counters do not prove GPU per-step batch occupancy or hardware bound"]
atomic_json(j/"reduction.json",out)
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="171controlledD1cachedfour-c2-4x4096/16384+warm1/current168TP4PP4DCP4native32/D1allocated8192threshold4096/noD0inference/privatewrites0/logicalD0readd/SDK0/nohelpers/source/fullwire/finiteTPS="+str(16384/wall)+"/SLO="+str(slo["slo_all_pass"])+"/"+verdict,execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualAISBenchallattempts/nativewire/usage-length-DONE/sourceepochs/cost/SLO")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(verdict=verdict,TPS=16384/wall,SLO=slo["slo_all_pass"],native_counters=counters,pressure=peaks)))
