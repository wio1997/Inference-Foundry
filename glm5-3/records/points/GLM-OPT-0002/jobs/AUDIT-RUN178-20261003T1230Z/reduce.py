from pathlib import Path
import sys,json,hashlib,subprocess,re,time,urllib.request,shlex
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process,process_identity
from protocol_receipts import verify_protocol_receipts
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0178";formal=r/"formal";bench=formal/"benchmark"
state=json.loads((r/"state.json").read_text())
assert state["status"]=="completed"and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for v in spec["stages"][0]["sources"]:
 assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
for stage in["prepare","experiment"]:
 a=json.loads((r/(stage+".phase.json")).read_text());assert a["status"]=="succeeded"and a["exit_code"]==0and not a["timed_out"]
assert json.loads((r/"formal_summary.json").read_text())["functional_acceptance"]
full=json.loads((bench/"full_execution.json").read_text());warm=json.loads((bench/"warmup_execution.json").read_text());assert full["status"]==warm["status"]=="succeeded"and full["exit_code"]==warm["exit_code"]==0
checks={}
for phase,count,inputt,outputt in[("warmup",2,73740,1),("full",4,81932,4096)]:
 a=json.loads((bench/(phase+"_acceptance.json")).read_text());assert a["valid"]and a["actual_count"]==count
 original=json.loads((bench/(phase+"_wire_acceptance.json")).read_text())
 receipt=verify_protocol_receipts(formal/"router_trace.jsonl",count,inputt,outputt,["D0","D1"]);assert receipt==original
 rows=[json.loads(x)for x in Path(a["details_path"]).read_text().splitlines()if x.strip()]
 assert len(rows)==count and len({x["id"]for x in rows})==count and all(x["success"]is True and x["output_tokens"]==outputt for x in rows)
 checks[phase]=dict(receipts=receipt,details_path=a["details_path"],all_actual_requests_valid=True)
ds=json.loads((bench/"dataset_validation.json").read_text());assert ds["valid"]and ds["prefix_text_matches"]and ds["raw_prompt_token_lengths"]==[81920]*4and ds["warmup_prefix_token_lengths"]==[73728]*2
prior=p/"runs/GLM-RUN-0152/formal/benchmark"
data=[("prefix-GSM8K-in73728-num2-GLM-5.2-w8a8.jsonl",943912,"2df32b5ebf0e9d58c1d5cd3a39463b7b47911874b72910dcaa70d8b4b7e93d6d"),("GSM8K-in81920-num4-GLM-5.2-w8a8-repeatRate0.9.jsonl",2097102,"c8cbc3f51fbc452882f031497d65ca6163fd1bd774f17e9d9106ec1e48d7fb4a")]
for name,size,sha in data:
 f=prior/"dataset"/name;assert f.stat().st_size==size and hashlib.sha256(f.read_bytes()).hexdigest()==sha
reuse=json.loads((bench/"dataset_reuse.json").read_text());assert reuse["source_run"]=="GLM-RUN-0152"and reuse["tokenizer_validation_preserved"]and reuse["first_source_lines_only"]
for row in reuse["files"]:
 b=Path(row["source_path"]).read_bytes();selected=b"".join(b.splitlines(keepends=True)[:row["selected_first_lines"]]);assert hashlib.sha256(b).hexdigest()==row["source_sha256"]and Path(row["path"]).read_bytes()==selected and len(selected)==row["bytes"]and hashlib.sha256(selected).hexdigest()==row["sha256"]
SALT="GLM-RUN-0178-dual-PP4-cached-four-4096-c4-t512-count-matched177-20261003T1230Z"
for phase in["warmup","full"]:
 code=(bench/(phase+"_temp_api.py")).read_text();assert "cache_salt="+repr(SALT)in code and "ignore_eos=True"in code and "temperature=0"in code
restore=json.loads((r/"restore_summary.json").read_text());assert restore["private_policy_writes"]==4and restore["native32same_idle"]and restore["D0_policy_restored"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3)and restore["D1_policy_restored"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=4096,prefill_cadence=1,serial=13)
events=json.loads((formal/"formal_events.json").read_text());assert len([x for x in events if x["event"]=="controlled_count_fallback_both_domains"])==1and len([x for x in events if x["event"]=="D0_full_peer_restored"])==1
for phase,each in[("warmup",1),("full",2)]:assert {k:sum(x["replica"]==k for x in checks[phase]["receipts"]["requests"])for k in["D0","D1"]}==dict(D0=each,D1=each)
command=json.loads((formal/"benchmark_command.json").read_text())["argv"]
for option,value in[("--input_len","81920"),("--output_len","4096"),("--data_num","4"),("--concurrency","4"),("--request_rate","0"),("--repeat_rate","0.9"),("--prefix_num","1"),("--seed","20260930"),("--host_port","8000"),("--npu_num","32"),("--dp","2")]:assert command[command.index(option)+1]==value
assert command[command.index("--pod_info")+1:command.index("--pod_info")+3]==["172.16.10.166:9081","172.16.10.167:9900"]
assert not list(Path(checks["full"]["details_path"]).parent.glob("request_scope_amendment.json"))
details=Path(checks["full"]["details_path"]);perf=details.with_name("gsm8k.csv");assert perf.exists()
argv=["python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/aisbench_slo.py","--perf_csv",str(perf),"--details_jsonl",str(details),"--expect_output_len","4096","--concurrency","4","--expected_requests","4","--out_json",str(j/"slo.json"),"--out_md",str(j/"slo.md")]
z=subprocess.run(argv,capture_output=True,timeout=30);(j/"slo.stdout").write_bytes(z.stdout);(j/"slo.stderr").write_bytes(z.stderr);z.check_returncode();slo=json.loads((j/"slo.json").read_text());assert slo["all_requests_succeeded"]and slo["output_len_ok"]and slo["n_success"]==4
acks=[json.loads(x)for x in(r/"native_formal.stdout").read_text().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
trace=[json.loads(x)for x in(formal/"router_trace.jsonl").read_text().splitlines()if x.strip()]
leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("method")=="POST"and x.get("path")=="/v1/chat/completions"]
assert len(leases)==6and sum(x["replica"]=="D0"for x in leases)==3and sum(x["replica"]=="D1"for x in leases)==3
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
 assert delta["vllm:request_success_total"]==3and 0<=delta["vllm:prefix_cache_hits_total"]<=237604and delta["vllm:prefix_cache_queries_total"]==237604and delta["vllm:generation_tokens_total"]==8193and delta["vllm:prompt_tokens_total"]==237604
 counters[node]=dict(delta=delta,idle=True,uncredited_output_cost=0)
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
config,_=checked_config(r/"service_config.json");roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());physical={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 ids={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);physical[key]=dict(root_same=True,actual_NPU16=True,root=v["root"])
 assert v["env"]["VLLM_PP_LAYER_PARTITION"]=="22,20,20,16"
 plugin="/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp175"if o["host"]=="166"else"local_pp168")
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(plugin)+");print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),source_sha256=hashlib.sha256((p/'issue_budget_scheduler_v3.py').read_bytes()).hexdigest())))"
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=30);z.check_returncode();v=json.loads(z.stdout);assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=1024 if o["host"]=="166"else 4096,prefill_cadence=1,serial=3 if o["host"]=="166"else 13);assert v["source_sha256"]==hashlib.sha256((p.parents[2]/"runtime/issue_budget_scheduler_v3.py").read_bytes()).hexdigest()
public=json.loads((p/"jobs/AUDIT-RUN175-20261003T1200Z/public_service_proof.json").read_text());host=public["host"];service=public["container"]
assert same_process(host)and [v.decode()for v in Path("/proc/"+str(host["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==public["argv"]
assert hashlib.sha256(Path(service["config"]["path"]).read_bytes()).hexdigest()==service["config"]["sha256"]and service["native_domains"]==config["native_domains"]
ports=subprocess.check_output(["ss","-ltnp"],text=True);rows=[x for x in ports.splitlines()if re.search(r":8000\s",x)];assert len(rows)==1and "pid="+str(host["pid"])+","in rows[0]
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as res:placement=json.loads(res.read())
assert len(placement["replicas"])==2and all(not x["temporarily_unhealthy"]and not x["group_faulted"]and not x["draining"]and x["active_requests"]==0and x["work_ranking_calibrated"]for x in placement["replicas"])
fault=json.loads((p/"runs/GLM-RUN-0125/response_owners.json.fault").read_text());assert fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
observer=observe(Path(service["config"]["path"]));assert all(x["status"]=="healthy"for x in observer["groups"])
obsowner=json.loads((p/"runs/GLM-RUN-0175/restored/identity_observer/process_owner.json").read_text());assert owner_alive(obsowner)
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
out=dict(at=utc(),run_id=r.name,kind="dual_PP4_count_fallback_c2each_four4096_t512_matched176177_finite_window",functional_acceptance=True,measurement_valid=True,verdict=verdict,source_pins=len(spec["stages"][0]["sources"]),observer_sources=observer_sources,complete_outputs=16384,completed_full_requests=4,warmup_outputs=2,warmup_requests=2,internal_helper_commits=0,total_minimum_native_commits=16386,uncredited_native_output_tokens=sum(x["uncredited_output_cost"]for x in counters.values()),full_cli_phase_wall_s=wall,effective_TPS_full_cli=16384/wall,warmup_cli_wall_s=warmwall,actual_active_domains=["D0","D1"],active_NPU_ranks=32,resident_NPU_ranks=32,actualfull_requests_perdomain=dict(D0=2,D1=2),native_counters=counters,native_pressure=peaks,physical_same=physical,public_frontend_same=True,cache_prediction=prediction,cache_prediction_matched=all(x["delta"]["vllm:prefix_cache_hits_total"]==prediction["conditional_hits_per_domain"]for x in counters.values()),public_listener=rows[0],public_HTTP_leases_zero=True,checks=checks,SLO=slo,actual_client_SDK_init_finalize=[acks[0],acks[-1]],artifact_index=ref(j/"artifact_index.json"),model_operations=0,gateway_operations=0,dataset_exact152=True,cache_salt=SALT,actual_native_both_threshold512=True,D1_threshold4096_restored=True,private_policy_writes=4,D1_allocation8192=True,D0_allocation8192=True,D1_native168_unchanged_except_private_threshold=True,D0_policy_restored=restore["D0_policy_restored"],D1_policy_restored=restore["D1_policy_restored"],count_fallback_all_available=True,calibrated_work_seconds_restored=True)

def hist(f):
 out={}
 for l in f.read_text().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{")[0].split()[0];out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
out["native_full_histogram_means"]={}
for key in["D0","D1"]:
 before=hist(formal/("after_cache_condition_"+key+".metrics"));after=hist(formal/("final_"+key+".metrics"))
 samples=[f for f in formal.glob("sample*_"+key+".metrics")if hist(f)["vllm:request_success_total"]==before["vllm:request_success_total"]+1and hist(f)["vllm:generation_tokens_total"]==before["vllm:generation_tokens_total"]+1];assert samples
 f=sorted(samples,key=lambda f:int(f.name.split("_")[0][6:]))[-1];warm=hist(f);values={}
 for name in["request_queue_time_seconds","request_prefill_time_seconds","request_decode_time_seconds","time_to_first_token_seconds"]:
  k="vllm:"+name;assert after[k+"_count"]-warm[k+"_count"]==2;values[name]=(after[k+"_sum"]-warm[k+"_sum"])/2
 out["native_full_histogram_means"][key]=dict(warm_excluded_sample=ref(f),means=values,limits=["Reportednativehistograms notGPU-onlykernelbound"])
selected=json.loads((r/"policy_consumption.json").read_text());assert selected["operator_math_changes"]==0and selected["both_threshold512"]and selected["allocated8192_unchanged"]
for node,serial in[("166",2),("167",12)]:
 v=selected["native_selected"][node];assert len(v)==1and v[0]["cohort"]=="GLM-COHORT-0137"and v[0]["serial"]==serial and v[0]["budget_tokens"]==8192and v[0]["prefill_threshold_tokens"]==512and v[0]["prefill_cadence"]==1and not v[0]["fallback"]and v[0]["native_max"]==8192
for stem,owner,t,serial,previous_t,previous_serial in[("D0_threshold512","D0",512,2,1024,1),("D1_threshold512","D1",512,12,4096,11),("D0_threshold1024_restore","D0",1024,3,512,2),("D1_threshold4096_restore","D1",4096,13,512,12)]:
 a=json.loads((r/(stem+"_ack.json")).read_text());assert a[owner+"_only_atomic_update"]and a["both_independent_engines_idle"]and a["model_operations"]==0and a["new"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=t,prefill_cadence=1,serial=serial)
 assert a["previous"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=previous_t,prefill_cadence=1,serial=previous_serial)
out["policy_consumption"]=selected
out["prior_windows"]={}
for num,job,sha in[(177,"AUDIT-RUN177-20261003T1218Z","b6168ddd2204cc9964292c1f86c59387a6708e14ae1e441e1d5b51c5151eb237"),(176,"AUDIT-RUN176-20261003T1211Z","d86065de068124465c7cd22f0595915a27c7f11707268d53bd42ef418437c991"),(172,"AUDIT-RUN172-20261003T1112Z","a6a1c27d665aa5d12dd378c1694d15538326f9d0a694bbc0df804e5b8c8b2201"),(173,"AUDIT-RUN173-20261003T1118Z","e06c78cabd201764487f034bc2260960168781329eff40a3f9b84789dbbf90ec"),(174,"AUDIT-RUN174-20261003T1124Z","50b8e4ed1cff75904926e3541ec16c6ec7ce75f375b828be217fdfa2f9b04654")]:
 f=p/"jobs"/job/"reduction.json";assert ref(f)["sha256"]==sha;v=json.loads(f.read_text())
 out["prior_windows"][str(num)]=dict(audit=ref(f),SLO=v["SLO"],native_histograms=v["native_full_histogram_means"],full_cli_TPS=v["effective_TPS_full_cli"],limits=["Same all4fullinput152/output4096, 176177bothnativePP4domains/all4output4096/c4/countfallback/twowarm/threshold1024 versus178threshold512 +freshsalt/orderedMTPcachehistories; prior172173174singleD1differentclass; noisolatedcausalstablegain"])
out["full_native_MTP"]={}
for key in["D0","D1"]:
 warmfile=Path(out["native_full_histogram_means"][key]["warm_excluded_sample"]["path"]);w=hist(warmfile);e=hist(formal/("final_"+key+".metrics"))
 delta={k:e[k]-v for k,v in w.items()if k.startswith("vllm:spec_decode_")and k.endswith("_total")}
 d=delta["vllm:spec_decode_num_draft_tokens_total"];accepted=delta["vllm:spec_decode_num_accepted_tokens_total"];n=delta["vllm:spec_decode_num_drafts_total"];assert d>0and n>0and 0<=accepted<=d
 out["full_native_MTP"][key]=dict(counters=delta,accepted_fraction=accepted/d,accepted_per_draft=accepted/n,warm_excluded=ref(warmfile),final_metrics=ref(formal/("final_"+key+".metrics")),limits=["Native speculative counters do not count physical GPU steps"])
fallback=next(x for x in events if x["event"]=="controlled_count_fallback_both_domains")
assert all(not x["work_ranking_calibrated"]and not x["active_requests"]and not x["group_faulted"]for x in fallback["placement"]["replicas"])
peer=fallback["full_original_D0_peer"];fallbackpeer=fallback["full_fallback_D0_peer"];assert fallbackpeer=={k:v for k,v in peer.items()if k not in["decode_tps","prefill_bytes_per_s"]}
compiled=config["environment"]["GLM_REPLICAS"];compiled=json.loads(compiled)if isinstance(compiled,str)else compiled;assert peer==next(x for x in compiled if x["id"]=="D0")
for row in placement["replicas"]:
 expected=next(x for x in compiled if x["id"]==row["id"]);assert row["decode_tps_hint"]==expected["decode_tps"]and row["prefill_bytes_per_s_hint"]==expected["prefill_bytes_per_s"]
# Both raw native STORE retrievals precede benchmarking and incur no inference.
for run,name,nativeid in[("GLM-RUN-0175","newD0_create","resp_glm_run175_D0_new"),("GLM-RUN-0168","newD1_create","resp_glm_run168_D1_new")]:
 c=json.loads((p/"runs"/run/"client_final_summary.json").read_text());base=next(x for x in c["requests"]if x["name"]==name);assert (formal/(nativeid+"_retained.wire")).read_bytes()==Path(base["wire"]["path"]).read_bytes()
guards={}
for key,run in[("D0","GLM-RUN-0175"),("D1","GLM-RUN-0168")]:
 f=p/"runs"/run/"restored/PP_empty_guard_workers.json";v=json.loads(f.read_text());assert v["workers"]==16and len(v["markers"])==16and v["operator_math_edits"]==0;guards[key]=dict(evidence=ref(f),fixed_source={x["fixed_source_sha256"]for x in v["markers"]}.pop())
assert guards["D0"]["fixed_source"]==guards["D1"]["fixed_source"];out["guard16_each"]=guards
out["limits"]=["Finitefull4x4096/2warm/c4/2eachnativePP4 threshold512; not full61440/globalstablecapacity/KEEP/robustN4P99 or GPUstepcapacitybound","Matched176177 workload/deployment/routing/cacheclass, onlybothperrequestthreshold1024→512 +freshsalt/orderednativecacheMTP histories; CPUcontrol existingbounds legalmultiple128/allocated8192 andoperator/MTPvalidation unchanged","D0ownedidleCAS1024serial1→512serial2→restore1024serial3; D1ownedidleCAS4096serial11→512serial12→restore4096serial13; actualnativeSELECTED2/12/no fallback/4policyacks/source111/freshsame32ownedidle","BothSTORE/nativewire/epoch/public175 preserved; countfallback viaonlyD0unknownrates/allfullgroupfields retained thenoriginalfullcompilerpeer+hints restored, no model/frontendops/helper-preempt-uncredited/SDK0/nativeIDs-wire-body-usage-length-DONE; originalsixSLOthresholds retained"]
atomic_json(j/"reduction.json",out)
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="178dualPP4matched176177/countfallback/c4actual2each/full4x4096+2warm/nativecomplete16386/boththreshold512/nativeSELECTED2-12/restoredD0t1024serial3-D1t4096serial13/SDK0/source111/nativewire/bothSTORE/native32same/fullpeerrestored/finiteTPS="+str(16384/wall)+"/SLO="+str(slo["slo_all_pass"])+"/"+verdict,execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualAISBenchallattempts/nativewire/usage-length-DONE/sourceepochs/cost/SLO")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(verdict=verdict,TPS=16384/wall,SLO=slo["slo_all_pass"],native_counters=counters,pressure=peaks)))
