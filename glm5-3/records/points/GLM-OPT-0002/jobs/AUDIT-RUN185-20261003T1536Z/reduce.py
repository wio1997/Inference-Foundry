from pathlib import Path
import sys,json,hashlib,subprocess,re,time,urllib.request,shlex
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process,process_identity
from protocol_receipts import verify_protocol_receipts
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0185";formal=r/"formal";bench=formal/"benchmark"
state=json.loads((r/"state.json").read_text())
assert state["status"]=="completed"and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for v in spec["stages"][0]["sources"]:
 assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
for stage in["prepare","experiment"]:
 a=json.loads((r/(stage+".phase.json")).read_text());assert a["status"]=="succeeded"and a["exit_code"]==0and not a["timed_out"]
assert all(json.loads((r/(window_name+"_wrapper_summary.json")).read_text())["functional_acceptance"]for window_name in["formal"])
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
config,_=checked_config(r/"service_config.json");roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());physical={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 ids={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);physical[key]=dict(root_same=True,actual_NPU16=True,root=v["root"])
 assert v["env"]["VLLM_PP_LAYER_PARTITION"]=="22,20,20,16"
 assert json.loads(o["argv"][o["argv"].index("--speculative-config")+1])["num_speculative_tokens"]==(3 if key=="D0"else 1)
 physical[key]["static_K"]=3 if key=="D0"else 1
 plugin="/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp175"if o["host"]=="166"else"local_pp184")
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(plugin)+");print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),source_sha256=hashlib.sha256((p/'issue_budget_scheduler_v3.py').read_bytes()).hexdigest())))"
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=30);z.check_returncode();v=json.loads(z.stdout);assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3 if o["host"]=="166"else 1);assert v["source_sha256"]==hashlib.sha256((p.parents[2]/"runtime/issue_budget_scheduler_v3.py").read_bytes()).hexdigest()
public=json.loads((r/"public_service_proof.json").read_text());host=public["host"];service=public["container"]
assert same_process(host)and [v.decode()for v in Path("/proc/"+str(host["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==public["argv"]
assert hashlib.sha256(Path(service["config"]["path"]).read_bytes()).hexdigest()==service["config"]["sha256"]and service["native_domains"]==config["native_domains"]
ports=subprocess.check_output(["ss","-ltnp"],text=True);rows=[x for x in ports.splitlines()if re.search(r":8000\s",x)];assert len(rows)==1and "pid="+str(host["pid"])+","in rows[0]
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as res:placement=json.loads(res.read())
assert len(placement["replicas"])==2and all(not x["temporarily_unhealthy"]and not x["group_faulted"]and not x["draining"]and x["active_requests"]==0and x["work_ranking_calibrated"]for x in placement["replicas"])
fault=json.loads((p/"runs/GLM-RUN-0125/response_owners.json.fault").read_text());assert fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
observer=observe(Path(service["config"]["path"]));assert all(x["status"]=="healthy"for x in observer["groups"])
obsowner=json.loads((p/"runs/GLM-RUN-0184/restored/identity_observer/process_owner.json").read_text());assert owner_alive(obsowner)
atomic_json(j/"public_service_proof.json",dict(public,at=utc(),placement=placement))
cursor=json.loads((r/"memo_trace_cursor.json").read_text());raw=Path(cursor["path"]).read_bytes()
assert len(raw)>=cursor["bytes"]and hashlib.sha256(raw[:cursor["bytes"]]).hexdigest()==cursor["sha256"]
(j/"scoped_token_memo_trace.jsonl").write_bytes(raw[cursor["bytes"]:])
memo_events=[json.loads(l)for l in raw[cursor["bytes"]:].decode().splitlines()]
def audit_window(window_name):
 formal=r/window_name;bench=formal/"benchmark"
 full=json.loads((bench/"full_execution.json").read_text());warm=json.loads((bench/"warmup_execution.json").read_text());assert full["status"]==warm["status"]=="succeeded"and full["exit_code"]==warm["exit_code"]==0
 checks={}
 for phase,count,inputt,outputt in[("warmup",2,73740,1),("full",4,81932,61440)]:
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
 SALT="GLM-RUN-0185-dual-PP4-K3-K1-memo-four61440-c4-t1024-"+window_name
 for phase in["warmup","full"]:
  code=(bench/(phase+"_temp_api.py")).read_text();assert "cache_salt="+repr(SALT)in code and "ignore_eos=True"in code and "temperature=0"in code
 events=json.loads((formal/"formal_events.json").read_text());assert len([x for x in events if x["event"]=="controlled_count_fallback_both_domains"])==1and len([x for x in events if x["event"]=="D0_full_peer_restored"])==1
 for phase,each in[("warmup",1),("full",2)]:assert {k:sum(x["replica"]==k for x in checks[phase]["receipts"]["requests"])for k in["D0","D1"]}==dict(D0=each,D1=each)
 command=json.loads((formal/"benchmark_command.json").read_text())["argv"]
 for option,value in[("--input_len","81920"),("--output_len","61440"),("--data_num","4"),("--concurrency","4"),("--request_rate","0"),("--repeat_rate","0.9"),("--prefix_num","1"),("--seed","20260930"),("--host_port","8000"),("--npu_num","32"),("--dp","2")]:assert command[command.index(option)+1]==value
 assert command[command.index("--pod_info")+1:command.index("--pod_info")+3]==["172.16.10.166:9081","172.16.10.167:9900"]
 assert not list(Path(checks["full"]["details_path"]).parent.glob("request_scope_amendment.json"))
 details=Path(checks["full"]["details_path"]);perf=details.with_name("gsm8k.csv");assert perf.exists()
 argv=["python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/aisbench_slo.py","--perf_csv",str(perf),"--details_jsonl",str(details),"--expect_output_len","61440","--concurrency","4","--expected_requests","4","--out_json",str(j/(window_name+"_slo.json")),"--out_md",str(j/(window_name+"_slo.md"))]
 z=subprocess.run(argv,capture_output=True,timeout=30);(j/(window_name+"_slo.stdout")).write_bytes(z.stdout);(j/(window_name+"_slo.stderr")).write_bytes(z.stderr);z.check_returncode();slo=json.loads((j/(window_name+"_slo.json")).read_text());assert slo["all_requests_succeeded"]and slo["output_len_ok"]and slo["n_success"]==4
 acks=[json.loads(x)for x in(r/(window_name+"_native_formal.stdout")).read_text().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
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
  assert delta["vllm:request_success_total"]==3and 0<=delta["vllm:prefix_cache_hits_total"]<=237604and delta["vllm:prefix_cache_queries_total"]==237604and delta["vllm:generation_tokens_total"]==122881and delta["vllm:prompt_tokens_total"]==237604
  counters[node]=dict(delta=delta,idle=True,uncredited_output_cost=0)
 peaks={}
 for node in["D0","D1"]:
  files=list(formal.glob("sample*_"+node+".metrics"));assert files
  vals=[metric(f)for f in files];peaks[node]={k:max(v.get(k,0)for v in vals)for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]}
  # 10s samples only observed pressure, no GPU per-step occupancy proof.
 wall=(datetime.fromisoformat(full["finished_at"])-datetime.fromisoformat(full["started_at"])).total_seconds()
 warmwall=(datetime.fromisoformat(warm["finished_at"])-datetime.fromisoformat(warm["started_at"])).total_seconds()
 out=dict(name=window_name,checks=checks,SLO=slo,measurement_valid=True,native_counters=counters,native_pressure=peaks,
  full_cli_phase_wall_s=wall,effective_TPS_full_cli=245760/wall,warmup_cli_wall_s=warmwall,
  completed_full_requests=4,complete_outputs=245760,warmup_outputs=2,total_native_outputs=245762,
  actualfull_requests_perdomain=dict(D0=2,D1=2),cache_salt=SALT,
  actual_client_SDK_init_finalize=[acks[0],acks[-1]],cost_includes_HTTP_CPUlookup_and_body_growth=True)
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
 out["full_native_MTP"]={}
 for key in["D0","D1"]:
  warmfile=Path(out["native_full_histogram_means"][key]["warm_excluded_sample"]["path"]);w=hist(warmfile);e=hist(formal/("final_"+key+".metrics"))
  delta={k:e[k]-v for k,v in w.items()if k.startswith("vllm:spec_decode_")and k.endswith("_total")}
  d=delta["vllm:spec_decode_num_draft_tokens_total"];accepted=delta["vllm:spec_decode_num_accepted_tokens_total"];n=delta["vllm:spec_decode_num_drafts_total"];assert d>0and n>0and 0<=accepted<=d
  out["full_native_MTP"][key]=dict(counters=delta,accepted_fraction=accepted/d,accepted_per_draft=accepted/n,warm_excluded=ref(warmfile),final_metrics=ref(formal/("final_"+key+".metrics")),limits=["Native speculative counters do not count physical GPU steps"])
 fallback=next(x for x in events if x["event"]=="controlled_count_fallback_both_domains")
 assert all(not x["work_ranking_calibrated"]and not x["active_requests"]and not x["group_faulted"]for x in fallback["placement"]["replicas"])
 peer=fallback["full_original_D0_peer"];assert fallback["full_fallback_D0_peer"]=={k:v for k,v in peer.items()if k not in["decode_tps","prefill_bytes_per_s"]}
 compiled=json.loads(config["environment"]["GLM_REPLICAS"]);assert peer==next(x for x in compiled if x["id"]=="D0")
 for run,requestname,nativeid in[("GLM-RUN-0175","newD0_create","resp_glm_run175_D0_new"),("GLM-RUN-0184","newD1_create","resp_glm_run184_D1_new")]:
  c=json.loads((p/"runs"/run/"client_final_summary.json").read_text());base=next(x for x in c["requests"]if x["name"]==requestname)
  assert(formal/(nativeid+"_retained.wire")).read_bytes()==Path(base["wire"]["path"]).read_bytes()
 memo_before=json.loads((r/(window_name+"_memo_before.json")).read_text());memo_after=json.loads((r/(window_name+"_memo_after.json")).read_text())
 diff={k:memo_after[k]-memo_before[k]for k in["hits","misses","singleflight_waits","rewrites","bypasses","tokenizer_CPU_calls","tokenizer_CPU_wall_s","audit_errors","lookup_errors"]}
 assert diff["rewrites"]==6 and diff["audit_errors"]==diff["lookup_errors"]==diff["singleflight_waits"]==0
 assert memo_after["pending"]==0and memo_after["cache_bytes"]<=memo_after["cache_byte_budget"]
 origin={x["id"]:x["url"]for x in compiled}
 selected=[]
 for lease in leases:
  matches=[x for x in memo_events if x["event"]=="chat_token_cache_native_body"and x["native_origin"]==origin[lease["replica"]]and x["original_body_sha256"]==lease["body_sha256"]and x["monotonic_ns"]>=lease["monotonic_ns"]]
  assert len(matches)==1;entry=matches[0]
  assert entry["native_epoch"]==next(g["epoch"]for g in config["native_domains"]if lease["replica"]in g["members"])
  for key in["original_body","native_body"]:
   v=entry[key];assert ref(Path(v["path"]))==v
  original=Path(entry["original_body"]["path"]).read_bytes();native=Path(entry["native_body"]["path"]).read_bytes()
  ids=json.loads(native)["kv_transfer_params"]["prompt_token_ids"]
  # Every original raw member byte remains in place, only append validated CPU metadata.
  assert native==original.rstrip()[:-1]+b',"kv_transfer_params":{"prompt_token_ids":'+json.dumps(ids,separators=(",",":")).encode()+b'}}'
  payload=json.loads(native);payload.pop("kv_transfer_params");assert payload==json.loads(original)
  assert entry["native_prompt_tokens"]==len(ids)in[73740,81932]
  source=entry["tokenize_source"]
  assert ref(Path(source["request"]["path"]))==source["request"]and ref(Path(source["response"]["path"]))==source["response"]
  tok=json.loads(Path(source["response"]["path"]).read_bytes());assert tok["count"]==len(ids)and tok["tokens"]==ids
  assert hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()==source["ids_sha256"]
  assert source["epoch"]==entry["native_epoch"]and source["origin"]==entry["native_origin"]
  selected.append(dict(lease_id=lease["lease_id"],replica=lease["replica"],original_body_sha256=entry["original_body_sha256"],
                      transformed_native_body_sha256=entry["native_body_sha256"],memo_result=entry["cache_result"],native_prompt_tokens=len(ids),
                      larger_body_bytes=len(native)-len(original),tokenize_source=source))
 assert len(selected)==6and sum(x["memo_result"]=="hit"for x in selected)==diff["hits"]
 if window_name=="memo_cold":assert diff["misses"]==diff["tokenizer_CPU_calls"]==6and diff["hits"]==0
 # Actual backend assignment decides cache reuse; do not assume all hits before checking.
 out["memo"]=dict(before=memo_before,after=memo_after,delta=diff,native_body_proofs=selected,
                 preprocessing_cost_inside_request=True,cache_salt_is_KV_isolation_not_CPU_tokenization_key=True)
 out["verdict"]="INCONCLUSIVE"if slo["slo_all_pass"]else"REJECT"
 return out
execution=json.loads((r/"formal_native_client_execution.json").read_text())
assert execution["status"]=="succeeded"and execution["exit_code"]==0and not execution["timed_out"]and execution["native_client_alive"]is False and execution["signal_attempts"]==[]and execution["native_client"]
assert not same_process(execution["native_client"])
for field in ["stdout","stderr"]:assert ref(Path(execution[field]["path"]))==execution[field]
cpu=json.loads((r/"SDK_CPU_acceptance.json").read_text());assert cpu["actual_Docker_owned_client_capture"]and cpu["new_inference"]==cpu["models_or_tensors_created"]==0and cpu["no_Docker_cancellation_test_claim"]
assert [v["event"]for v in cpu["SDK_init_finalize0"]]==["task_acl_init","task_acl_finalize"]and all(v["returncode"]==0for v in cpu["SDK_init_finalize0"])
windows={"formal":audit_window("formal")};w=windows["formal"]
restore=json.loads((r/"restore_summary.json").read_text());assert restore["private_policy_writes"]==0and restore["native32_same_idle"]and restore["model_operations"]==0
assert restore["D0_policy_unchanged"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3)
assert restore["D1_policy_unchanged"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
selected=json.loads((r/"policy_consumption.json").read_text());assert selected["operator_math_changes"]==0and selected["allocated8192_unchanged"]and selected["D0_policy_unchanged"]
for node,serial in[("166",3),("167",1)]:
 values=selected["native_selected"][node];assert values and all(v["serial"]==serial and v["budget_tokens"]==8192and v["prefill_threshold_tokens"]==1024and not v["fallback"]and v["native_max"]==8192for v in values)
stats=json.loads((p/"runs/GLM-RUN-0184/token_memo_stats.json").read_text());assert stats["pending"]==0and stats["cache_bytes"]<=stats["cache_byte_budget"]and stats["audit_errors"]==stats["lookup_errors"]==0
atomic_json(j/"token_memo_stats_snapshot.json",stats)
assert public["container_marker"]["gateway_version"]=="V12"
def metric_all(f):
 d={}
 for l in f.read_text().splitlines():
  if l and not l.startswith("#"):
   k=l.split("{")[0].split()[0];d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
 return d
total={}
for key,node in[("D0","166"),("D1","167")]:
 before=metric_all(r/("prepare_"+node+".metrics"));after=metric_all(r/("epoch_terminal_"+node+".metrics"))
 delta={k:after[k]-before[k] for k in w["native_counters"][key]["delta"]}
 assert delta["vllm:generation_tokens_total"]==122881and delta["vllm:request_success_total"]==3and delta["vllm:num_preemptions_total"]==0
 assert delta==w["native_counters"][key]["delta"]
 total[key]=delta
prior={}
for run,job,sha in[("179","AUDIT-RUN179V3-20261003T1342Z","242822259e3b113d49f1764f65a2c2c8233aad1df921a8c5f44474ac0e59bb19"),("152","AUDIT-RUN152V2-20261003T0616Z","8afc89d4dd68c8d9039a863bfd59935c6e5c3362b0235acf363092326a9324b6")]:
 f=p/"jobs"/job/"reduction.json"
 # Actual Run152 V2 directory can be found by its SHA, rather than guessed timestamp.
 if not f.exists():
  found=[x for x in(p/"jobs").glob("AUDIT-RUN152*/reduction.json")if ref(x)["sha256"]==sha];assert len(found)==1;f=found[0]
 assert ref(f)["sha256"]==sha;v=json.loads(f.read_text())
 prior[run]=dict(audit=ref(f),limits=v["limits"],
   SLO=v["SLO"]if "SLO"in v else{k:x["SLO"]for k,x in v["windows"].items()},
   full_cli_TPS=v.get("effective_TPS_full_cli"),explicit_class_differences=["179samePP4/V12/memo/routing but4Koutput; notmatchedlongdecode",
    "152samefullinput152/output61440/count2each, differentPP2TP8DCP8/KVblock1024/privatebudget4096/t1024/V11/nativeepochs/history/cache72704 vscurrentPP4TP4DCP4/block512/cache73216; notisolatedcodegain"])
artifacts=[ref(f)for f in sorted(r.rglob("*"))if f.is_file()and f.name not in["state.json","manifest.json","controller.log","monitor_samples.jsonl"]]
atomic_json(j/"artifact_index.json",artifacts)
verdict=w["verdict"]
out=dict(at=utc(),run_id=r.name,kind="two_PP4_domains_full61440_c4_countfallback_HTTP_tokenmemo_finite_window",
 measurement_valid=True,functional_acceptance=True,verdict=verdict,
 source_pins=len(spec["stages"][0]["sources"]),native32_same_owned_idle=True,physical_same=physical,windows=windows,SLO=w["SLO"],
 effective_TPS_full_cli=w["effective_TPS_full_cli"],full_cli_phase_wall_s=w["full_cli_phase_wall_s"],
 complete_outputs=245760,warmup_outputs=2,total_native_outputs=245762,total_native_complete_requests=6,
 total_native_counters=total,uncredited_output_tokens=0,model_operations=0,gateway_operations=0,
 public184_same=True,retainedpublic_SDKinit0=True,memo_stats=stats,restore=restore,policy_consumption=selected,
 native_client_execution=ref(r/"formal_native_client_execution.json"),actual_Docker_CPU_SDK_probe=ref(r/"SDK_CPU_acceptance.json"),public=ref(j/"public_service_proof.json"),artifact_index=ref(j/"artifact_index.json"),prior_windows=prior,
 limits=["All4 full61440 native outputs/81932input/c4/count2each+2warm, no helper/preempt/uncredited, SDK0/nativewire/transform/source32/STORE",
 "Current184resident CPUmemo hits/misses determined by actualrouting, no tokenpreload; ordinaryHTTP lookup and largerbody costs fullycharged",
 "Finite N4 closedarrival full143372 context does not prove stable/open/variableload capacity or robusttail/globalhardwareupperbound/KEEP",
 "Comparison152PP2 changes topology/DCPblock/KV/policy/epochs/publicV12/memo/nativeMTP trajectory; noisolatedcausalgain",
 "Different prompt/domain/newD1 epoch/KV and MTP trajectories: K3vsK1 not isolated K-gain; nativeDynamicSpec+DCP4 guard preserved",
 "Nativehistograms/MTP/front iteration counts are not real GPU SchedulerOutput/step occupancy"])
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",
 summary="185 same184V12/native32 full4x61440/count2each/t1024/native245762/no uncredited/SDK-source-nativewire-transform/STORE/source101/policiesunchangedD0serial3-D1serial1/finiteTPS="+str(w["effective_TPS_full_cli"])+"/SLO="+str(w["SLO"]["slo_all_pass"])+"/"+verdict,
 execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="completefull61440/ordinaryHTTPmemo/nativewire-SDK-counts-source32/STORE/twoeach/SLO")],
 unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,verdict=verdict,TPS=w["effective_TPS_full_cli"],SLO=w["SLO"],memo=w["memo"]["delta"],evidence=e)))
