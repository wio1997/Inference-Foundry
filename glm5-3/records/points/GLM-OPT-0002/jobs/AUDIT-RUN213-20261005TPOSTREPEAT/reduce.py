from pathlib import Path
import sys,json,hashlib,subprocess,re,time,urllib.request,shlex
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process,process_identity
from protocol_receipts import verify_protocol_receipts
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0213";formal=r/"formal";bench=formal/"benchmark"
state=json.loads((r/"state.json").read_text())
assert state["status"]=="completed"and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for v in spec["stages"][0]["sources"]:
 assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
for stage in["prepare","dynamic"]:
 a=json.loads((r/(stage+".phase.json")).read_text());assert a["status"]=="succeeded"and a["exit_code"]==0and not a["timed_out"]
assert json.loads((r/"dynamic_wrapper_summary.json").read_text())["functional_acceptance"]
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
sys.path.insert(0,str(r.parent/"GLM-RUN-0211/runtime_bundle"))
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
config,_=checked_config(r/"service_config.json");roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());physical={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 ids={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);physical[key]=dict(root_same=True,actual_NPU16=True,root=v["root"])
 assert v["env"]["VLLM_PP_LAYER_PARTITION"]==("42,36"if o["host"]=="166"else"22,20,20,16")
 assert json.loads(o["argv"][o["argv"].index("--speculative-config")+1])["num_speculative_tokens"]==(3 if o["host"]=="166"else 1)
 for flag,value in[("--tensor-parallel-size","8"if o["host"]=="166"else"4"),("--pipeline-parallel-size","2"if o["host"]=="166"else"4"),("--decode-context-parallel-size","8"if o["host"]=="166"else"4"),("--nnodes","1")]:assert o["argv"][o["argv"].index(flag)+1]==value
 assert "--headless"not in o["argv"]and o["role"]=="API"
 physical[key]["static_K"]=3 if o["host"]=="166"else 1
 plugin="/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp204"if o["host"]=="166"else"local_pp211")
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(plugin)+");print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),source_sha256=hashlib.sha256((p/'issue_budget_scheduler_v5.py').read_bytes()).hexdigest())))"
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=30);z.check_returncode();v=json.loads(z.stdout);assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0204"if o["host"]=="166"else"GLM-COHORT-0211",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1);assert v["source_sha256"]==hashlib.sha256((p.parents[2]/"runtime/issue_budget_scheduler_v5.py").read_bytes()).hexdigest()
public=json.loads((r/"public_service_proof.json").read_text());host=public["host"];service=public["container"]
assert same_process(host)and [v.decode()for v in Path("/proc/"+str(host["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==public["argv"]
assert hashlib.sha256(Path(service["config"]["path"]).read_bytes()).hexdigest()==service["config"]["sha256"]and service["native_domains"]==config["native_domains"]
ports=subprocess.check_output(["ss","-ltnp"],text=True);rows=[x for x in ports.splitlines()if re.search(r":8000\s",x)];assert len(rows)==1and "pid="+str(host["pid"])+","in rows[0]
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as res:placement=json.loads(res.read())
assert len(placement["replicas"])==2and all(not x["temporarily_unhealthy"]and not x["group_faulted"]and not x["draining"]and x["active_requests"]==0and x["placement_policy"]=="shape_split_idle_spill"for x in placement["replicas"])
fault=json.loads((p/"runs/GLM-RUN-0125/response_owners.json.fault").read_text());assert fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
observer=observe(Path(service["config"]["path"]));assert all(x["status"]=="healthy"for x in observer["groups"])
obsowner=json.loads((r.parent/"GLM-RUN-0211/restored/identity_observer/process_owner.json").read_text());assert owner_alive(obsowner)
atomic_json(j/"public_service_proof.json",dict(public,at=utc(),placement=placement))
cursor=json.loads((r/"memo_trace_cursor.json").read_text());raw=Path(cursor["path"]).read_bytes()
# Public211 active trace: immutable prefix cursor beforewave; newevents uniqueoriginalbodySHA and lease timestamp.
assert len(raw)>=cursor["bytes"]and hashlib.sha256(raw[:cursor["bytes"]]).hexdigest()==cursor["sha256"]
(j/"scoped_token_memo_trace.jsonl").write_bytes(raw[cursor["bytes"]:])
memo_events=[json.loads(l)for l in raw[cursor["bytes"]:].decode().splitlines()]
from sse_observer import NativeSSEObserver
summary=json.loads((r/"arrival_summary.json").read_text());dynamic=json.loads((r/"dynamic_summary.json").read_text())
assert summary["functional_acceptance"]and summary["actual_new_requests"]==16and summary["effective_public_output_tokens"]==14208and not summary["errors"]
plan=json.loads((r/"arrival_plan.json").read_text());prior_plan=json.loads((p/"runs/GLM-RUN-0147/arrival_plan.json").read_text());cases=plan["cases"];rows=summary["requests"]
fields=["id","input_kind","arrival_s","expected_outputs","expected_prompt_tokens"]
assert [{k:c[k]for k in fields}for c in cases[:12]]==[{k:c[k]for k in fields}for c in prior_plan["cases"]]
assert [{k:c[k]for k in fields}for c in cases[12:]]==[dict(id="late"+str(i),input_kind="short",arrival_s=t,expected_outputs=1024,expected_prompt_tokens=21)for i,t in enumerate([55,55.1,60,60.1])]
trace=[json.loads(l)for l in(p/"runs/GLM-RUN-0125/router_trace.jsonl").read_text().splitlines()]
leases=[e for e in trace if e["event"]=="lease_acquired"and e.get("request_header_id")in{r.name+"-"+x["id"]for x in rows}]
assert len(leases)==16;ids={e["lease_id"]for e in leases};owned=[e for e in trace if e.get("lease_id")in ids];assert not any(e["event"].startswith("pd_")for e in owned)
compact=[];selected_memo=[]
compiled=json.loads(config["environment"]["GLM_REPLICAS"]);origin={x["id"]:x["url"]for x in compiled}
for row,case in zip(rows,cases):
 assert row["id"]==case["id"]and row["completed"]and row["http_status"]==200and row["effective_public_output_credit"]==case["expected_outputs"]
 assert row["usage"]==dict(prompt_tokens=case["expected_prompt_tokens"],completion_tokens=case["expected_outputs"],total_tokens=case["expected_prompt_tokens"]+case["expected_outputs"])
 wire=Path(row["wire"]["path"]);raw=wire.read_bytes();assert ref(wire)==row["wire"];obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True);obs.feed(raw)
 assert obs.contract()==row["contract"]and obs.done and not obs.contract_error and not obs.contract_unknown and obs.finish_reasons=={"0":"length"}
 tokens=[]
 for frame in raw.split(b"\n\n"):
  data=b"\n".join(l[5:].removeprefix(b" ")for l in frame.splitlines()if l.startswith(b"data:"))
  if not data or data==b"[DONE]":continue
  v=json.loads(data);assert not v.get("error")
  for ch in v.get("choices",[]):
   assert ch["index"]==0
   tok=ch.get("token_ids")or[];assert all(type(x)is int and x>=0for x in tok);tokens.extend(tok)
 assert tokens==row["committed_token_ids"]and len(tokens)==case["expected_outputs"]
 body=r/case["body"];assert ref(body)["sha256"]==row["body_sha256"]
 actual=json.loads(body.read_bytes());prior=json.loads((p/"runs/GLM-RUN-0147"/("short0.body.json"if row["id"].startswith("late")else case["body"])).read_bytes());assert actual.pop("cache_salt")==r.name+"-"+row["id"];prior.pop("cache_salt");assert actual==prior
 lease=next(x for x in leases if x["request_header_id"]==r.name+"-"+row["id"]);assert lease["body_sha256"]==row["body_sha256"]
 stream=[e for e in owned if e["event"]=="upstream_stream_contract"and e.get("lease_id")==lease["lease_id"]]
 release=[e for e in owned if e["event"]=="lease_released"and e.get("lease_id")==lease["lease_id"]]
 assert len(stream)==len(release)==1and stream[0]["audit_error"]is None and stream[0]["wire_sha256"]==row["wire"]["sha256"]and Path(stream[0]["wire_path"]).read_bytes()==raw and release[0]["released"]and not release[0]["backend_failure"]
 matches=[x for x in memo_events if x["event"]=="chat_token_cache_native_body"and x["native_origin"]==origin[lease["replica"]]and x["original_body_sha256"]==lease["body_sha256"]and x["monotonic_ns"]>=lease["monotonic_ns"]]
 assert len(matches)==1;entry=matches[0];assert entry["native_epoch"]==next(g["epoch"]for g in config["native_domains"]if lease["replica"]in g["members"])
 for key in["original_body","native_body"]:assert ref(Path(entry[key]["path"]))==entry[key]
 original=Path(entry["original_body"]["path"]).read_bytes();native=Path(entry["native_body"]["path"]).read_bytes();input_ids=json.loads(native)["kv_transfer_params"]["prompt_token_ids"]
 assert original==body.read_bytes()and native==original.rstrip()[:-1]+b',"kv_transfer_params":{"prompt_token_ids":'+json.dumps(input_ids,separators=(",",":")).encode()+b'}}'
 source=entry["tokenize_source"]
 assert all(ref(Path(source[k]["path"]))==source[k]for k in["request","response"])
 tok=json.loads(Path(source["response"]["path"]).read_bytes());assert tok["tokens"]==input_ids and tok["count"]==len(input_ids)==case["expected_prompt_tokens"]and source["epoch"]==entry["native_epoch"]and source["origin"]==entry["native_origin"]
 assert hashlib.sha256(json.dumps(input_ids,separators=(",",":")).encode()).hexdigest()==source["ids_sha256"]
 selected_memo.append(dict(lease_id=lease["lease_id"],native_owner=lease["replica"],cache_result=entry["cache_result"],prompt_tokens=len(input_ids),larger_body_bytes=len(native)-len(original),original=entry["original_body"],native=entry["native_body"],source=source))
 compact.append({k:v for k,v in row.items()if k not in["committed_token_ids","native_token_chunk_points","headers"]}|dict(native_owner=lease["replica"],committed_token_ids_count=len(tokens),committed_token_ids_sha256=hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest()))
acks=[json.loads(l)for l in(r/"native_dynamic.stdout").read_text().splitlines()if l.startswith('{"event":')]
assert [x["event"]for x in acks]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in acks)
exe=json.loads((r/"native_dynamic_execution.json").read_text());assert exe["status"]=="succeeded"and exe["exit_code"]==0and not exe["timed_out"]and exe["native_client_alive"]is False and exe["signal_attempts"]==[]and exe["native_client"]and not same_process(exe["native_client"])
for k in["stdout","stderr"]:assert ref(Path(exe[k]["path"]))==exe[k]
def metrics(f):
 out={}
 for line in f.read_text().splitlines():
  if not line or line.startswith("#"):continue
  k=line.split("{")[0].split()[0];out[k]=out.get(k,0)+float(line.rsplit(" ",1)[1])
 return out
deltas={};MTP={}
for key,node in[("D0","166"),("D1","167")]:
 before=metrics(r/("dynamic_initial_"+node+".metrics"));after=metrics(r/("epoch_final_"+node+".metrics"))
 assert after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==after["vllm:kv_cache_usage_perc"]==0
 delta={k:after[k]-before[k]for k in before if k.startswith("vllm:")and k.endswith("_total")};deltas[key]=delta
 mine=[x for x in compact if x["native_owner"]==key]
 assert delta["vllm:generation_tokens_total"]==sum(x["effective_public_output_credit"]for x in mine)and delta["vllm:request_success_total"]==len(mine)
 assert delta["vllm:prompt_tokens_total"]==sum(x["usage"]["prompt_tokens"]for x in mine)and delta["vllm:prefix_cache_hits_total"]==delta["vllm:num_preemptions_total"]==0
 d=delta["vllm:spec_decode_num_draft_tokens_total"];a=delta["vllm:spec_decode_num_accepted_tokens_total"];MTP[key]=dict(counters={k:v for k,v in delta.items()if k.startswith("vllm:spec_decode")},accepted_fraction=a/d if d else None)
assert sum(d["vllm:generation_tokens_total"]for d in deltas.values())==14208and sum(d["vllm:request_success_total"]for d in deltas.values())==16
from importlib.util import spec_from_file_location,module_from_spec
specmod=spec_from_file_location("retained_store",r/"retained_store.py");mod=module_from_spec(specmod);specmod.loader.exec_module(mod)
for native_id,source in mod.retained_responses():
 for stage in["before","after"]:assert(r/(native_id+"_"+stage+"_retained.wire")).read_bytes()==Path(source).read_bytes()
assert json.loads((r/"validation_native400.json").read_text())["status"]==400
samples=[json.loads(l)for l in(r/"native_samples.jsonl").read_text().splitlines()];assert len(samples)==summary["metrics_samples"]and not(r/"sample_error.json").exists()
assert all(x["placement_policy"]=="shape_split_idle_spill"for s in samples for x in s["placement"]["replicas"])
peak={key:{k:max(s["native"][key][k]for s in samples)for k in["num_requests_running","num_requests_waiting","kv_cache_usage_perc"]}for key in["D0","D1"]}
def percentile(values,p):
 vals=sorted(values);index=(len(vals)-1)*p/100;lo=int(index);hi=min(lo+1,len(vals)-1);return vals[lo]+(vals[hi]-vals[lo])*(index-lo)
latency={}
for kind in["all","long81932","medium8k","medium32k","short"]:
 rs=rows if kind=="all"else[x for x in rows if x["input_kind"]==kind]
 latency[kind]=dict(n=len(rs),ttft_ms={str(p):1000*percentile([x["ttft_s"]for x in rs],p)for p in[50,75,90,99]},mean_after_first_output_ms={str(p):1000*percentile([x["mean_after_first_output_s"]for x in rs],p)for p in[50,90]},max_chunk_gap_s=max(x["max_native_chunk_gap_s"]for x in rs))
assert latency==dynamic["latency"]
SLO={key:latency["all"][metric][str(p)]<bound for key,metric,p,bound in[
 ("TTFT_P50_lt_4000ms","ttft_ms",50,4000),("TTFT_P75_lt_8000ms","ttft_ms",75,8000),("TTFT_P90_lt_12000ms","ttft_ms",90,12000),("TTFT_P99_lt_30000ms","ttft_ms",99,30000),("mean_after_first_output_P50_lt_18ms","mean_after_first_output_ms",50,18),("mean_after_first_output_P90_lt_40ms","mean_after_first_output_ms",90,40)]}
assert SLO==dynamic["diagnostic_reference_SLO"]
assert config["placement"]==dict(kind="shape_split_idle_spill",input_threshold_bytes=32768,prefill_members=["D1"],decode_members=["D0"])
assert all(x["native_owner"]==("D0"if x["input_kind"]=="short"else"D1")for x in compact if not x["id"].startswith("late"))
spills=[x for x in compact if x["id"].startswith("late")and x["native_owner"]=="D1"];assert dynamic["short_spill_requests"]==[x["id"]for x in spills]
assert sum(x["effective_public_output_credit"]for x in compact if x["native_owner"]=="D1")==2560+1024*len(spills)
before=json.loads((r/"dynamic_memo_before.json").read_text());after=json.loads((r/"dynamic_memo_after.json").read_text())
memo_delta={k:after[k]-before[k]for k in["hits","misses","rewrites","bypasses","tokenizer_CPU_calls","tokenizer_CPU_wall_s","audit_errors","lookup_errors"]}
assert memo_delta["audit_errors"]==memo_delta["lookup_errors"]==0and after["pending"]==0and after["cache_bytes"]<=after["cache_byte_budget"]
atomic_json(j/"token_memo_stats_snapshot.json",after)
functional_path=p/"jobs/AUDIT-RUN211-20261005TPOSTFUNCTION/reduction.json"
functional=json.loads(functional_path.read_text());assert functional["measurement_valid"]and functional["functional_acceptance"]and functional["active_NPU_ranks"]==32and functional["D0_epoch_unchanged"]
for name in["standalone_root_identities.json","standalone_native_members.json"]:assert(r/name).read_bytes()==(p/"runs/GLM-RUN-0211/restored"/name).read_bytes()
assert {x["id"]:x["epoch"]for x in config["native_domains"]}==functional["new_native_epochs"]
assert len(spec["stages"][0]["sources"])==165
assert all(x["sources"]==spec["stages"][0]["sources"]for x in spec["stages"])
manifest=json.loads((r/"manifest.json").read_text());assert ref(r/"controller_spec.json")["sha256"]==manifest["spec"]["sha256"]==state["spec_sha256"]
wrapper=json.loads((r/"dynamic_wrapper_summary.json").read_text());assert wrapper["policies_unchanged"]and wrapper["private_policy_writes"]==wrapper["models_operations"]==wrapper["frontend_operations"]==0
capture_path=p/"runs/GLM-RUN-0211/restored/native_dense_capture_proof.json";capture=json.loads(capture_path.read_text())
assert capture["capture_count4_native_progress"]and capture["native_config"]==dict(cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=[2,4,8,16,32],max_cudagraph_capture_size=32)
assert ref(Path(capture["source"]["path"]))==capture["source"]
prior_path=p/"jobs/AUDIT-RUN212-20261005TPOSTLATE/reduction.json";prior=json.loads(prior_path.read_text());assert prior["measurement_valid"]and prior["effective_outputs"]==14208
prior_plan=json.loads((p/"runs/GLM-RUN-0210/arrival_plan.json").read_text())
assert [{k:c[k]for k in fields}for c in cases]==[{k:c[k]for k in fields}for c in prior_plan["cases"]]
policy_code="from pathlib import Path;import json;p=Path('/data/tiankuan/wio/glm52-pd/deploy/logs/local_pp211_167.log');print(json.dumps([json.loads(l.split('GLM_ISSUE_BUDGET_SELECTED ',1)[1])for l in p.read_text().splitlines()if 'GLM_ISSUE_BUDGET_SELECTED 'in l]))"
z=subprocess.run(["ssh","-o","BatchMode=yes","root@172.16.10.167","python3 -c "+shlex.quote(policy_code)],capture_output=True,timeout=60);(j/"native_D1_selected.stdout").write_bytes(z.stdout);z.check_returncode();selected=json.loads(z.stdout)
assert selected and all(x["cohort"]=="GLM-COHORT-0211"and x["serial"]==1and x["prefill_threshold_tokens"]==1024and x["prefill_cadence"]==1and x["budget_tokens"]==x["native_max"]==8192and not x["fallback"]for x in selected)
limits=["Exactresident repeat212late16/14208 fullnativeIDs-wire-counts-dynamicSDK0/native32/policiesepoch-STOREheld; no model/public/policyoperations in212","211actualGraph2fullCLI/resolver/noSP/nativecapture4/4 fit; SameD1physicalepoch-public lifetime held versus212, async-MTP-newcoldsalts/native trajectories confound repeated magnitude andisolated causal padding gain","HTTPstreammeanoutputinterval isnotnativeGPU pertoken time; repeatedQoS/stablecapacity/globalupperbound/actualruntimebatchdispatch unknown","Three STORE exact204D0-210D0-211D1; old200D1 unavailable per211audit epoch binding, no migration/replay","Actualshortspill observedonlyif availableleasepeerempty; publicleaseempty isnot GPU utilization/continuousidlecertificate; CurrentNone"]
artifacts=[ref(f)for f in sorted(r.iterdir())if f.is_file()and f.name not in["state.json","manifest.json","controller.log"]];atomic_json(j/"artifact_index.json",artifacts)
out=dict(at=utc(),run_id=r.name,measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE"if all(SLO.values())else"REJECT",source_count=165,Current=None,complete_requests=16,effective_outputs=14208,helper_outputs=0,uncredited_outputs=0,native32_same_owned_idle=True,physical=physical,policies_unchanged=True,private_policy_writes=0,model_operations=0,public_operations=0,actual_short_spill=[x["id"]for x in spills],functional211=ref(functional_path),full_client_SDK_init_finalize=acks,native_execution=ref(r/"native_dynamic_execution.json"),native_counters=deltas,native_D1_selected=selected,native_MTP=MTP,peak_native_pressure=peak,elapsed_s=summary["elapsed_s"],finite_effective_output_tps=14208/summary["elapsed_s"],requests=compact,latency=latency,diagnostic_reference_SLO=SLO,memo=dict(before=before,after=after,delta=memo_delta,body_proofs=selected_memo),public=ref(j/"public_service_proof.json"),retainedSTOREexact=True,artifact_index=ref(j/"artifact_index.json"),native_Graph2_capture=ref(capture_path),actual_runtime_batch_dispatch_unknown=True,prior212=dict(evidence=ref(prior_path),TPS=prior["finite_effective_output_tps"],latency=prior["latency"],SLO=prior["diagnostic_reference_SLO"],actual_short_spill=prior["actual_short_spill"]),limits=limits)
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="213 valid residentGraph2 same212repeat204D0-211D1 Graph2/late16new14208/fullnativeIDs-SDK-source165/native32/STORE3/policiesunchanged/finiteTPS="+str(out["finite_effective_output_tps"])+"/"+out["verdict"],execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="native16allIDs-wire-counts-SDK-epochs-STORE3-memo-SLO-MTP-pressure")],unknowns=limits,decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,verdict=out["verdict"],TPS=out["finite_effective_output_tps"],SLO=SLO,latency=latency,MTP=MTP,spill=out["actual_short_spill"])))
