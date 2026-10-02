from pathlib import Path
import json,sys,time,hashlib,re,subprocess,shlex,urllib.request,statistics
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0115";deadline=time.monotonic()+2000
while True:
 state=json.loads((r/"state.json").read_text())
 if state["status"]!="running":break
 assert same_process(state["owner"])and time.monotonic()<deadline;time.sleep(10)
assert state["status"]=="completed"and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for v in spec["stages"][0]["sources"]:
 b=Path(v["path"]).read_bytes();assert b==Path(v["snapshot"]).read_bytes()and hashlib.sha256(b).hexdigest()==v["sha256"]
for name in["prepare","dynamic"]:
 a=json.loads((r/(name+".phase.json")).read_text());assert a["status"]=="succeeded"and a["exit_code"]==0 and not a["timed_out"]
a=json.loads((r/"dynamic_summary.json").read_text());plan=json.loads((r/"arrival_plan.json").read_text());by={x["id"]:x for x in plan["cases"]};assert a["functional_acceptance"]and len(a["requests"])==12 and a["effective_public_output_tokens"]==10112 and not(r/"sample_error.json").exists()
trace=[json.loads(x)for x in(r/"router_trace.jsonl").read_text().splitlines()];rows=[]
for v in a["requests"]:
 c=by[v["id"]];wire=Path(v["wire"]["path"]);raw=wire.read_bytes();assert len(raw)==v["wire"]["bytes"]and hashlib.sha256(raw).hexdigest()==v["wire"]["sha256"]
 body=(r/c["body"]).read_bytes();value=json.loads(body);assert value["cache_salt"]==r.name+"-"+v["id"]and value["max_tokens"]==c["expected_outputs"]and not value.get("kv_transfer_params")and value["return_token_ids"]is True
 obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True)
 for n in range(0,len(raw),733):obs.feed(raw[n:n+733])
 con=obs.contract();assert con["done"]and not con["unknown"]and not con["native_error"]and con["usage"]==v["usage"]and con["finish_reasons"]=={"0":"length"}
 frames=[]
 for part in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
  data=b"\n".join(l[5:].removeprefix(b" ")for l in part.splitlines()if l.startswith(b"data:"))
  if data and data!=b"[DONE]":frames.append(json.loads(data))
 ids=[t for f in frames for ch in f.get("choices",[])for t in ch.get("token_ids")or[]];assert ids==v["committed_token_ids"]and len(ids)==c["expected_outputs"]
 prompts=[f["prompt_token_ids"]for f in frames if f.get("prompt_token_ids")is not None]
 assert prompts and len(prompts[0])==c["expected_prompt_tokens"]and v["usage"]["prompt_tokens"]==c["expected_prompt_tokens"]
 pts=v["native_token_chunk_points"];assert pts and pts[-1]["committed_token_count"]==len(ids)and all(pts[k]["elapsed_s"]>=pts[k-1]["elapsed_s"]and pts[k]["committed_token_count"]>pts[k-1]["committed_token_count"]for k in range(1,len(pts)))
 leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==r.name+"-"+v["id"]];assert len(leases)==1;l=leases[0];assert l["replica"]==v["native_owner"]and l["body_sha256"]==hashlib.sha256(body).hexdigest()
 ss=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==l["lease_id"]];assert len(ss)==1 and ss[0]["audit_error"]is None and Path(ss[0]["wire_path"]).read_bytes()==raw
 releases=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==l["lease_id"]];assert len(releases)==1 and releases[0]["released"]and not releases[0]["backend_failure"]
 assert abs(v["actual_dispatch_s"]-c["arrival_s"])<.1
 row={k:v[k]for k in["id","input_kind","expected_prompt_tokens","expected_outputs","native_owner","actual_dispatch_s","ttft_s","wall_s","mean_committed_arrival_interval_s","mean_after_first_output_s","max_native_chunk_gap_s","usage","wire"]}
 row["gateway_observer_unknown"]=ss[0]["contract"]["unknown"];rows.append(row)
tokenized=json.loads((r/"native_tokenized_counts.json").read_text());assert tokenized=={"long81932":81932,"medium8k":7875,"medium32k":31416,"short":21}
for kind,count in tokenized.items():
 v=json.loads((r/("tokenize_"+kind+".wire")).read_text());assert v["count"]==len(v["tokens"])==count
samples=[json.loads(x)for x in(r/"native_samples.jsonl").read_text().splitlines()];assert len(samples)>=20 and samples[-1]["elapsed_s"]>=a["elapsed_s"]-5 and all(samples[k]["elapsed_s"]>=samples[k-1]["elapsed_s"]for k in range(1,len(samples)))
calibration=None
assert not any(x["event"].startswith("pd_helper_")for x in trace)
for sample in samples:
 for row in sample["placement"]["replicas"]:
  assert row["placement_policy"]=="shape_split"and row["shape_split_hint"]==dict(input_threshold_bytes=8192,prefill_members=["D1"],decode_members=["D0"])and row["decode_tps_hint"]is None and row["prefill_bytes_per_s_hint"]is None
  assert row["reserved_work_seconds_hint"]is None
pressure=dict(samples=len(samples),peak_native_running={k:max(x["native"][k]["num_requests_running"]for x in samples)for k in["D0","D1"]},peak_native_waiting={k:max(x["native"][k]["num_requests_waiting"]for x in samples)for k in["D0","D1"]},peak_KV={k:max(x["native"][k]["kv_cache_usage_perc"]for x in samples)for k in["D0","D1"]},peak_HTTP_leases=max(sum(x["active_requests"]for x in s["placement"]["replicas"])for s in samples))
def metrics(f):
 out={}
 for l in f.read_text().splitlines():
  if not l or l.startswith("#"):continue
  name=l.split("{")[0].split()[0]
  if any(v in name for v in["prefix_cache_hits_total","prefix_cache_queries_total","prompt_tokens_total","generation_tokens_total","request_success_total","num_preemptions_total","num_requests_running","num_requests_waiting","kv_cache_usage_perc"]):out[name]=out.get(name,0)+float(l.rsplit(" ",1)[1])
 return out
counters={}
for key in["D0","D1"]:
 before=metrics(r/("before_0_"+key+".metrics"));files=sorted(r.glob("after_*_"+key+".metrics"),key=lambda f:int(f.name.split("_")[1]));assert files;after=metrics(files[-1]);delta={k:after.get(k,0)-v for k,v in before.items()if k.endswith("_total")}
 outputs=sum(x["usage"]["completion_tokens"]for x in rows if x["native_owner"]==key);prompts=sum(x["usage"]["prompt_tokens"]for x in rows if x["native_owner"]==key);requests=sum(x["native_owner"]==key for x in rows)
 assert (outputs,requests)==((7552,7)if key=="D0"else(2560,5))
 assert delta.get("vllm:generation_tokens_total")==outputs and delta.get("vllm:prompt_tokens_total")==prompts and delta.get("vllm:request_success_total")==requests
 assert delta.get("vllm:prefix_cache_hits_total",0)==delta.get("vllm:num_preemptions_total",0)==0 and delta.get("vllm:prefix_cache_queries_total",0)>=prompts
 assert delta.get("vllm:external_prefix_cache_hits_total",0)==0
 assert all(after.get(k,0)==0 for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"])
 counters[key]=dict(delta=delta,idle=True,cold_verified=True,requests=requests)
API=json.loads((r/"adopted_model_identities.json").read_text());members=json.loads((r/"native_member_identities.json").read_text());physical={}
check="""import pathlib,json,sys,subprocess,re
a=json.load(sys.stdin);o=a['owner'];boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for x in a['targets']+[dict(pid=o['pid'],identity=o['identity'])]:
 p=pathlib.Path('/proc/'+str(x['pid']));b=(p/'stat').read_text();v=b[b.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity']
assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);ids={int(v)for v in re.findall(r'^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker',raw,re.M)};assert ids=={x['pid']for x in a['targets']}
policy=json.loads(pathlib.Path(a['policy_path']).read_text());assert policy==a['policy'];print(json.dumps(dict(API_same=True,NPU16_same=True,npu_pids=sorted(ids),policy=policy)))
"""
policy=dict(schema_version=1,cohort_id="GLM-COHORT-0089",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
for key,o in API.items():
 argv=o["argv"];assert json.loads(argv[argv.index("--speculative-config")+1])["num_speculative_tokens"]==(5 if key=="D0"else 3)
 assert json.loads(argv[argv.index("--compilation-config")+1])["cudagraph_capture_sizes"]==([6,12,24,48]if key=="D0"else[4,8,16,32])
 policy=dict(policy,cohort_id="GLM-COHORT-0089"if o["host"]=="166"else"GLM-COHORT-0109",serial=3 if o["host"]=="166"else 3)
 args=["python3","-c",check]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 targets=[v for v in members[key]["owned_targets"]if v["pid"]in members[key]["npu_worker_pids"]]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,targets=targets,policy=policy,policy_path=str(Path(o["argv"][1]).parent/"issue_budget_policy.json"))).encode(),capture_output=True,timeout=90);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();physical[key]=json.loads(z.stdout)
ack=[json.loads(x)for x in(r/"native_dynamic.stdout").read_text().splitlines()if x.startswith('{"event":')];assert ack[0]["event"]=="task_acl_init"and ack[-1]["event"]=="task_acl_finalize"and ack[0]["returncode"]==ack[-1]["returncode"]==0
cleanup=json.loads((r/"gateway_cleanup.json").read_text());assert cleanup["exit_code"]in[0,-15]
fault=json.loads((r/"response_owners.json.fault").read_text());assert not fault["open"]and all(not x["faulted"]for x in fault["groups"].values())
assert not re.search(r":8002\s",subprocess.check_output(["ss","-ltnp"],text=True))

def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
PD=[x for x in trace if x["event"].startswith("pd_")]
assert len(PD)==9 and all(x["event"]=="pd_geometry_native_fallback"and x["internal_output_tokens"]==0 for x in PD)
assert not list((r/"native_PD_raw").glob("*"))
assert all(x["native_owner"]==("D0"if x["input_kind"]=="short"else"D1")for x in rows)
bad=json.loads((r/"validation_native400.json").read_text());assert bad["status"]==400 and bad["public_credit"]==0 and hashlib.sha256((r/"validation_temperature_negative.wire").read_bytes()).hexdigest()==bad["wire_sha256"]
negative=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==r.name+"-validation-negative-temperature"]
assert len(negative)==1 and negative[0]["body_sha256"]==bad["body_sha256"]and negative[0]["replica"]=="D1"
ns=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==negative[0]["lease_id"]];nh=[x for x in trace if x["event"]=="upstream_headers"and x.get("lease_id")==negative[0]["lease_id"]]
assert len(ns)==len(nh)==1 and nh[0]["status"]==400 and Path(ns[0]["wire_path"]).read_bytes()==(r/"validation_temperature_negative.wire").read_bytes()
assert sum(x["delta"]["vllm:generation_tokens_total"]for x in counters.values())==10112 and sum(x["delta"]["vllm:request_success_total"]for x in counters.values())==12
for key,o in API.items():
 argv=o["argv"]
 assert int(argv[argv.index("--tensor-parallel-size")+1])==(16 if key=="D0"else 8)
 assert int(argv[argv.index("--decode-context-parallel-size")+1])==(16 if key=="D0"else 8)
 if key=="D1":
  assert int(argv[argv.index("--pipeline-parallel-size")+1])==2 and "--enable-expert-parallel"not in argv and "--kv-transfer-config"not in argv
  assert o["pp_layer_partition"]==[42,36]and o["expert_parallel_enabled"]is False
helpers=[]

selected=json.loads((r/"D109_budget2048_threshold256_ack.json").read_text())
restored=json.loads((r/"restore_D109_budget4096_threshold1024_ack.json").read_text())
original=dict(schema_version=1,cohort_id="GLM-COHORT-0109",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
during=dict(original,budget_tokens=2048,prefill_threshold_tokens=256,serial=2)
after=dict(original,serial=3)
assert selected["previous"]==original and selected["new"]==during and restored["previous"]==during and restored["new"]==after
assert json.loads((r/"expected_D_policy.json").read_text())==after
for event in[selected,restored]:
 assert set(event["acks"])=={"D1"}and event["model_operations"]==0 and event["onlyD109_independent_idle_atomic_update"]is True
 assert event["acks"]["D1"]["atomic_replace"]and event["acks"]["D1"]["new"]==event["new"]
assert json.loads((r/"epoch_selected_D1.stdout").read_text())["policy"]==during
assert json.loads((r/"epoch_restored_D1.stdout").read_text())["policy"]==after
planD=next(x for x in json.loads((r/"planned_launch.json").read_text())if x["node"]=="167")
code="import pathlib;print(''.join(l for l in pathlib.Path("+repr(planD["log"])+").read_text().splitlines(True)if 'GLM_ISSUE_BUDGET_SELECTED' in l))"
z=subprocess.run(["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["python3","-c",code])],capture_output=True,timeout=90);(j/"native_D_budget_selected.stdout").write_bytes(z.stdout);(j/"native_D_budget_selected.stderr").write_bytes(z.stderr);z.check_returncode()
nativeSelected=[]
for line in z.stdout.decode().splitlines():
 if "GLM_ISSUE_BUDGET_SELECTED"not in line:continue
 part=line.split("GLM_ISSUE_BUDGET_SELECTED",1)[1].strip()
 try:value=json.loads(part)
 except ValueError:continue
 if value.get("cohort")=="GLM-COHORT-0109"and value.get("serial")==2:
  assert value["budget_tokens"]==2048 and value["prefill_threshold_tokens"]==256 and value["prefill_cadence"]==1 and value["policy_error"]is None and value["fallback"]is False
  nativeSelected.append(value)
assert nativeSelected

out=dict(D109_selected_policy=during,D109_restored_policy=after,native_D109_policy_observed=nativeSelected,policy_acknowledgements=[selected,restored],native_PD_helpers=helpers,native_geometry_fallback_events=PD,internal_helper_commits=0,total_native_commits=10112,public_native_consumer="P89short+D109large",negative_temperature_native400_preserved=True,full_helper_raw_complete=True,calibration=calibration,at=utc(),functional_acceptance=True,verdict="INCONCLUSIVE",actual_new_requests=12,effective_public_output_tokens=10112,elapsed_s=a["elapsed_s"],finite_effective_output_tps=a["finite_effective_output_tps"],native_tokenized_counts=tokenized,requests=rows,placement_policy='shape_split',latency=a["latency"],diagnostic_reference_SLO=a["diagnostic_reference_SLO"],pressure=pressure,counters=counters,physical_same=physical,pins=len(spec["stages"][0]["sources"]),model_operations=0,gateway_port8002_free=True,limits=a["limits"]+["P89 local7short output7552 plus D109 local5large output2560 TP8PP2DCP8/42,36/EPdisabledAllGather/MTP3/noSP/Graph4,8,16,32/noKV and nativepipelinebatchqueue; nativeD hotpolicy2048t256c1 duringwindow, restored4096t1024serial3 afteridle; GPUallocation4096 unchanged; P89epoch retained and executinglocalshorts. Same106 twelve NEWcold variablearrival/context/output workpackage, but TP/DCP/PP/EPbackend/localprefill/PDcost differ jointly; no isolatedgain.","V11shapeactualfactory/PDv3 rejects ineligible nativegeometry beforehelper; 9DPOST including3largetokenize andnative400 fall back originalD once. Zero internalPoutput.","Finite twelve cold requests/bursts/drain, not stablearrival SLOcapacity or hardware bound/KEEP; pipeline stages sameDinstance faultdomain. Nativebackground occupancy not exercised.","Gateway64KiB telemetry maydrop large promptIDframes; unchanged native clientwire audited independently2MiB."])

atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run115 12NEW variablecontext/output burst nativegateway E2E/10112commits/originalP/Dwire/coldcache/routing/queues/epoch/SDK0/idle verified; finite latency/TPS noKEEP/stablecapacity",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="12rawnativewire/tokenIDs/promptcounts/bodySHA/originalAPI2NPU32/policy/metrics")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(outputs=10112,helpers=0,elapsed=a["elapsed_s"],latency=a["latency"],pressure=pressure)))
