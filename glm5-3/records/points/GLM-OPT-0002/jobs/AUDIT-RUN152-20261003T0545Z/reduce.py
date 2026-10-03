from pathlib import Path
import sys,json,hashlib,subprocess,re,time,urllib.request,shlex
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process,process_identity
from protocol_receipts import verify_protocol_receipts
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0152";formal=r/"formal";bench=formal/"benchmark";deadline=time.monotonic()+8800
while True:
 state=json.loads((r/"state.json").read_text())
 if state["status"]!="running":break
 assert same_process(state["owner"])and time.monotonic()<deadline;time.sleep(10)
assert state["status"]=="completed"and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for v in spec["stages"][0]["sources"]:
 assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
for stage in["prepare","formal"]:
 a=json.loads((r/(stage+".phase.json")).read_text());assert a["status"]=="succeeded"and a["exit_code"]==0and not a["timed_out"]
assert json.loads((r/"formal_summary.json").read_text())["functional_acceptance"]
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
prior=p/"runs/GLM-RUN-0128/formal/benchmark"
datasets=list((bench/"dataset").glob("GSM8K-in81920-num4-*.jsonl"));oldsets=list((prior/"dataset").glob("GSM8K-in81920-num4-*.jsonl"));assert len(datasets)==len(oldsets)==1and datasets[0].read_bytes()==oldsets[0].read_bytes()
oldreceipt=json.loads((prior/"full_wire_acceptance.json").read_text());assert sorted(x["body_sha256"]for x in checks["full"]["receipts"]["requests"])==sorted(x["body_sha256"]for x in oldreceipt["requests"])
for phase,each in [("warmup",1),("full",2)]:assert {k:sum(x["replica"]==k for x in checks[phase]["receipts"]["requests"])for k in ["D0","D1"]}==dict(D0=each,D1=each)
command=json.loads((formal/"benchmark_command.json").read_text())["argv"]
for option,value in[("--input_len","81920"),("--output_len","61440"),("--data_num","4"),("--concurrency","4"),("--request_rate","0"),("--repeat_rate","0.9"),("--prefix_num","1"),("--seed","20260930"),("--host_port","8000"),("--npu_num","32"),("--dp","2")]:assert command[command.index(option)+1]==value
assert command[command.index("--pod_info")+1:command.index("--pod_info")+3]==["172.16.10.166:9081","172.16.10.167:9900"]
assert not list(Path(checks["full"]["details_path"]).parent.glob("request_scope_amendment.json"))
details=Path(checks["full"]["details_path"]);perf=details.with_name("gsm8k.csv");assert perf.exists()
argv=["python3","/data/tiankuan/wio/glm52-pd/deploy/scripts/analyze_slo.py","--perf_csv",str(perf),"--details_jsonl",str(details),"--expect_output_len","61440","--concurrency","4","--out_json",str(j/"slo.json"),"--out_md",str(j/"slo.md")]
z=subprocess.run(argv,capture_output=True,timeout=30);(j/"slo.stdout").write_bytes(z.stdout);(j/"slo.stderr").write_bytes(z.stderr);z.check_returncode();slo=json.loads((j/"slo.json").read_text());assert slo["all_requests_succeeded"]and slo["output_len_ok"]and slo["n_success"]==4
acks=[json.loads(x)for x in(r/"native_formal.stdout").read_text().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
trace=[json.loads(x)for x in(formal/"router_trace.jsonl").read_text().splitlines()if x.strip()]
leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("method")=="POST"and x.get("path")=="/v1/chat/completions"]
assert len(leases)==6and sum(x["replica"]=="D0"for x in leases)==sum(x["replica"]=="D1"for x in leases)==3
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
 assert delta["vllm:request_success_total"]==3
 assert delta["vllm:generation_tokens_total"]>=122881and delta["vllm:prompt_tokens_total"]==73740+81932*2
 counters[node]=dict(delta=delta,idle=True,uncredited_output_cost=delta["vllm:generation_tokens_total"]-122881)
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
config,_=checked_config(r/"service_config.json");roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());physical={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 ids={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);physical[key]=dict(root_same=True,actual_NPU16=True,root=v["root"])
 code="from pathlib import Path;import json,hashlib;print(json.dumps(dict(policy=json.loads(Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/issue_budget_policy.json').read_text()),source_sha256=hashlib.sha256(Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/issue_budget_scheduler_v3.py').read_bytes()).hexdigest())))"
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=30);z.check_returncode();v=json.loads(z.stdout);assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3);assert v["source_sha256"]==hashlib.sha256((p.parents[2]/"runtime/issue_budget_scheduler_v3.py").read_bytes()).hexdigest()
public=json.loads((p/"jobs/AUDIT-RUN151V2-20261003T0523Z/public_service_proof.json").read_text());host=public["host"];service=public["container"]
assert same_process(host)and [v.decode()for v in Path("/proc/"+str(host["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==public["argv"]
assert hashlib.sha256(Path(service["config"]["path"]).read_bytes()).hexdigest()==service["config"]["sha256"]and service["native_domains"]==config["native_domains"]
ports=subprocess.check_output(["ss","-ltnp"],text=True);rows=[x for x in ports.splitlines()if re.search(r":8000\s",x)];assert len(rows)==1and "pid="+str(host["pid"])+","in rows[0]
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as res:placement=json.loads(res.read())
assert len(placement["replicas"])==2and all(not x["temporarily_unhealthy"]and not x["group_faulted"]and not x["draining"]and x["active_requests"]==0and x["work_ranking_calibrated"]for x in placement["replicas"])
fault=json.loads((p/"runs/GLM-RUN-0125/response_owners.json.fault").read_text());assert fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
observer=observe(Path(service["config"]["path"]));assert all(x["status"]=="healthy"for x in observer["groups"])
obsowner=json.loads((p/"runs/GLM-RUN-0149/restored/identity_observer/process_owner.json").read_text());assert owner_alive(obsowner)
atomic_json(j/"public_service_proof.json",dict(public,at=utc(),placement=placement))
peaks={}
for node in["D0","D1"]:
 files=list(formal.glob("sample*_"+node+".metrics"));assert files
 vals=[metric(f)for f in files];peaks[node]={k:max(v.get(k,0)for v in vals)for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]}
 assert peaks[node]["vllm:num_requests_running"]==2
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
verdict="INCONCLUSIVE"if slo["slo_all_pass"]else"REJECT"
verdict="INCONCLUSIVE"if slo["slo_all_pass"]else"REJECT"
out=dict(at=utc(),run_id=r.name,kind="formal_e2e",functional_acceptance=True,measurement_valid=True,verdict=verdict,source_pins=len(spec["stages"][0]["sources"]),complete_outputs=245760,completed_full_requests=4,warmup_outputs=2,warmup_requests=2,internal_helper_commits=0,total_minimum_native_commits=245762,uncredited_native_output_tokens=sum(x["uncredited_output_cost"]for x in counters.values()),full_cli_phase_wall_s=wall,effective_TPS_full_cli=245760/wall,warmup_cli_wall_s=warmwall,actual_active_domains=["D0","D1"],active_NPU_ranks=32,resident_NPU_ranks=32,actualfull_requests_perdomain=2,native_counters=counters,native_pressure=peaks,physical_same=physical,public_frontend_same=True,public_listener=rows[0],public_HTTP_leases_zero=True,checks=checks,SLO=slo,actual_client_SDK_init_finalize=[acks[0],acks[-1]],artifact_index=ref(j/"artifact_index.json"),model_operations=0,gateway_operations=0,dataset_exact128=True,request_bodyhashes_exact128=True,calibrated_work_seconds=True,limits=["Four closedc4 fullrequests/bothdomains prefixwarm/native32 are not openarrival stablecapacity/repeatedKEEP/fullframeworkcompletion/robustP99/globalhardwarebound","Same rawinputdataset andfullbody hashes as128; placement/all32active/twoAPIwarmup/concurrency/nativenumerictrajectory/resourceepoch joint differences preventisolatedcodegain claims","128fourfullclosedc2 wasoneD1active16 while32resident;21bothwarm/TP16PP1EP16K5 differsfromcurrentlocalTP8PP2DCP8/noEPAllGather/noKV/K3; compareconditions explicitly","Nativecountersincludetwopublicprefixwarmtokens, uncreditedoutputcostseparate; no internalMTPdraft/helper substitution","Residentcache observed/no reset; actualnative prefixhits; dp2benchmark warmcount doesnotchange eachengine nativeDP1","ImmutablecheckedD0hintrestoredbylogicalfullpeer readd, no physicalepochchange/frontendrestart; D1epoch149 observedrates, not GPUprogress/batching/background forecast","Additionalreadonlynativecounter observer beganafterfullphase started; AISBenchtimersincludeobserver/audit overhead without subtraction","ExistingAISBench TTFT/TPOT/reference6thresholds unchanged; fullnativeIDs-usage-length-DONE/body-nativewire strongerthanCLI0"])
atomic_json(j/"reduction.json",out)
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="152full4x61440/245760/closedc4/bothdomains/warm2/native32/SDK0/nohelpers/source/fullwire/finiteTPS="+str(245760/wall)+"/SLO="+str(slo["slo_all_pass"])+"/"+verdict,execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualAISBenchallattempts/nativewire/usage-length-DONE/sourceepochs/cost/SLO")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(verdict=verdict,TPS=245760/wall,SLO=slo["slo_all_pass"],native_counters=counters,pressure=peaks)))
