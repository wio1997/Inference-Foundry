from pathlib import Path
import sys,json,hashlib,subprocess,re,time,urllib.request,shlex
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process,process_identity
from protocol_receipts import verify_protocol_receipts
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0128";formal=r/"formal";bench=formal/"benchmark";deadline=time.monotonic()+9800
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
for phase,count,inputt,outputt in[("warmup",1,73740,1),("full",4,81932,61440)]:
 a=json.loads((bench/(phase+"_acceptance.json")).read_text());assert a["valid"]and a["actual_count"]==count
 original=json.loads((bench/(phase+"_wire_acceptance.json")).read_text())
 receipt=verify_protocol_receipts(formal/"router_trace.jsonl",count,inputt,outputt,["D1"]);assert receipt==original
 rows=[json.loads(x)for x in Path(a["details_path"]).read_text().splitlines()if x.strip()]
 assert len(rows)==count and len({x["id"]for x in rows})==count and all(x["success"]is True and x["output_tokens"]==outputt for x in rows)
 checks[phase]=dict(receipts=receipt,details_path=a["details_path"],all_actual_requests_valid=True)
ds=json.loads((bench/"dataset_validation.json").read_text());assert ds["valid"]and ds["prefix_text_matches"]and ds["raw_prompt_token_lengths"]==[81920]*4and ds["warmup_prefix_token_lengths"]==[73728]
command=json.loads((formal/"benchmark_command.json").read_text())["argv"]
for option,value in[("--input_len","81920"),("--output_len","61440"),("--data_num","4"),("--concurrency","2"),("--request_rate","0"),("--repeat_rate","0.9"),("--prefix_num","1"),("--seed","20260930"),("--host_port","8000"),("--npu_num","16"),("--dp","1")]:assert command[command.index(option)+1]==value
assert command[command.index("--pod_info")+1]=="172.16.10.167:9900"
assert not list(Path(checks["full"]["details_path"]).parent.glob("request_scope_amendment.json"))
details=Path(checks["full"]["details_path"]);perf=details.with_name("gsm8k.csv");assert perf.exists()
argv=["python3","/data/tiankuan/wio/glm52-pd/deploy/scripts/analyze_slo.py","--perf_csv",str(perf),"--details_jsonl",str(details),"--expect_output_len","61440","--concurrency","2","--out_json",str(j/"slo.json"),"--out_md",str(j/"slo.md")]
z=subprocess.run(argv,capture_output=True,timeout=30);(j/"slo.stdout").write_bytes(z.stdout);(j/"slo.stderr").write_bytes(z.stderr);z.check_returncode();slo=json.loads((j/"slo.json").read_text());assert slo["all_requests_succeeded"]and slo["output_len_ok"]and slo["n_success"]==4
acks=[json.loads(x)for x in(r/"native_formal.stdout").read_text().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
trace=[json.loads(x)for x in(formal/"router_trace.jsonl").read_text().splitlines()if x.strip()]
leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("method")=="POST"and x.get("path")=="/v1/chat/completions"];assert len(leases)==5and all(x["replica"]=="D1"for x in leases)
pd=[x for x in trace if x["event"].startswith("pd_")];assert len(pd)==5and all(x["event"]=="pd_geometry_native_fallback"and x["internal_output_tokens"]==0and"decoder_connector_unavailable"in x["reasons"]for x in pd)
def metric(f):
 out={}
 for line in f.read_text().splitlines():
  if not line or line.startswith("#"):continue
  name=line.split("{")[0].split()[0]
  if name.startswith("vllm:")and any(v in name for v in["generation_tokens_total","prompt_tokens_total","request_success_total","num_requests_running","num_requests_waiting","prefix_cache_hits_total","prefix_cache_queries_total","num_preemptions_total","kv_cache_usage_perc"]):
   out[name]=out.get(name,0)+float(line.rsplit(" ",1)[1])
 return out
counters={}
for node in["P166","D167"]:
 before=metric(formal/("after_cache_condition_"+node+".metrics"));after=metric(formal/("final_"+node+".metrics"));delta={k:after[k]-v for k,v in before.items()if k.endswith("_total")}
 assert after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==0and delta["vllm:num_preemptions_total"]==0
 assert delta["vllm:request_success_total"]==(0if node=="P166"else 5)
 if node=="P166":assert all(v==0for v in delta.values())
 else:assert delta["vllm:generation_tokens_total"]>=245761and delta["vllm:prompt_tokens_total"]==73740+81932*4
 counters[node]=dict(delta=delta,idle=True)
owners=json.loads((r/"adopted_model_identities.json").read_text());members=json.loads((r/"native_member_identities.json").read_text())
check="""import pathlib,json,sys,subprocess,re
a=json.load(sys.stdin);o=a['owner'];boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for x in a['targets']+[dict(pid=o['pid'],identity=o['identity'])]:
 p=pathlib.Path('/proc/'+str(x['pid']));b=(p/'stat').read_text();v=b[b.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity']
assert[v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);ids={int(v)for v in re.findall(r'^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker',raw,re.M)};assert ids=={x['pid']for x in a['targets']}
assert json.loads(pathlib.Path(a['policy_path']).read_text())==a['policy'];print(json.dumps(dict(API_same=True,NPU16_same=True,npu_pids=sorted(ids),policy=a['policy'])))
"""
physical={}
for key,o in owners.items():
 targets=[v for v in members[key]["owned_targets"]if v["pid"]in members[key]["npu_worker_pids"]]
 policy=dict(schema_version=1,cohort_id="GLM-COHORT-0089"if key=="D0"else"GLM-COHORT-0123",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3if key=="D0"else 1)
 args=["python3","-c",check]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,targets=targets,policy=policy,policy_path=str(Path(o["argv"][1]).parent/"issue_budget_policy.json"))).encode(),capture_output=True,timeout=90);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();physical[key]=json.loads(z.stdout)
public=json.loads((p/"jobs/AUDIT-RUN127-20261003T0005Z/public_service_proof.json").read_text());host=public["host_identity"];service=public["container_owner"]
assert same_process(host)and[v.decode()for v in Path("/proc/"+str(host["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==service["argv"]
ports=subprocess.check_output(["ss","-ltnp"],text=True);rows=[x for x in ports.splitlines()if re.search(r":8000\s",x)];assert len(rows)==1and("pid="+str(host["pid"])+",")in rows[0]and not re.search(r":8002\s",ports)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as res:placement=json.loads(res.read())
assert len(placement["replicas"])==2and all(not x["temporarily_unhealthy"]and not x["group_faulted"]and not x["draining"]and x["active_requests"]==0for x in placement["replicas"])
fault=json.loads((p/"runs/GLM-RUN-0125/response_owners.json.fault").read_text());assert fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
peaks={}
for node in["P166","D167"]:
 files=list(formal.glob("sample*_"+node+".metrics"));assert files
 vals=[metric(f)for f in files];peaks[node]={k:max(v.get(k,0)for v in vals)for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]}
 assert peaks[node]["vllm:num_requests_running"]==(0if node=="P166"else 2)
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
artifact_files=[f for f in sorted(r.rglob("*"))if f.is_file()and f.name not in["manifest.json","state.json","controller.log","artifact_index.json","reduction_brief.json"]]
artifacts=[ref(f)for f in artifact_files];atomic_json(r/"artifact_index.json",artifacts)
verdict="INCONCLUSIVE"if slo["slo_all_pass"]else"REJECT"
out=dict(at=utc(),run_id=r.name,kind="formal_e2e",functional_acceptance=True,measurement_valid=True,verdict=verdict,source_pins=len(spec["stages"][0]["sources"]),complete_outputs=245760,completed_full_requests=4,warmup_outputs=1,warmup_requests=1,internal_helper_commits=0,total_minimum_native_commits=245761,uncredited_native_output_tokens=counters["D167"]["delta"]["vllm:generation_tokens_total"]-245761,full_cli_phase_wall_s=wall,effective_TPS_full_cli=245760/wall,warmup_cli_wall_s=warmwall,actual_active_domain="D1",active_NPU_ranks=16,resident_NPU_ranks=32,native_counters=counters,native_pressure=peaks,physical_same=physical,public127_frontend_same=True,public_listener=rows[0],public_HTTP_leases_zero=True,checks=checks,SLO=slo,actual_client_SDK_init_finalize=[acks[0],acks[-1]],artifact_index=ref(r/"artifact_index.json"),model_operations=0,gateway_operations=0,limits=["Fourclosedrequests withoneD1prefixwarmup arenotopenarrival stablecapacity/repeatedKEEP/fullframeworkcompletion/robustP99/hardwarebound","CurrentV11inputbytes policy sendsallfourlargeoutputstoD1; P89idle isnot a bound on two-node feasiblecapacity; DPP2TP8DCP8K3/noEP/noKV/Graph32 differsfrom21TP16K5 balancedtwoAPIwarmup2","Greedy outputtrajectory may differwithnativeK/parallel/batchnumericsepoch; no exactmath-equivalence/quality/isolatedcost claim","Nativecountersincludeonewarmup; uncreditedoutputcostseparatefromeffective245760/4; no internalMTPdraft substitution","Residentcache observed/no reset; cachehits counted bynative; dp1benchwarmup doesnotmean independentliveAPI replica domain disappeared","No frontend/model operations; public127retained; APIstore isnotreplicated/physicalfaultautomaticmonitoring notclaimed","ExistingAISBench per-request TTFT/TPOT/SLOsameoriginal thresholds; rawprotocolallattempts strongerthanCLI0 only"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict=verdict,results=out);atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\n4/4 nativefull81932to61440/245760effective/closedc2/D1only16activeNPU32resident/P89idle/singlewarm73740to1/zerohelpers/SDK0/API2NPU32same/P3D1/idle/public127retained. FullCLI "+str(wall)+"s/"+str(245760/wall)+"finiteTPS; originalSLOallpass="+str(slo["slo_all_pass"])+". "+verdict+" forcurrentformalcontract; notglobalroute/parallelrejection orstablecapacity/KEEP. Nativehits/cost/pressure/rawwire/source indexedserver.\n")
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="128full4x61440/245760/closedc2/D1only/SDK0/currentmodels-publicretained/finiteTPS="+str(245760/wall)+"/SLO="+str(slo["slo_all_pass"])+"/"+verdict,execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualAISBenchallattempts/nativewire/usage-length-DONE/sourceepochs/counters/SLO")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(verdict=verdict,TPS=245760/wall,SLO=slo["slo_all_pass"],native_counters=counters,pressure=peaks)))
