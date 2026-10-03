from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request,time
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0164";folder=r/"restored";state125=p/"runs/GLM-RUN-0125"
def ref(f):
 raw=Path(f).read_bytes();return dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def metric(f):
 out={}
 for l in Path(f).read_text().splitlines():
  if not l or l.startswith("#"):continue
  name=l.split("{")[0].split()[0]
  if name.startswith("vllm:"):out[name]=out.get(name,0)+float(l.rsplit(" ",1)[1])
 return out
s=json.loads((r/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
m=json.loads((r/"manifest.json").read_text());pins=m["source"]["source_identities"]
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(x["path"])["sha256"]==x["sha256"]
assert ref(r/"controller_spec.json")["sha256"]==m["controller_spec_sha256"]==s["spec_sha256"]
for n in["prepare","pilot","switch","finalize"]:
 v=json.loads((r/(n+".phase.json")).read_text());assert v["status"]=="succeeded"and v["exit_code"]==0and not v["timed_out"]
summary=json.loads((r/"functional_summary.json").read_text());assert summary["valid"]and summary["functional_acceptance"]and summary["new_native_completed"]==9and summary["D0_new_output_tokens"]==0
pilot=json.loads((r/"pilot_SDK_summary.json").read_text());assert pilot["measurement_valid"]and pilot["SDKinit_finalize0"]and pilot["output_tokens"]==192and pilot["cold_HTTP_overlap_s"]>0
ids=[];requests=[];direct=[]
for i in[0,1]:
 d=r/("pilot_"+str(i));v=json.loads((d/"pilot_summary.json").read_text());assert v["measurement_valid"]and v["effective_public_output_tokens"]==96and len(v["requests"])==2
 events=[json.loads(l)for l in(r/("D1_pilot_"+str(i)+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
 for row in v["requests"]:
  raw=Path(row["wire"]["path"]).read_bytes();assert ref(row["wire"]["path"])==row["wire"]and ref(d/(row["name"]+".body.json"))["sha256"]==row["request_body_sha256"]
  ob=NativeSSEObserver(max_bytes=2097152,collect_contract=True);ob.feed(raw);contract=ob.contract();assert contract==row["contract"]and contract["done"]and not contract["unknown"]and not contract["native_error"]and contract["finish_reasons"]=={"0":"length"}
  chunks=[]
  for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
   data=b"\n".join(l[5:].lstrip(b" ")for l in frame.splitlines()if l.startswith(b"data:"))
   if data and data!=b"[DONE]":chunks.append(json.loads(data))
  native_id="chatcmpl-GLM-RUN-0164-"+str(i)+"-"+row["name"]
  assert chunks and all(x["id"]==native_id for x in chunks);ids.append(native_id)
  inp=21 if row["name"]=="short_D1"else 81932;out=32 if inp==21 else 64
  assert sum(len(c.get("token_ids")or[])for x in chunks for c in x["choices"])==out
  assert len(chunks[0]["prompt_token_ids"])==inp and row["usage"]==dict(prompt_tokens=inp,completion_tokens=out,total_tokens=inp+out)
  assert row["completed"]and row["http_status"]==200and row["owner"]=="D1"
  body=json.loads((d/(row["name"]+".body.json")).read_text());assert body["cache_salt"].startswith(r.name+"-"+str(i)+"-")or body["cache_salt"].startswith(r.name+"-pilot_"+str(i)+"-")
  direct.append(dict(client=i,native_id=native_id,request=row))
  requests.append(row)
assert len(ids)==len(set(ids))==4
counters={}
for node in["166","167"]:
 before=metric(r/("pilot_before_"+node+".metrics"));after=metric(r/("pilot_after_"+node+".metrics"));delta={k:after[k]-v for k,v in before.items()if k.endswith("_total")}
 assert after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==0
 if node=="166":assert all(x==0for x in delta.values())
 else:
  for name,expected in[("generation_tokens_total",192),("prompt_tokens_total",163906),("request_success_total",4),("prefix_cache_hits_total",0),("external_prefix_cache_hits_total",0),("num_preemptions_total",0)]:assert delta["vllm:"+name]==expected
 counters[node]=delta
trace=[json.loads(l)for l in(state125/"router_trace.jsonl").read_text().splitlines()]
clients={};output=192;public_completed=0
for mode in["before","survivor","final"]:
 mr=(p/"runs/GLM-RUN-0163")if mode=="before"else r
 v=json.loads((mr/("client_"+mode+"_summary.json")).read_text());assert v["valid"]
 events=[json.loads(l)for l in(mr/("client_"+mode+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
 for n,row in enumerate(v["requests"]):
  for k in["body","wire"]:assert ref(row[k]["path"])==row[k]
  if row["status"]==503:assert row["rejected_before_lease_and_RPC"];continue
  assert ref(row["native_wire"]["path"])==row["native_wire"]and Path(row["wire"]["path"]).read_bytes()==Path(row["native_wire"]["path"]).read_bytes()
  header=mr.name+"-client_"+mode+"_"+row["name"]+"_"+str(n)
  leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==header];assert len(leases)==1
  lease=leases[0];assert lease["lease_id"]==row["lease_id"]and lease["replica"]==row["native_owner"]and lease["body_sha256"]==row["body"]["sha256"]
  wire=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];release=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]]
  assert len(wire)==len(release)==1and wire[0]["audit_error"]is None and release[0]["released"]and not release[0]["backend_failure"]
  if row.get("effective_output_tokens",0)>0:public_completed+=1
 assert sum(x["generation_tokens_total"]for x in v["native_delta"].values())==v["effective_output_tokens"]
 if mode!="final":assert v["effective_output_tokens"]==0
 else:
  assert v["native_delta"]["D0"]["generation_tokens_total"]==0
  semantics=[x for x in v["requests"]if x.get("semantic_pass")];assert len(semantics)==3
  for row,answer in zip(semantics,["4","42","GLM_OK_731"]):
   value=json.loads(Path(row["wire"]["path"]).read_text());assert value["choices"][0]["message"]["content"].strip()==answer and value["choices"][0]["finish_reason"]=="stop"
  base=next(x for x in v["requests"]if x["name"]=="newD1_create");got=next(x for x in v["requests"]if x["name"]=="newD1_retrieve")
  assert json.loads(Path(base["wire"]["path"]).read_text())==json.loads(Path(got["wire"]["path"]).read_text())
  basev=json.loads(Path(base["wire"]["path"]).read_text());assert basev["id"]=="resp_glm_run164_D1_new"and basev["usage"]["output_tokens"]==32
  child=next(x for x in v["requests"]if x["name"]=="newD1_previous");retrieved=next(x for x in v["requests"]if x["name"]=="newD1_child_retrieve")
  childv=json.loads(Path(retrieved["wire"]["path"]).read_text());assert childv["previous_response_id"]==basev["id"]and childv["usage"]["output_tokens"]==16
 output+=v["effective_output_tokens"];clients[mode]=dict(summary=ref(mr/("client_"+mode+"_summary.json")),requests=len(v["requests"]),outputs=v["effective_output_tokens"],SDK0=True)
assert public_completed==5and output==summary["effective_output_tokens"]
config,_=checked_config(folder/"service_config.json");old,_=checked_config(p/"runs/GLM-RUN-0156/restored/service_config.json")
epochs={x["id"]:x["epoch"]for x in config["native_domains"]};prior={x["id"]:x["epoch"]for x in old["native_domains"]}
assert epochs["local-166"]==prior["local-166"]and epochs["local-167"]!=prior["local-167"]
roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text());physical={};policies={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 ids_by_pid={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids_by_pid[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 assert v["env"]["VLLM_PP_LAYER_PARTITION"]==("42,36"if o["host"]=="166"else"38,40")
 physical[key]=dict(root_same=True,actual_NPU16=True,root=v["root"],proof=ref(j/(key+".owner.stdout")))
 path="/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_engines137"if o["host"]=="166"else"local_pp163")
 log="/data/tiankuan/wio/glm52-pd/deploy/logs/"+("local_engines137_166.log"if o["host"]=="166"else"local_pp163_167.log")
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(path)+");print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),files={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in p.glob('*.py')},SDK_events=[json.loads(l)for l in Path("+repr(log)+").read_text(errors='replace').splitlines()if l.startswith('{\\\"event\\\":')])))"
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);(j/("native_source_"+o["host"]+".stdout")).write_bytes(z.stdout);(j/("native_source_"+o["host"]+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096 if o["host"]=="166"else 8192,prefill_threshold_tokens=1024 if o["host"]=="166"else 4096,prefill_cadence=1,serial=11 if o["host"]=="166"else 1)
 assert v["policy"]==expected
 for name,sha in v["files"].items():assert sha==ref(p/"runs/GLM-RUN-0163/plugin_src"/name)["sha256"]
 assert v["SDK_events"][0]["event"]=="task_acl_init"and v["SDK_events"][0]["returncode"]==0
 policies[o["host"]]=v
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and [x.decode()for x in Path("/proc/"+str(public["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==public["argv"]and ref(public["config"]["path"])==public["config"]
owner=json.loads((folder/"identity_observer/process_owner.json").read_text());assert owner_alive(owner)
obs=observe(folder/"service_config.json");assert all(x["status"]=="healthy"for x in obs["groups"]);atomic_json(j/"HOST_observer_now.json",obs)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(url):
 with http.open(url,timeout=15)as res:return json.loads(res.read())
health=get("http://127.0.0.1:8000/healthcheck");assert health["status"]=="ok"and health["request_num"]==0
placement=get("http://127.0.0.1:8000/control/replicas");assert len(placement["replicas"])==2and all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]and x["work_ranking_calibrated"]for x in placement["replicas"])
fault=json.loads((state125/"response_owners.json.fault").read_text());assert fault["open"]and all(not x["faulted"]for x in fault["groups"].values())
adopt=json.loads((r/"adoption_summary.json").read_text());assert adopt["valid"]and adopt["native_process_restart0"]and adopt["PP38_40_epoch_adopted"]
proof163=json.loads((p/"jobs/AUDIT-FAIL163V2-20261003T0924Z/reduction.json").read_text());assert proof163["readonly_terminal_acceptance"]and proof163["new_native_inference_outputs"]==64
signals=[]
retired=json.loads((r/"retire156_restored.json").read_text());assert retired["SDKinit_finalize0"]and retired["models_signalled"]==0
assert (p/"runs/GLM-RUN-0156/restored/identity_observer/terminal.json").exists()
log=(folder/"public.gateway.log").read_text();events=[json.loads(l)for l in log.splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[0]["returncode"]==0and not any(x["event"]=="task_acl_finalize"for x in events)
native_final={}
for key,o in roots.items():
 with http.open("http://172.16.10."+o["host"]+":"+str(9081 if o["host"]=="166"else 9900)+"/metrics",timeout=15)as response:raw=response.read()
 f=j/("terminal_"+key+".metrics");f.write_bytes(raw);v=metric(f);assert v["vllm:num_requests_running"]==v["vllm:num_requests_waiting"]==0
 if o["host"]=="167":assert v["vllm:generation_tokens_total"]==output+64 and v["vllm:request_success_total"]==11and v["vllm:num_preemptions_total"]==0
 native_final[key]=ref(f)
samples=json.loads((r/"pilot_observations.json").read_text());values=[]
for row in samples:assert ref(row["metrics"]["path"])==row["metrics"];values.append(metric(row["metrics"]["path"]))
pressure={k:max(x[k]for x in values)for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]}
policy=json.loads((r/"policy_consumption.json").read_text());assert policy["operator_math_changes"]==0and policy["D0_unchanged"]
marker=public["container_marker"];container=dict(host="166",namespace="glm52-single_container",pid=marker["pid"],identity=dict(pid=marker["pid"],boot_id=marker["boot_id"],start_ticks=marker["start_ticks"]),argv=public["argv"],port=8000,config=public["config"],native_domains=config["native_domains"])
proof=dict(at=utc(),host=public["host"],container=container,argv=public["argv"],marker=marker,placement=placement,SDK_init0=True,retained=True,state_dir=str(state125));atomic_json(j/"public_service_proof.json",proof)
artifacts=[ref(f)for f in sorted(r.rglob("*"))if f.is_file()and f.name not in["manifest.json","state.json","controller.log"]and "identity_observer"not in f.parts];atomic_json(j/"artifact_index.json",artifacts)
limits=["Finite nativecoldpair fit/functional API and workload onnewD1 epoch only; no formal61440/stable-global capacity/repeated isolatedgain/KEEP","HTTP coldpair overlap/nativeRunning observation doesnot prove eachnativeGPUstep used8192 tokens","Privatebudget preserves nativeoperators/math/guards and perrequestthreshold4096; D1PP42,36→38,40 withsameallocated8192/threshold4096; newnativeepoch/weight-HBM/cachehistories, notisolatedrepeatgain","OldD1 STORE remains unavailable503beforeRPC; no replication or request replay. Native ownedretirement has no claimed SDK finalizeACK; clients/oldpublic observedinit-final0 and newmodels/publicinit0active","Coarsepairedshort/cold HTTP hints are epochbounddiagnostics, not GPU time or accurate batch/progress/background predictors"]
out=dict(at=utc(),run_id=r.name,measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",source_pins=len(pins),active_NPU_ranks=32,physical=physical,native_policy=policies,effective_output_tokens=output,new_native_completed=9,internal_helper_commits=0,uncredited_native_output_tokens=0,D0_new_output_tokens=0,pairedcold_HTTP_overlap_s=pilot["cold_HTTP_overlap_s"],direct_native=direct,clients=clients,native_pilot_counters=counters,native_pressure=pressure,native_final=native_final,policy_consumption=ref(r/"policy_consumption.json"),public_service=proof,D0_epoch_unchanged=True,new_native_epochs=epochs,retired_native_epochs=prior,signals=dict(owned_D1=0,unknown=0,D0=0),native_restarts=0,adopted_native163_epoch=True,prior_native_outputs64_excluded=True,SDKclients_init_finalize0=True,retired_public_init_finalize0=True,newnative_public_SDKinit0active=True,STORE_replication=False,request_replay=False,model_operations_during_audit=0,artifact_index=ref(j/"artifact_index.json"),limits=limits)
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run164 nativePP38,40/allocated8192 D1 fit/coldpair192 + public nativeJSON-SSE-STORE semantics/"+str(output)+" completeoutputs/SDK0/source/native32idle/nohelpers, functional VALID INCONCLUSIVE capacity",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="actualnativeallIDs-bodywire-tokenIDs-usage-DONE/cost/SDK/epoch/STORE")],unknowns=limits,decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,output_tokens=output,pressure=pressure,epochs=epochs)))
