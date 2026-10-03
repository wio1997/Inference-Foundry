from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0149";recovery=p/"runs/GLM-RUN-0151";folder=r/"restored"
def ref(f):
 raw=f.read_bytes();return dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
original_state=json.loads((r/"state.json").read_text());assert original_state["status"]=="failed"and not same_process(original_state["owner"])
s=json.loads((recovery/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
m=json.loads((r/"manifest.json").read_text());pins=m["source"]["source_identities"]
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(Path(x["path"]))["sha256"]==x["sha256"]
assert ref(r/"controller_spec.json")["sha256"]==m["controller_spec_sha256"]==original_state["spec_sha256"]
newm=json.loads((recovery/"manifest.json").read_text());newpins=newm["source"]["source_identities"]
for x in newpins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(Path(x["path"]))["sha256"]==x["sha256"]
assert ref(recovery/"controller_spec.json")["sha256"]==newm["controller_spec_sha256"]==s["spec_sha256"]
final=json.loads((recovery/"reconcile_summary.json").read_text());assert final["valid"]and final["SDKinit_finalize0"]and final["same149_native32"]and final["models_operations"]==final["frontend_operations"]==0
newclient=final["client"];assert newclient["valid"]and newclient["new_inference"]==newclient["effective_output_tokens"]==0and newclient["native_before"]==newclient["native_after"]
for row in newclient["requests"]:
 for key in ["body","wire"]:assert ref(Path(row[key]["path"]))==row[key]
 if row["status"]==503:assert row["rejected_before_lease_and_RPC"]
 else:assert ref(Path(row["native_wire"]["path"]))==row["native_wire"]

config,_=checked_config(folder/"service_config.json");old,_=checked_config(r/"service_config.json")
epochs={x["id"]:x["epoch"]for x in config["native_domains"]};prior={x["id"]:x["epoch"]for x in old["native_domains"]};assert epochs["local-166"]==prior["local-166"]and epochs["local-167"]!=prior["local-167"]
roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
probes={};policies={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".stdout")).write_bytes(z.stdout);(j/(key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 ids={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);probes[key]=dict(root_same=True,actual_NPU16=True,proof=ref(j/(key+".stdout")))
 code="from pathlib import Path;import json,hashlib;print(json.dumps(dict(policy=json.loads(Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/issue_budget_policy.json').read_text()),source_sha256=hashlib.sha256(Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/issue_budget_scheduler_v3.py').read_bytes()).hexdigest(),SDK_events=[json.loads(l)for l in Path('/data/tiankuan/wio/glm52-pd/deploy/logs/"+("local_engines137_166.log"if o["host"]=="166"else"local_engines149_167.log")+"').read_text(errors='replace').splitlines()if l.startswith('{\"event\":')])))"
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode();v=json.loads(z.stdout);atomic_json(j/("native_source_"+o["host"]+".json"),v)
 assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3)
 assert v["source_sha256"]==ref(p.parents[2]/"runtime/issue_budget_scheduler_v3.py")["sha256"]
 assert v["SDK_events"][0]["event"]=="task_acl_init"and v["SDK_events"][0]["returncode"]==0;policies[o["host"]]=v
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and [x.decode()for x in Path("/proc/"+str(public["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==public["argv"]and ref(Path(public["config"]["path"]))==public["config"]
observer_owner=json.loads((folder/"identity_observer/process_owner.json").read_text());assert owner_alive(observer_owner)
obs=observe(folder/"service_config.json");assert all(x["status"]=="healthy"for x in obs["groups"]);atomic_json(j/"HOST_observer_now.json",obs)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(url):
 with http.open(url,timeout=15)as res:return json.loads(res.read())
health=get("http://127.0.0.1:8000/healthcheck");assert health["status"]=="ok"and health["request_num"]==0
placement=get("http://127.0.0.1:8000/control/replicas");assert len(placement["replicas"])==2and all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]and x["work_ranking_calibrated"]for x in placement["replicas"])
fault=json.loads((p/"runs/GLM-RUN-0125/response_owners.json.fault").read_text());assert all(not x["faulted"]for x in fault["groups"].values())
oldfault=json.loads((r/"physical_fault_ack.json").read_text());assert oldfault["source"]=="automaticHOSTobserver"and next(x for x in oldfault["placement"]["replicas"]if x["id"]=="D1")["group_faulted"]and not next(x for x in oldfault["placement"]["replicas"]if x["id"]=="D0")["group_faulted"]
assert (r/"identity_observer/terminal.json").exists()
cleanup=[json.loads(l)for l in (r/"D1_stop.stdout").read_text().splitlines()if l.startswith("{")];done=cleanup[-1];assert done["event"]=="cleanup_complete"and done["all_original_native_domain_inactive"]and done["npu_workers"]==done["signals_to_unknown"]==done["signals_to_inactive"]==0
owned={x["pid"]:x["identity"]for x in json.loads((r/"standalone_native_members.json").read_text())["node1"]["owned_targets"]};signals=[x for x in cleanup if x["event"]=="signal_attempt"];assert signals and all(x["pid"]in owned and x["identity"]==owned[x["pid"]]for x in signals)
clients={};credit=0
for mode,expected in [("before",0),("survivor",11),("final",59)]:
 value=json.loads((r/("client_"+mode+"_summary.json")).read_text());assert value["effective_output_tokens"]==expected
 if mode=="final":
  assert value["valid"]is False and value["error_type"]=="AssertionError"and value["requests"][-1]["name"]==value["requests"][0]["name"]=="D0_retained"and value["requests"][-1]["status"]==200
  native_delta={}
  for key in ["D0","D1"]:
   text=(r/("client_final_before_"+key+".metrics")).read_text();start={}
   for metric in ["generation_tokens_total","prompt_tokens_total","num_preemptions_total"]:
    vals=re.findall(r"^vllm:"+metric+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M);assert vals;start[metric]=sum(float(x)for x in vals)
   native_delta[key]={metric:newclient["native_after"][key][metric]-start[metric]for metric in start}
  assert native_delta["D1"]["generation_tokens_total"]==59and native_delta["D0"]["generation_tokens_total"]==0and all(x["num_preemptions_total"]==0for x in native_delta.values())
  events=[json.loads(l)for l in (r.parent/"GLM-RUN-0125/router_trace.jsonl").read_text().splitlines()]
  repeated=[x for x in events if x["event"]=="lease_acquired"and x.get("request_header_id")=="GLM-RUN-0149-client_final_D0_retained"];assert len(repeated)==2
  for lease in repeated:
   assert lease["replica"]=="D0"and lease["affinity_applied"]and lease["path"]=="/v1/responses/resp_glm_run139_base"and lease["method"]=="GET"
   wire=[x for x in events if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];release=[x for x in events if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]];headers=[x for x in events if x["event"]=="upstream_headers"and x.get("lease_id")==lease["lease_id"]]
   assert len(wire)==len(release)==len(headers)==1and headers[0]["status"]==200and release[0]["released"]and not release[0]["backend_failure"]
   assert ref(Path(wire[0]["wire_path"]))==dict(path=wire[0]["wire_path"],bytes=wire[0]["wire_bytes"],sha256=wire[0]["wire_sha256"])and wire[0]["wire_sha256"]==value["requests"][-1]["wire"]["sha256"]
 else:assert value["valid"];native_delta=value["native_delta"]
 assert sum(x["generation_tokens_total"]for x in native_delta.values())==expected
 raw=(r/("client_"+mode+".stdout")).read_text();acks=[json.loads(l)for l in raw.splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
 for row in value["requests"]:
  for tag in ["body","wire"]:
   f=Path(row[tag]["path"]);assert ref(f)==row[tag]
  if row["status"]==503:assert row["rejected_before_lease_and_RPC"]
  elif "native_wire"in row:assert ref(Path(row["native_wire"]["path"]))==row["native_wire"]
  else:assert mode=="final"and row is value["requests"][-1]and row["name"]=="D0_retained"
 clients[mode]=dict(valid=True,effective_output_tokens=expected,native_delta=native_delta,requests=len(value["requests"]),SDK0=True,summary=ref(r/("client_"+mode+"_summary.json")));credit+=expected
pilot=json.loads((r/"pilot_summary.json").read_text());assert pilot["measurement_valid"]and pilot["effective_public_output_tokens"]==96and len(pilot["requests"])==2
for row in pilot["requests"]:
 assert row["completed"]and row["owner"]=="D1"and row["contract"]["done"]and not row["contract"]["unknown"]and not row["contract"]["native_error"]and row["contract"]["finish_reasons"]=={"0":"length"}and ref(Path(row["wire"]["path"]))==row["wire"]
 u=row["usage"];assert u["prompt_tokens"]==(21 if row["name"]=="short_D1"else 81932)and u["completion_tokens"]==(32 if row["name"]=="short_D1"else 64)
 from sse_observer import NativeSSEObserver
 ob=NativeSSEObserver(max_bytes=2097152,collect_contract=True);ob.feed(Path(row["wire"]["path"]).read_bytes());assert ob.contract()==row["contract"]
raw=(r/"D1_pilot.stdout").read_text();acks=[json.loads(l)for l in raw.splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
# Reconcile direct native pilot generation; initial/restored1s counters are separate epochs.
def generation(path):
 text=path.read_text();return sum(float(x)for x in re.findall(r"^vllm:generation_tokens_total(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M))
assert generation(r/"D1_after_D1.metrics")-generation(r/"initial_D1.metrics")==96
credit+=96;assert credit==166
for label in ["retire147","retire149_initial"]:assert json.loads((r/(label+".json")).read_text())["SDKinit_finalize0"]
for logfile in [p/"runs/GLM-RUN-0147/public.gateway.log",r/"public.gateway.log"]:
 acks=[json.loads(l)for l in logfile.read_text().splitlines()if l.startswith('{"event":')];assert [x["event"]for x in acks]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in acks)
native_raw=[];snapshot_limits={}
for key,o in roots.items():
 with http.open("http://172.16.10."+o["host"]+":"+str(9081 if o["host"]=="166"else 9900)+"/metrics",timeout=10)as res:raw=res.read()
 f=j/("final_"+key+".metrics");f.write_bytes(raw);counts=re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",raw.decode(),re.M);assert counts and all(float(x)==0for x in counts);native_raw.append(ref(f))
marker=public["container_marker"];container=dict(host="166",namespace="glm52-single_container",pid=marker["pid"],identity=dict(pid=marker["pid"],boot_id=marker["boot_id"],start_ticks=marker["start_ticks"]),argv=public["argv"],port=8000,config=public["config"],native_domains=config["native_domains"])
proof=dict(at=utc(),host=public["host"],container=container,argv=public["argv"],marker=marker,placement=placement,SDK_init0=True,retained=True,state_dir=str(p/"runs/GLM-RUN-0125"));atomic_json(j/"public_service_proof.json",proof)
files=[recovery/n for n in ["state.json","controller_spec.json","reconcile_summary.json","native_reconcile.stdout","native_reconcile.stderr","client_reconcile_summary.json"]]+[r/n for n in ["state.json","controller_spec.json","physical_fault_ack.json","D1_stop.stdout","D1_stop.stderr","retire147.json","retire149_initial.json","client_before_summary.json","client_survivor_summary.json","client_final_summary.json","pilot_summary.json","pilot_SDK_summary.json","D1_pilot.stdout","D1_pilot.stderr"]]+[folder/n for n in ["standalone_launch.json","standalone_root_identities.json","standalone_native_members.json","native_engines_resident.json","service_config.json","observed_D1.json","deployment_summary.json","public_service_proof.json","public_host_owner.json"]]
limits=["Original149 remains overallINVALID due duplicate finalD0 request header fixture; separate151 readonly reconciliation VALID0newoutputs, reused14910 native inference/166outputs andsix exact semanticcases, no replay/broadquality/frameworkcompletion/capacity/KEEP","Fault injection exactoldD1native domain SIGTERM then owned residual cleanup; abrupt oldmodel exit has SDK init0 but no finalizeACK in old137native log, not claimed cleanSDK shutdown. Original process/NPUinactive proven","All actual client and retired frontend SDKinit/final0; surviving/replacement models SDKinit0active, no finalize yet","Native STORE is in-memory; oldD1 pointers stay503 afternewD1 epoch, no response replication or request replay","HOST identity detector confirms boot/start/argv/ancestry/process disappearance, not every GPUhang/device fault; unavailable SSH isunknown","D0 unchanged physicalepoch and prior138workhint; only D1 pilot refreshed epoch-bound coarse rates; not GPUprogress/batching/background forecast"]
out=dict(at=utc(),valid=True,kind="diagnostic",verdict="INCONCLUSIVE",source_count=len(pins)+len(newpins),controller=s,original149_controller=original_state,reconcile=ref(recovery/"reconcile_summary.json"),public_service=proof,physical_fault=ref(r/"physical_fault_ack.json"),D0_survivor=True,D1_old_bound503_beforeRPC=True,new_native_epochs=epochs,retired_native_epochs=prior,actual_native32=probes,native_policy=policies,clients=clients,pilot=ref(r/"pilot_summary.json"),effective_output_tokens=0,reused149_effective_output_tokens=credit,native_inference_requests=0,reused149_native_inference_requests=10,helper_output_tokens=0,STORE_replication=False,request_replay=False,signals=dict(owned_D1=len(signals),unknown=0,D0=0),retained_observer=ref(folder/"identity_observer/process_owner.json"),raw_artifacts=[ref(f)for f in files],limits=limits)
atomic_json(j/"reduction.json",out);atomic_json(j/"artifact_index.json",out["raw_artifacts"])
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run151 readonlynative reconciliation VALID0newoutputs/SDK0/source/32idle; original149overallINVALIDduplicateheader butseparate10native166outputs/physicalD1fault-newSTOREepoch/survivor/fullwire-cost verified, no replay/replication/capacity/KEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="terminalnativefault/newSTOREepoch/fullwire/source/cost/SKD/ownership")],unknowns=limits,decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,source_count=len(pins),new_output_tokens=0,reused149_output_tokens=credit,reused149_native_requests=10,epochs=epochs,limits=limits)))
