from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0211";folder=r/"restored";state125=p/"runs/GLM-RUN-0125"
sys.path.insert(0,str(r/"runtime_bundle"))
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
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
m=json.loads((r/"manifest.json").read_text());spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"]
assert len(pins)==180
for st in spec["stages"]:assert st["sources"]==pins
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(x["path"])["sha256"]==x["sha256"]
assert ref(r/"controller_spec.json")["sha256"]==m["spec"]["sha256"]==s["spec_sha256"]
for n in["prepare","fault","restore","switch","finalize"]:
 v=json.loads((r/(n+".phase.json")).read_text());assert v["status"]=="succeeded"and v["exit_code"]==0and not v["timed_out"]
summary=json.loads((r/"functional_summary.json").read_text());assert summary["valid"]and summary["functional_acceptance"]and summary["new_native_completed"]==5and summary["D0_new_output_tokens"]==0and summary["effective_output_tokens"]==59
guard=json.loads((folder/"PP_empty_guard_workers.json").read_text());assert guard["workers"]==16and guard["operator_math_edits"]==0and len(guard["markers"])==16
trace=[json.loads(l)for l in(state125/"router_trace.jsonl").read_text().splitlines()]
clients={};output=0;public_completed=0
store_sources={}
for run,name,id in[("GLM-RUN-0204","newD0_create","resp_glm_run204_D0_new"),("GLM-RUN-0210","newD0_create","resp_glm_run210_D0_new"),("GLM-RUN-0200","D1_create","resp_glm_run200_D1_new")]:
 c=json.loads((p/"runs"/run/"client_final_summary.json").read_text());store_sources[id]=next(x for x in c["requests"]if x["name"]==name)["wire"]
for mode in["before","survivor","final"]:
 v=json.loads((r/("client_"+mode+"_summary.json")).read_text());assert v["valid"]
 events=[json.loads(l)for l in(r/("client_"+mode+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
 for n,row in enumerate(v["requests"]):
  for k in["body","wire"]:assert ref(row[k]["path"])==row[k]
  header=r.name+"-client_"+mode+"_"+row["name"]+"_"+str(n)
  leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==header]
  if row["status"]==503:
   assert row["rejected_before_lease_and_RPC"]and not leases
   assert row["name"]in["retired200_get","retired200_previous","retired202_get","retired202_previous"];continue
  assert ref(row["native_wire"]["path"])==row["native_wire"]and Path(row["wire"]["path"]).read_bytes()==Path(row["native_wire"]["path"]).read_bytes()
  assert len(leases)==1;lease=leases[0];assert lease["lease_id"]==row["lease_id"]and lease["replica"]==row["native_owner"]and lease["body_sha256"]==row["body"]["sha256"]
  wire=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];release=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]]
  assert len(wire)==len(release)==1and wire[0]["audit_error"]is None and release[0]["released"]and not release[0]["backend_failure"]
  assert wire[0]["wire_sha256"]==row["wire"]["sha256"]and wire[0]["wire_bytes"]==row["wire"]["bytes"]
  if row.get("effective_output_tokens",0)>0:
   public_completed+=1;assert row["native_owner"]=="D1"
  if row["name"]in["D0_retained","D0_210_retained","D1_retained"]:
   id=row["path"].rsplit("/",1)[1];assert Path(row["wire"]["path"]).read_bytes()==Path(store_sources[id]["path"]).read_bytes()and row["affinity_applied"]
   assert row["native_owner"]==("D1"if id=="resp_glm_run200_D1_new"else"D0")
  if row.get("semantic_pass"):
   value=json.loads(Path(row["wire"]["path"]).read_text());choice=value["choices"][0];expected={"add2":"4","add17":"42","literal":"GLM_OK_731"}[row["name"]]
   assert choice["message"]["content"].strip()==expected and choice["finish_reason"]=="stop"and len(choice["token_ids"])==row["effective_output_tokens"]==value["usage"]["completion_tokens"]
 assert sum(x["generation_tokens_total"]for x in v["native_delta"].values())==v["effective_output_tokens"]and v["native_delta"]["D0"]["generation_tokens_total"]==0
 assert any(x["name"]=="D0_retained"for x in v["requests"])and any(x["name"]=="D0_210_retained"for x in v["requests"])
 if mode=="before":assert v["effective_output_tokens"]==0and any(x["name"]=="D1_retained"for x in v["requests"])
 if mode=="survivor":assert v["effective_output_tokens"]==0and all(next(x for x in v["requests"]if x["name"]==name)["status"]==503for name in["retired200_get","retired200_previous"])
 if mode=="final":
  assert v["effective_output_tokens"]==59
  base=next(x for x in v["requests"]if x["name"]=="newD1_create");got=next(x for x in v["requests"]if x["name"]=="newD1_retrieve")
  basev=json.loads(Path(base["wire"]["path"]).read_text());assert basev==json.loads(Path(got["wire"]["path"]).read_text())and basev["id"]=="resp_glm_run211_D1_new"and basev["usage"]["output_tokens"]==32and got["affinity_applied"]
  child=next(x for x in v["requests"]if x["name"]=="newD1_previous");retrieved=next(x for x in v["requests"]if x["name"]=="newD1_child_retrieve")
  frames=[]
  for frame in Path(child["wire"]["path"]).read_bytes().replace(b"\r\n",b"\n").split(b"\n\n"):
   data=b"\n".join(l[5:].lstrip(b" ")for l in frame.splitlines()if l.startswith(b"data:"))
   if data:frames.append(json.loads(data))
  assert [x["sequence_number"]for x in frames]==list(range(len(frames)))and frames[-1]["type"]=="response.completed"
  childv=json.loads(Path(retrieved["wire"]["path"]).read_text());assert frames[-1]["response"]==childv and childv["previous_response_id"]==basev["id"]and childv["usage"]["output_tokens"]==16and child["affinity_applied"]and retrieved["affinity_applied"]
 output+=v["effective_output_tokens"];clients[mode]=dict(summary=ref(r/("client_"+mode+"_summary.json")),requests=len(v["requests"]),outputs=v["effective_output_tokens"],SDK0=True)
assert public_completed==5and output==59
config,_=checked_config(folder/"service_config.json");old,_=checked_config(r/"service_config.json")
assert config["placement"]==old["placement"]==dict(kind="shape_split_idle_spill",input_threshold_bytes=32768,prefill_members=["D1"],decode_members=["D0"])
epochs={x["id"]:x["epoch"]for x in config["native_domains"]};prior={x["id"]:x["epoch"]for x in old["native_domains"]}
assert epochs["local-204-166"]==prior["local-204-166"]and epochs["local-211-167"]!=prior["local-200-167"]
roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text());physical={};policies={};rankmaps={}
plans=json.loads((folder/"standalone_launch.json").read_text())
for key,o in roots.items():
 node=o["host"];TP=8 if node=="166"else 4;PP=2 if node=="166"else 4;K=3 if node=="166"else 1
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 bypid={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(bypid[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 a=o["argv"]
 for flag,value in [("--nnodes","1"),("--tensor-parallel-size",str(TP)),("--pipeline-parallel-size",str(PP)),("--decode-context-parallel-size",str(TP)),("--max-num-batched-tokens","8192")]:assert a[a.index(flag)+1]==value
 assert json.loads(a[a.index("--speculative-config")+1])["num_speculative_tokens"]==K
 assert v["env"]["VLLM_PP_LAYER_PARTITION"]==("42,36"if node=="166"else "22,20,20,16")
 physical[key]=dict(root_same=True,actual_NPU16=True,root=v["root"],proof=ref(j/(key+".owner.stdout")),TP=TP,PP=PP,DCP=TP,K=K)
 path="/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp204"if node=="166"else "local_pp211")
 log=plans[key]["log"]
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(path)+");print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),files={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in p.glob('*.py')})))"
 args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);(j/("native_source_"+node+".stdout")).write_bytes(z.stdout);z.check_returncode();v=json.loads(z.stdout)
 assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0204"if node=="166"else "GLM-COHORT-0211",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 src=p/"runs/GLM-RUN-0204/plugin_src"if node=="166"else r/"plugin_src"
 for name,sha in v["files"].items():assert ref(src/name)["sha256"]==sha
 policies[node]=v
 logf=p/"runs/GLM-RUN-0204/D0_native_resident.stdout"if node=="166"else r/"D1_native_resident.stdout"
 text=logf.read_text();ev=[json.loads(l)for l in text.splitlines()if l.startswith('{"event":')]
 assert ev[0]["event"]=="task_acl_init"and ev[0]["returncode"]==0and not any(x["event"]=="task_acl_finalize"for x in ev)
 assert "Graph capturing finished"in text
 primary=json.loads((p/"runs/GLM-RUN-0190/restored/PP_empty_guard_workers.json").read_text())["markers"][0]
 markers=[json.loads(l.split("GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED ",1)[1])for l in text.splitlines()if "GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED "in l];assert len(markers)==16and all(x==primary for x in markers)
 entries=[]
 for line in text.splitlines():
  match=re.search(r"\(Worker pid=(\d+)\).*world_size=16 rank=(\d+) local_rank=(\d+)",line)
  if match:entries.append(dict(container_pid=int(match[1]),rank=int(match[2]),local_rank=int(match[3])))
 assert len(entries)==16and {x["rank"]for x in entries}==set(range(16))and all(x["rank"]==x["local_rank"]for x in entries)
 mapcode="import pathlib,json,sys;rows=[];a=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()\nfor x in a:\n p=pathlib.Path('/proc/'+str(x['pid']));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity'];ns=next(l for l in(p/'status').read_text().splitlines()if l.startswith('NSpid:')).split()[1:];rows.append(dict(host_pid=x['pid'],identity=x['identity'],container_pid=int(ns[-1])))\nprint(json.dumps(rows))"
 args=["python3","-c",mapcode]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 targets=[x for x in members[key]["owned_targets"]if x["pid"]in members[key]["npu_worker_pids"]]
 z=subprocess.run(args,input=json.dumps(targets).encode(),capture_output=True,timeout=60);z.check_returncode();(j/(key+".NSpid.stdout")).write_bytes(z.stdout);nspids=json.loads(z.stdout)
 assert {x["container_pid"]for x in nspids}=={x["container_pid"]for x in entries}
 mapped=[]
 for x in entries:
  hostrow=next(y for y in nspids if y["container_pid"]==x["container_pid"]);PPindex=x["rank"]//TP;TPindex=x["rank"]%TP
  label="Worker_PP"+str(PPindex)+"_TP"+str(TPindex)+"_DCP"+str(TPindex)+" pid="+str(x["container_pid"])
  assert label in text;mapped.append(dict(x,**hostrow,PP=PPindex,TP=TPindex,DCP=TPindex))
 rankmaps[node]=dict(source=ref(logf),actual_rank_HOST_NPU_map=mapped,NSpid_proof=ref(j/(key+".NSpid.stdout")),limits="init labels and current HOST NPU PID identity, not percollective GPU activity")
oldroots=json.loads((r/"standalone_root_identities.json").read_text());oldmembers=json.loads((r/"standalone_native_members.json").read_text())
assert roots["node0"]==oldroots["node0"]and members["node0"]==oldmembers["node0"]
pre=[json.loads(l)for l in(r/"D1_stop_preflight.stdout").read_text().splitlines()if l.startswith("{")]
stop=[json.loads(l)for l in(r/"D1_stop.stdout").read_text().splitlines()if l.startswith("{")]
known={x["pid"]:x["identity"]for x in json.loads((r/"before_fault_replace_D1.stdout").read_text())["owned_targets"]}
assert pre[0]["event"]=="ownership_preflight"and pre[0]["signals"]==0and pre[0]["npu_workers"]==16
signals=[x for x in stop if"signal"in x]
for x in signals:
 assert x["pid"]in known
 if"identity"in x:assert x["identity"]==known[x["pid"]]
assert signals and stop[-1]["all_original_native_domain_inactive"]and stop[-1]["npu_workers"]==0and stop[-1]["signals_to_unknown"]==stop[-1]["signals_to_inactive"]==0
faultack=json.loads((r/"physical_fault_ack.json").read_text());groups={x["id"]:x for x in faultack["observer"]["groups"]}
assert groups["local-200-167"]["status"]=="fault"and groups["local-204-166"]["status"]=="healthy"and groups["local-204-166"]["epoch"]==epochs["local-204-166"]
cpu=json.loads((p/"jobs/D1-K1-GRAPH2-FULLCLI-CPU5-20261005T0831Z/reduction.json").read_text());assert cpu["CPU_config_VALID"]and cpu["SDKinit_finalize0"]and cpu["current_native32_same_idle_counters"]and cpu["native_dense_Graph_CPU"]["full_native_config"]
assert cpu["exact_fullCLI"]["TP"]==cpu["exact_fullCLI"]["DCP"]==4and cpu["exact_fullCLI"]["PP"]==4and cpu["exact_fullCLI"]["K"]==1
capture=json.loads((folder/"native_dense_capture_proof.json").read_text());assert capture["native_config"]==dict(cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=[2,4,8,16,32],max_cudagraph_capture_size=32)and capture["capture_count4_native_progress"]and ref(capture["source"]["path"])==capture["source"]and capture["operator_math_changes"]==0
assert b"4/4"in Path(capture["source"]["path"]).read_bytes()
local=json.loads((folder/"local_cadence_installed.json").read_text());assert local["markers"][0]["eligibility_conservation"]and local["markers"][0]["native_scheduler"]=="AsyncScheduler"and local["markers"][0]["source_control"]=="issue_budget_scheduler_v5"
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and [x.decode()for x in Path("/proc/"+str(public["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==public["argv"]and ref(public["config"]["path"])==public["config"]
assert owner_alive(json.loads((folder/"identity_observer/process_owner.json").read_text()))
obs=observe(folder/"service_config.json");assert all(x["status"]=="healthy"for x in obs["groups"]);atomic_json(j/"HOST_observer_now.json",obs)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(url):
 with http.open(url,timeout=15)as res:return json.loads(res.read())
health=get("http://127.0.0.1:8000/healthcheck");assert health["status"]=="ok"and health["request_num"]==0
placement=get("http://127.0.0.1:8000/control/replicas");assert len(placement["replicas"])==2and all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]for x in placement["replicas"])
fault=json.loads((state125/"response_owners.json.fault").read_text());assert fault["open"]and all(not x["faulted"]for x in fault["groups"].values())
retired=json.loads((r/"retire210_public.json").read_text());assert retired["SDKinit_finalize0"]and retired["models_signalled"]==0
assert (p/"runs/GLM-RUN-0210/restored/identity_observer/terminal.json").exists()
events=[json.loads(l)for l in(folder/"public.gateway.log").read_text().splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[0]["returncode"]==0and not any(x["event"]=="task_acl_finalize"for x in events)
before=metric(r/"before_166.metrics");after=metric(r/"terminal_166.metrics");assert all(after[k]==v for k,v in before.items()if k.endswith("_total"))
assert after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==after["vllm:kv_cache_usage_perc"]==0
native_final={}
for key,o in roots.items():
 with http.open("http://172.16.10."+o["host"]+":"+str(9081 if o["host"]=="166"else 9900)+"/metrics",timeout=15)as res:raw=res.read()
 f=j/("native_final_"+key+".metrics");f.write_bytes(raw);v=metric(f);assert v["vllm:num_requests_running"]==v["vllm:num_requests_waiting"]==0
 if key=="node1":
  assert v["vllm:generation_tokens_total"]==59and v["vllm:request_success_total"]==5and v["vllm:num_preemptions_total"]==v["vllm:prefix_cache_hits_total"]==v["vllm:external_prefix_cache_hits_total"]==0
 native_final[key]=ref(f)
policy=json.loads((r/"policy_consumption.json").read_text());assert policy["D0_unchanged"]and policy["operator_math_changes"]==0
assert policy["native_D1_selected"]and all(x["serial"]==1and x["cohort"]=="GLM-COHORT-0211"and x["prefill_threshold_tokens"]==1024and not x["fallback"]for x in policy["native_D1_selected"])
atomic_json(j/"public_service_proof.json",public)
limits=["NativeGraph size2 accepted byactual completeCPUconfig/resolver andcapture4/4 fit; actualruntime GPUdispatch/padding reduction/latency benefit unknown","Functional59/5 nativeJSON-SSE typedResponses/STOREprevious16/retrieve/semantics4-42-literal/nativecounts/wire/SDK0, noKEEP/stablecapacity/globalbound","ExactoldD1epoch200 retired,oldSTORE503preleaseRPC; nativeD0epoch204+STORE204/210+allvllmcounters unchanged; no STOREmigration/replay/operators/math/guards changes","Public210SDKfinal0/oldobserverterminal thenpublic211init0active/fullnative32HOSTNSpid rankmap; nativeSDKinit0active isnot finalize0"]
out=dict(at=utc(),run_id=r.name,measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",source_pins=len(pins),active_NPU_ranks=32,physical=physical,native_policy=policies,effective_output_tokens=59,new_native_completed=5,internal_helper_commits=0,uncredited_native_output_tokens=0,D0_new_output_tokens=0,D1_new_output_tokens=59,clients=clients,native_final=native_final,CPU_full_config=ref(p/"jobs/D1-K1-GRAPH2-FULLCLI-CPU5-20261005T0831Z/reduction.json"),native_Graph2_capture=ref(folder/"native_dense_capture_proof.json"),actual_runtime_batch_dispatch_unknown=True,mixed_shape_QoS_pending=True,rankinit=rankmaps,policy_consumption=ref(r/"policy_consumption.json"),public_service=public,D0_epoch_unchanged=True,new_native_epochs=epochs,retired_native_epochs=prior,signals=dict(owned_D1=len(signals),unknown=0,D0=0),SDKclients_init_finalize0=True,retired_public_init_finalize0=True,newnative_public_SDKinit0active=True,STORE_replication=False,request_replay=False,model_operations_during_audit=0,Current=None,limits=limits)
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run211 newD1 Graph2 PP4TP4DCP4K1/nativecapture4fit/retainedD0epoch-STORE204210 andcounters/native59-5/SDK0/source180/native32/fulltyped-JSON-SSE/functionalVALID INCONCLUSIVEperformance",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="native32-epochs-capture-source180-JSONSSEtypedSTOREsemantics59-5-counters-SDK0")],unknowns=limits,decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,outputs=59,native32=True,epochs=epochs)))
