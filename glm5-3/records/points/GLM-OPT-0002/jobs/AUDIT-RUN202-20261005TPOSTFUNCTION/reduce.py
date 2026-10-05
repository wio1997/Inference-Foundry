from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request,time
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0202";folder=r/"restored";state125=p/"runs/GLM-RUN-0125"
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
for st in spec["stages"]:assert st["sources"]==pins
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(x["path"])["sha256"]==x["sha256"]
assert ref(r/"controller_spec.json")["sha256"]==m["spec_sha256"]==s["spec_sha256"]
for n in["prepare","fault","restore","pilot","switch","finalize"]:
 v=json.loads((r/(n+".phase.json")).read_text());assert v["status"]=="succeeded"and v["exit_code"]==0and not v["timed_out"]
summary=json.loads((r/"functional_summary.json").read_text());assert summary["valid"]and summary["functional_acceptance"]and summary["new_native_completed"]==9and summary["D1_new_output_tokens"]==0
guard=json.loads((folder/"PP_empty_guard_workers.json").read_text());assert guard["workers"]==16 and guard["operator_math_edits"]==0 and len(guard["markers"])==16 and len({x["fixed_source_sha256"]for x in guard["markers"]})==1
pilot=json.loads((r/"pilot_SDK_summary.json").read_text());assert pilot["measurement_valid"]and pilot["SDKinit_finalize0"]and pilot["output_tokens"]==192and pilot["cold_HTTP_overlap_s"]>0
ids=[];requests=[];direct=[]
for i in[0,1]:
 d=r/("pilot_"+str(i));v=json.loads((d/"pilot_summary.json").read_text());assert v["measurement_valid"]and v["effective_public_output_tokens"]==96and len(v["requests"])==2
 events=[json.loads(l)for l in(r/("D0_pilot_"+str(i)+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
 for row in v["requests"]:
  raw=Path(row["wire"]["path"]).read_bytes();assert ref(row["wire"]["path"])==row["wire"]and ref(d/(row["name"]+".body.json"))["sha256"]==row["request_body_sha256"]
  ob=NativeSSEObserver(max_bytes=2097152,collect_contract=True);ob.feed(raw);contract=ob.contract();assert contract==row["contract"]and contract["done"]and not contract["unknown"]and not contract["native_error"]and contract["finish_reasons"]=={"0":"length"}
  chunks=[]
  for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
   data=b"\n".join(l[5:].lstrip(b" ")for l in frame.splitlines()if l.startswith(b"data:"))
   if data and data!=b"[DONE]":chunks.append(json.loads(data))
  native_id="chatcmpl-GLM-RUN-0202-"+str(i)+"-"+row["name"]
  assert chunks and all(x["id"]==native_id for x in chunks);ids.append(native_id)
  inp=21 if row["name"]=="short_D0"else 81932;out=32 if inp==21 else 64
  assert sum(len(c.get("token_ids")or[])for x in chunks for c in x["choices"])==out
  assert len(chunks[0]["prompt_token_ids"])==inp and row["usage"]==dict(prompt_tokens=inp,completion_tokens=out,total_tokens=inp+out)
  assert row["completed"]and row["http_status"]==200and row["owner"]=="D0"
  body=json.loads((d/(row["name"]+".body.json")).read_text());assert body["cache_salt"].startswith(r.name+"-"+str(i)+"-")or body["cache_salt"].startswith(r.name+"-pilot_"+str(i)+"-")
  direct.append(dict(client=i,native_id=native_id,request=row))
  requests.append(row)
assert len(ids)==len(set(ids))==4
counters={}
for node in["166","167"]:
 before=metric(r/("pilot_before_"+node+".metrics"));after=metric(r/("pilot_after_"+node+".metrics"));delta={k:after[k]-v for k,v in before.items()if k.endswith("_total")}
 assert after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==0
 if node=="167":assert all(x==0for x in delta.values())
 else:
  for name,expected in[("generation_tokens_total",192),("prompt_tokens_total",163906),("request_success_total",4),("prefix_cache_hits_total",0),("external_prefix_cache_hits_total",0),("num_preemptions_total",0)]:assert delta["vllm:"+name]==expected
 counters[node]=delta
trace=[json.loads(l)for l in(state125/"router_trace.jsonl").read_text().splitlines()]
clients={};output=192;public_completed=0
for mode in["before","survivor","final"]:
 v=json.loads((r/("client_"+mode+"_summary.json")).read_text());assert v["valid"]
 events=[json.loads(l)for l in(r/("client_"+mode+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
 for n,row in enumerate(v["requests"]):
  for k in["body","wire"]:assert ref(row[k]["path"])==row[k]
  if row["status"]==503:assert row["rejected_before_lease_and_RPC"];continue
  assert ref(row["native_wire"]["path"])==row["native_wire"]and Path(row["wire"]["path"]).read_bytes()==Path(row["native_wire"]["path"]).read_bytes()
  header=r.name+"-client_"+mode+"_"+row["name"]+"_"+str(n)
  leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==header];assert len(leases)==1
  lease=leases[0];assert lease["lease_id"]==row["lease_id"]and lease["replica"]==row["native_owner"]and lease["body_sha256"]==row["body"]["sha256"]
  wire=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];release=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]]
  assert len(wire)==len(release)==1and wire[0]["audit_error"]is None and release[0]["released"]and not release[0]["backend_failure"]
  if row.get("effective_output_tokens",0)>0:public_completed+=1
 assert sum(x["generation_tokens_total"]for x in v["native_delta"].values())==v["effective_output_tokens"]
 if mode!="final":assert v["effective_output_tokens"]==0
 else:
  assert v["native_delta"]["D1"]["generation_tokens_total"]==0
  semantics=[x for x in v["requests"]if x.get("semantic_pass")];assert len(semantics)==3
  for row,answer in zip(semantics,["4","42","GLM_OK_731"]):
   value=json.loads(Path(row["wire"]["path"]).read_text());assert value["choices"][0]["message"]["content"].strip()==answer and value["choices"][0]["finish_reason"]=="stop"
  base=next(x for x in v["requests"]if x["name"]=="newD0_create");got=next(x for x in v["requests"]if x["name"]=="newD0_retrieve")
  assert json.loads(Path(base["wire"]["path"]).read_text())==json.loads(Path(got["wire"]["path"]).read_text())
  basev=json.loads(Path(base["wire"]["path"]).read_text());assert basev["id"]=="resp_glm_run202_D0_new"and basev["usage"]["output_tokens"]==32
  child=next(x for x in v["requests"]if x["name"]=="newD0_previous");retrieved=next(x for x in v["requests"]if x["name"]=="newD0_child_retrieve")
  childv=json.loads(Path(retrieved["wire"]["path"]).read_text());assert childv["previous_response_id"]==basev["id"]and childv["usage"]["output_tokens"]==16
 output+=v["effective_output_tokens"];clients[mode]=dict(summary=ref(r/("client_"+mode+"_summary.json")),requests=len(v["requests"]),outputs=v["effective_output_tokens"],SDK0=True)
assert public_completed==5and output==summary["effective_output_tokens"]
config,_=checked_config(folder/"service_config.json");old,_=checked_config(r/"service_config.json")
assert config["placement"]==dict(kind="shape_split",input_threshold_bytes=32768,prefill_members=["D1"],decode_members=["D0"])
epochs={x["id"]:x["epoch"]for x in config["native_domains"]};prior={x["id"]:x["epoch"]for x in old["native_domains"]}
assert epochs["local-200-167"]==prior["local-200-167"]and epochs["local-202-166"]!=prior["local-200-166"]
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
 path="/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp202"if node=="166"else "local_pp200")
 log=plans[key]["log"]
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(path)+");print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),files={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in p.glob('*.py')})))"
 args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);(j/("native_source_"+node+".stdout")).write_bytes(z.stdout);z.check_returncode();v=json.loads(z.stdout)
 assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0202"if node=="166"else "GLM-COHORT-0200",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 src=r/"plugin_src"if node=="166"else p/"runs/GLM-RUN-0200/plugin_src"
 for name,sha in v["files"].items():assert ref(src/name)["sha256"]==sha
 policies[node]=v
 logf=r/"D0_native_resident.stdout"if node=="166"else p/"runs/GLM-RUN-0200/native_resident_node1.stdout"
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
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and [x.decode()for x in Path("/proc/"+str(public["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==public["argv"]and ref(public["config"]["path"])==public["config"]
owner=json.loads((folder/"identity_observer/process_owner.json").read_text());assert owner_alive(owner)
obs=observe(folder/"service_config.json");assert all(x["status"]=="healthy"for x in obs["groups"]);atomic_json(j/"HOST_observer_now.json",obs)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(url):
 with http.open(url,timeout=15)as res:return json.loads(res.read())
health=get("http://127.0.0.1:8000/healthcheck");assert health["status"]=="ok"and health["request_num"]==0
placement=get("http://127.0.0.1:8000/control/replicas");assert len(placement["replicas"])==2and all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]for x in placement["replicas"])
fault=json.loads((state125/"response_owners.json.fault").read_text());assert fault["open"]and all(not x["faulted"]for x in fault["groups"].values())
pre=[json.loads(l)for l in(r/"D0_stop_preflight.stdout").read_text().splitlines()if l.startswith("{")]
stop=[json.loads(l)for l in(r/"D0_stop.stdout").read_text().splitlines()if l.startswith("{")]
oldroots=json.loads((r/"standalone_root_identities.json").read_text());oldmembers=json.loads((r/"standalone_native_members.json").read_text())
known={x["pid"]:x["identity"]for x in json.loads((r/"before_fault_D0.stdout").read_text())["owned_targets"]}
assert pre[0]["event"]=="ownership_preflight"and pre[0]["signals"]==0and pre[0]["npu_workers"]==16
signals=[x for x in stop if "signal"in x]
for x in signals:
 assert x["pid"]in known
 if "identity"in x:assert x["identity"]==known[x["pid"]]
assert signals and stop[-1]["all_original_native_domain_inactive"]and stop[-1]["npu_workers"]==0and stop[-1]["signals_to_unknown"]==stop[-1]["signals_to_inactive"]==0
assert roots["node1"]==oldroots["node1"]and members["node1"]==oldmembers["node1"]
faultack=json.loads((r/"physical_fault_ack.json").read_text());g={x["id"]:x for x in faultack["observer"]["groups"]}
assert g["local-200-166"]["status"]=="fault"and g["local-200-167"]["status"]=="healthy"
assert g["local-200-167"]["epoch"]==epochs["local-200-167"]
cpu=json.loads((p/"jobs/PP2-SHORT-NATIVE-CPU-REDUCE2-20261005T0543Z/reduction.json").read_text());assert cpu["CPU_config_VALID"]and cpu["SDKinit_finalize0"]and cpu["current_native32_same_idle_counters"]and not cpu["replayed_nativeCPU_config_calls"]
assert cpu["exact_fullCLI"]["TP"]==cpu["exact_fullCLI"]["DCP"]==8and cpu["exact_fullCLI"]["PP"]==2and cpu["exact_fullCLI"]["world"]==16and cpu["exact_fullCLI"]["K"]==3
local=json.loads((folder/"local_cadence_installed.json").read_text());assert local["actual_APPLIED_still_pending"]and len(local["markers"])==1
assert local["markers"][0]["eligibility_conservation"]and local["markers"][0]["source_control"]=="issue_budget_scheduler_v5"
assert local["markers"][0]["native_scheduler"]=="AsyncScheduler"and local["markers"][0]["geometry"]["DP"]==1and local["markers"][0]["native_math_changes"]==0
assert roots["node0"]["argv"][roots["node0"]["argv"].index("--scheduler-cls")+1]=="issue_budget_scheduler_v5.BudgetScheduler"
for key,K in[("node0",3),("node1",1)]:assert json.loads(roots[key]["argv"][roots[key]["argv"].index("--speculative-config")+1])["num_speculative_tokens"]==K
retired=json.loads((r/"retire200_public.json").read_text());assert retired["SDKinit_finalize0"]and retired["models_signalled"]==0
assert (p/"runs/GLM-RUN-0200/restored/identity_observer/terminal.json").exists()
log=(folder/"public.gateway.log").read_text();events=[json.loads(l)for l in log.splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[0]["returncode"]==0and not any(x["event"]=="task_acl_finalize"for x in events)
native_final={}
for key,o in roots.items():
 with http.open("http://172.16.10."+o["host"]+":"+str(9081 if o["host"]=="166"else 9900)+"/metrics",timeout=15)as response:raw=response.read()
 f=j/("terminal_"+key+".metrics");f.write_bytes(raw);v=metric(f);assert v["vllm:num_requests_running"]==v["vllm:num_requests_waiting"]==0
 if o["host"]=="166":assert v["vllm:generation_tokens_total"]==output and v["vllm:request_success_total"]==9and v["vllm:num_preemptions_total"]==0
 native_final[key]=ref(f)
samples=json.loads((r/"pilot_observations.json").read_text());values=[]
for row in samples:assert ref(row["metrics"]["path"])==row["metrics"];values.append(metric(row["metrics"]["path"]))
pressure={k:max(x[k]for x in values)for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]}
policy=json.loads((r/"policy_consumption.json").read_text());assert policy["operator_math_changes"]==0and policy["D1_unchanged"]
other_guard=json.loads((p/"runs/GLM-RUN-0200/restored/PP_empty_guard_workers.json").read_text())
assert other_guard["workers"]==32and len(other_guard["markers"])==32and {x["fixed_source_sha256"]for x in guard["markers"]}=={x["fixed_source_sha256"]for x in other_guard["markers"]}
for mode in["before","survivor","final"]:
 c=json.loads((r/("client_"+mode+"_summary.json")).read_text())
 retained=[x for x in c["requests"]if x["name"]=="D1_retained"];assert retained
 # Resolve historical authoritative STORE source without relying on request ordinal.
 priorclient=json.loads((p/"runs/GLM-RUN-0200/client_final_summary.json").read_text());base=next(x for x in priorclient["requests"]if x["name"]=="D1_create")
 for x in retained:assert Path(x["wire"]["path"]).read_bytes()==Path(base["wire"]["path"]).read_bytes()and x["native_owner"]=="D1"and x["affinity_applied"]
 if mode=="before":assert any(x["name"]=="D0_retained"and x["status"]==200for x in c["requests"])
 else:assert all(next(x for x in c["requests"]if x["name"]==name)["status"]==503for name in["retired200_get","retired200_previous"])
 for row in c["requests"]:
  if row.get("semantic_pass"):
   val=json.loads(Path(row["wire"]["path"]).read_text());assert len(val["choices"][0]["token_ids"])==row["effective_output_tokens"]==val["usage"]["completion_tokens"]
priorD0=p/"runs/GLM-RUN-0200/client_final_summary.json";assert ref(priorD0)["sha256"]=='32ada6815a0673356610889f0261f8ec4f27be11eb42ab146b99429a83e54cc3'
baseD0=next(x for x in json.loads(priorD0.read_text())["requests"]if x["name"]=="D0_create")
beforeclient=json.loads((r/"client_before_summary.json").read_text());row=next(x for x in beforeclient["requests"]if x["name"]=="D0_retained")
assert Path(row["wire"]["path"]).read_bytes()==Path(baseD0["wire"]["path"]).read_bytes()and row["native_owner"]=="D0"and row["affinity_applied"]
for mode in["survivor","final"]:
 c=json.loads((r/("client_"+mode+"_summary.json")).read_text())
 for name in["retired200_get","retired200_previous"]:
  row=next(x for x in c["requests"]if x["name"]==name);assert row["status"]==503and row["rejected_before_lease_and_RPC"]
# Counter continuity for retained D1 throughout migration, accounting historical generations separately.
beforeD1=metric(r/"before_167.metrics");afterD1=metric(j/"terminal_node1.metrics")
assert all(afterD1[k]==v for k,v in beforeD1.items()if k.endswith("_total"))
assert afterD1["vllm:num_requests_running"]==afterD1["vllm:num_requests_waiting"]==afterD1["vllm:kv_cache_usage_perc"]==0

marker=public["container_marker"];container=dict(host="166",namespace="glm52-single_container",pid=marker["pid"],identity=dict(pid=marker["pid"],boot_id=marker["boot_id"],start_ticks=marker["start_ticks"]),argv=public["argv"],port=8000,config=public["config"],native_domains=config["native_domains"])
proof=dict(at=utc(),host=public["host"],container=container,argv=public["argv"],marker=marker,placement=placement,SDK_init0=True,retained=True,state_dir=str(state125));atomic_json(j/"public_service_proof.json",proof)
artifacts=[ref(f)for f in sorted(r.rglob("*"))if f.is_file()and f.name not in["manifest.json","state.json","controller.log"]and "identity_observer"not in f.parts and f.name not in["token_memo_stats.json","token_memo_trace.jsonl","public.gateway.log"]];atomic_json(j/"artifact_index.json",artifacts)
limits=["D0 only new PP2TP8DCP8/K3/partition42,36/Graph32/8192t1024c1serial1 native V5 fullfit/API/STORE; retained200 D1 PP4TP4DCP4/K1/epoch/STORE/source/policy exact. No native state/math/KV guard edits",
"CPU2 readonly reuses actualCPU1 fullCLI SDK0, no config replay; CPU1 overallINVALID processCPUquery .02seconds fixture preserved; native inference counters unchanged",
"All32 current HOST NPU identity and nativeinit rank/PP/TP/DCP labels, not percollective GPU trace",
"Finite pairedcold192 + semantics/STORE59 outputs with fullnative IDs-wire-usage-DONE/SDK0 and actualretirement prelease503/retainedSTORE; activepublicSDKinit0 is not finalize0",
"ChangedD0 geometry/physicalepoch and newtrajectories notisolatedcausalgain or capacityKEEP; CurrentNone/stablecapacity/globalboundunknown"]

out=dict(at=utc(),run_id=r.name,measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",source_pins=len(pins),active_NPU_ranks=32,physical=physical,native_policy=policies,effective_output_tokens=output,new_native_completed=9,internal_helper_commits=0,uncredited_native_output_tokens=0,D1_new_output_tokens=0,D0_new_output_tokens=output,pairedcold_HTTP_overlap_s=pilot["cold_HTTP_overlap_s"],direct_native=direct,clients=clients,native_pilot_counters=counters,native_pressure=pressure,native_final=native_final,local_cadence_installed=local,CPU_full_config=ref(p/"jobs/PP2-SHORT-NATIVE-CPU-REDUCE2-20261005T0543Z/reduction.json"),mixed_shape_QoS_pending=True,rankinit=rankmaps,policy_consumption=ref(r/"policy_consumption.json"),PP_empty_guard_workers=ref(folder/"PP_empty_guard_workers.json"),public_service=proof,D1_epoch_unchanged=True,new_native_epochs=epochs,retired_native_epochs=prior,signals=dict(owned_D0=len(signals),unknown=0,D1=0),SDKclients_init_finalize0=True,retired_public_init_finalize0=True,newnative_public_SDKinit0active=True,STORE_replication=False,request_replay=False,model_operations_during_audit=0,artifact_index=ref(j/"artifact_index.json"),limits=limits)
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run202 newD0 PP2TP8DCP8/allocated8192t1024 fit/coldpair192 + public nativeJSON-SSE-STORE semantics/"+str(output)+" completeoutputs/SDK0/source/native32idle/nohelpers, functional VALID INCONCLUSIVE capacity",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="actualnativeallIDs-bodywire-tokenIDs-usage-DONE/cost/SDK/epoch/STORE")],unknowns=limits,decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,output_tokens=output,pressure=pressure,epochs=epochs)))
