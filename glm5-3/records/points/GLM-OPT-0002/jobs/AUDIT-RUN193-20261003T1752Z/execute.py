from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request,time
from datetime import datetime
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0193";folder=r/"restored";state125=p/"runs/GLM-RUN-0125"
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
for n in["prepare","fault","deploy","pilot","public"]:
 v=json.loads((r/(n+".phase.json")).read_text());assert v["status"]=="succeeded"and v["exit_code"]==0and not v["timed_out"]
summary=json.loads((r/"functional_summary.json").read_text());assert summary["valid"]and summary["functional_acceptance"]and summary["new_native_completed"]==9and summary["frontend_count"]==summary["headless_count"]==1
guard=json.loads((folder/"PP_empty_guard_workers.json").read_text());assert guard["workers"]==32and guard["operator_math_edits"]==0and len(guard["markers"])==32and {x["fixed_source_sha256"]for x in guard["markers"]}=={"3e8cccfae16feac0f8dfbc894ae922193ffd94052734d61ce478440f8b6c4fcb"}
for node in["166","167"]:
 a=metric(r/("before_"+node+".metrics"));b=metric(r/("afterCPU_"+node+".metrics"));assert all(a[k]==b[k]for k in a if k.endswith("_total"))
 rows=[json.loads(l)for l in(r/("CPU_exactplan_"+node+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert rows[0]["event"]=="task_acl_init"and rows[-1]["event"]=="task_acl_finalize"and rows[0]["returncode"]==rows[-1]["returncode"]==0
 c=next(x for x in rows if x["event"]=="fullCLI_PP32_exactplan");assert (c["TP"],c["PP"],c["DCP"],c["world"],c["local_world"])==(8,4,8,32,16)and c["node_rank"]==int(node=="167")and c["workers"]==c["models"]==c["inference"]==c["SDKcommunicators"]==0
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
  native_id="chatcmpl-GLM-RUN-0193-"+str(i)+"-"+row["name"]
  assert chunks and all(x["id"]==native_id for x in chunks);ids.append(native_id)
  inp=21 if row["name"]=="short_D0"else 81932;out=32 if inp==21 else 64
  assert sum(len(c.get("token_ids")or[])for x in chunks for c in x["choices"])==out
  assert len(chunks[0]["prompt_token_ids"])==inp and row["usage"]==dict(prompt_tokens=inp,completion_tokens=out,total_tokens=inp+out)
  assert row["completed"]and row["http_status"]==200and row["owner"]=="D0"
  body=json.loads((d/(row["name"]+".body.json")).read_text());assert body["cache_salt"].startswith(r.name+"-"+str(i)+"-")or body["cache_salt"].startswith(r.name+"-pilot_"+str(i)+"-")
  direct.append(dict(client=i,native_id=native_id,request=row))
  requests.append(row)
assert len(ids)==len(set(ids))==4
before=metric(r/"pilot_before_166.metrics");after=metric(r/"pilot_after_166.metrics");delta={k:after[k]-v for k,v in before.items()if k.endswith("_total")}
for name,want in[("generation_tokens_total",192),("prompt_tokens_total",163906),("request_success_total",4),("prefix_cache_hits_total",0),("external_prefix_cache_hits_total",0),("num_preemptions_total",0)]:assert delta["vllm:"+name]==want
trace=[json.loads(l)for l in(state125/"router_trace.jsonl").read_text().splitlines()];clients={};output=192;completed=4
for mode in["before","retired","final"]:
 v=json.loads((r/("client_"+mode+"_summary.json")).read_text());assert v["valid"]
 rows=[json.loads(l)for l in(r/("client_"+mode+".stdout")).read_text().splitlines()if l.startswith('{"event":')];assert rows[0]["event"]=="task_acl_init"and rows[-1]["event"]=="task_acl_finalize"and rows[0]["returncode"]==rows[-1]["returncode"]==0
 for n,row in enumerate(v["requests"]):
  for k in["body","wire"]:assert ref(row[k]["path"])==row[k]
  header=r.name+"-client_"+mode+"_"+row["name"]+"_"+str(n);leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==header]
  if row["status"]==503:assert row["rejected_before_lease_and_RPC"]and not leases;continue
  assert len(leases)==1;lease=leases[0];assert lease["lease_id"]==row["lease_id"]and lease["replica"]==row["native_owner"]and lease["body_sha256"]==row["body"]["sha256"]
  wire=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];release=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]]
  assert len(wire)==len(release)==1and wire[0]["audit_error"]is None and release[0]["released"]and not release[0]["backend_failure"]
  assert ref(row["native_wire"]["path"])==row["native_wire"]and Path(row["wire"]["path"]).read_bytes()==Path(row["native_wire"]["path"]).read_bytes()
  if row.get("effective_output_tokens",0)>0:completed+=1
 if mode!="retired":assert sum(x["generation_tokens_total"]for x in v["native_delta"].values())==v["effective_output_tokens"]
 if mode!="final":assert v["effective_output_tokens"]==0
 output+=v["effective_output_tokens"];clients[mode]=dict(summary=ref(r/("client_"+mode+"_summary.json")),outputs=v["effective_output_tokens"],requests=len(v["requests"]),SDKinit_finalize0=True)
assert completed==9and output==summary["effective_output_tokens"]
pre=json.loads((r/"client_before_summary.json").read_text())
for key,runid,name in[("D0","GLM-RUN-0190","newD0_create"),("D1","GLM-RUN-0184","newD1_create")]:
 prior=json.loads((p/"runs"/runid/"client_final_summary.json").read_text());orig=next(x for x in prior["requests"]if x["name"]==name);row=next(x for x in pre["requests"]if x["name"]==key+"_retained");assert row["affinity_applied"]and row["native_owner"]==key and Path(row["wire"]["path"]).read_bytes()==Path(orig["wire"]["path"]).read_bytes()
for mode in["retired","final"]:
 v=json.loads((r/("client_"+mode+"_summary.json")).read_text())
 for key in["D0","D1"]:
  for suffix in["_retired_get","_retired_previous"]:
   row=next(x for x in v["requests"]if x["name"]==key+suffix);assert row["status"]==503and row["rejected_before_lease_and_RPC"]
final=json.loads((r/"client_final_summary.json").read_text());sem=[x for x in final["requests"]if x.get("semantic_pass")];assert len(sem)==3
for row,answer in zip(sem,["4","42","GLM_OK_731"]):
 v=json.loads(Path(row["wire"]["path"]).read_text());c=v["choices"][0];assert c["message"]["content"].strip()==answer and c["finish_reason"]=="stop"and len(c["token_ids"])==row["effective_output_tokens"]==v["usage"]["completion_tokens"]
base_row=next(x for x in final["requests"]if x["name"]=="joint_create");getrow=next(x for x in final["requests"]if x["name"]=="joint_retrieve")
basev=json.loads(Path(base_row["wire"]["path"]).read_text());assert basev["id"]=="resp_glm_run193_joint_new"and basev["usage"]["output_tokens"]==32and json.loads(Path(getrow["wire"]["path"]).read_text())==basev
childrow=next(x for x in final["requests"]if x["name"]=="joint_previous");getchild=next(x for x in final["requests"]if x["name"]=="joint_child_retrieve")
events=[]
for frame in Path(childrow["wire"]["path"]).read_bytes().replace(bytes([13,10]),bytes([10])).split(bytes([10,10])):
 data=bytes([10]).join(l[5:].lstrip(b" ")for l in frame.splitlines()if l.startswith(b"data:"))
 if data:events.append(json.loads(data))
assert [x["sequence_number"]for x in events]==list(range(len(events)))and events[-1]["type"]=="response.completed"and len([x for x in events if x["type"]=="response.completed"])==1
child=events[-1]["response"];assert child==json.loads(Path(getchild["wire"]["path"]).read_text())and child["previous_response_id"]==basev["id"]and child["usage"]["output_tokens"]==16
config,_=checked_config(folder/"service_config.json");assert len(config["native_domains"])==1and config["native_domains"][0]["id"]=="joint-193"and config["native_domains"][0]["members"]==["D0"]
material=json.loads((folder/"native_engines_resident.json").read_text());engine=material["engines"][0];assert len(material["engines"])==1and set(engine["roots"])==set(engine["members"])==set(engine["plans"])=={"node0","node1"}
roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text());physical={};policies={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 known={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(known[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 assert v["env"]["VLLM_PP_LAYER_PARTITION"]=="22,20,20,16"and v["env"]["HCCL_LOGIC_SUPERPOD_ID"]=="0"and v["env"]["GLM_ISSUE_BUDGET_COHORT"]=="GLM-COHORT-0193"
 for flag,want in[("--tensor-parallel-size","8"),("--pipeline-parallel-size","4"),("--decode-context-parallel-size","8"),("--nnodes","2"),("--node-rank",str(int(key=="node1")))]:assert o["argv"][o["argv"].index(flag)+1]==want
 assert ("--headless"in o["argv"])==(key=="node1")and json.loads(o["argv"][o["argv"].index("--speculative-config")+1])["num_speculative_tokens"]==3
 physical[key]=dict(root_same=True,actual_NPU16=True,root=v["root"],proof=ref(j/(key+".owner.stdout")))
 path="/data/tiankuan/wio/glm52-pd/deploy/plugins/coupled_pp193";log="/data/tiankuan/wio/glm52-pd/deploy/logs/coupled_pp193_"+o["host"]+".log"
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(path)+");print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),files={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in p.glob('*.py')},SDK_events=[json.loads(l)for l in Path("+repr(log)+").read_text(errors='replace').splitlines()if l.startswith('{'+chr(34)+'event'+chr(34)+':')])))"
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);(j/("native_source_"+o["host"]+".stdout")).write_bytes(z.stdout);(j/("native_source_"+o["host"]+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0193",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 for name,sha in v["files"].items():assert sha==ref(r/"plugin_src"/name)["sha256"]
 assert v["SDK_events"][0]["event"]=="task_acl_init"and v["SDK_events"][0]["returncode"]==0
 policies[o["host"]]=v
 oldmembers=json.loads((r/"standalone_native_members.json").read_text())[key];known={x["pid"]:x["identity"]for x in oldmembers["owned_targets"]}
 pre=[json.loads(l)for l in(r/("stop_preflight_"+key+".stdout")).read_text().splitlines()if l.startswith("{")]
 stop=[json.loads(l)for l in(r/("stop_"+key+".stdout")).read_text().splitlines()if l.startswith("{")]
 assert pre[0]["event"]=="ownership_preflight"and pre[0]["signals"]==0and pre[0]["npu_workers"]==16
 for x in stop:
  if "signal"in x:
   assert x["pid"]in known
   if "identity"in x:assert x["identity"]==known[x["pid"]]
 assert stop[-1]["all_original_native_domain_inactive"]and stop[-1]["npu_workers"]==0and stop[-1]["signals_to_unknown"]==stop[-1]["signals_to_inactive"]==0
faultack=json.loads((r/"physical_fault_ack.json").read_text());assert all(x["status"]=="fault"for x in faultack["observer"]["groups"])and faultack["both_exact_old_native_domains_retired"]
retire=json.loads((r/"retire190_public.json").read_text());assert retire["SDKinit_finalize0"]and (p/"runs/GLM-RUN-0190/restored/identity_observer/terminal.json").exists()
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and [x.decode()for x in Path("/proc/"+str(public["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==public["argv"]and ref(public["config"]["path"])==public["config"]
owner=json.loads((folder/"identity_observer/process_owner.json").read_text());assert owner_alive(owner)
obs=observe(folder/"service_config.json");assert len(obs["groups"])==1and obs["groups"][0]["status"]=="healthy";atomic_json(j/"HOST_observer_now.json",obs)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(url):
 with http.open(url,timeout=15)as z:return json.loads(z.read())
health=get("http://127.0.0.1:8000/healthcheck");placement=get("http://127.0.0.1:8000/control/replicas");assert health["status"]=="ok"and health["request_num"]==0and len(placement["replicas"])==1and all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]and x["work_ranking_calibrated"]for x in placement["replicas"])
with http.open("http://172.16.10.166:9081/metrics",timeout=15)as z:raw=z.read()
f=j/"terminal_native.metrics";f.write_bytes(raw);v=metric(f);assert v["vllm:num_requests_running"]==v["vllm:num_requests_waiting"]==0and v["vllm:generation_tokens_total"]==output and v["vllm:request_success_total"]==9and v["vllm:num_preemptions_total"]==0
local=json.loads((folder/"local_cadence_installed.json").read_text());assert len(local)==1and local[0]["source_control"]=="issue_budget_scheduler_v5"and local[0]["native_math_changes"]==0
acks=[json.loads(l)for l in(folder/"public.gateway.log").read_text().splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[0]["returncode"]==0and not any(x["event"]=="task_acl_finalize"for x in acks)
marker=public["container_marker"];proof=dict(at=utc(),host=public["host"],argv=public["argv"],container=dict(host="166",namespace="glm52-single_container",pid=marker["pid"],identity={k:marker[k]for k in["pid","boot_id","start_ticks"]},argv=public["argv"],port=8000,config=public["config"],native_domains=config["native_domains"]),marker=marker,placement=placement,SDK_init0=True,retained=True,state_dir=str(state125));atomic_json(j/"public_service_proof.json",proof)
artifacts=[ref(f)for f in sorted(r.rglob("*"))if f.is_file()and f.name not in["manifest.json","state.json","controller.log","token_memo_stats.json","token_memo_trace.jsonl","public.gateway.log"]and "identity_observer"not in f.parts];atomic_json(j/"artifact_index.json",artifacts)
limits=["Joint32 TP8PP4DCP8/noEP-noKV/DP1/K3/oneAPI166+167headless/PP22,20,20,16 actualfit Graph32/fullcoldpair/nativeIDs/semantic4-42-literal/typedResponsesSTOREprevious-retrieve/SDK0/source113/native32/guard32; nooperator-MTP-KV math changes",
"Existingcompiler includesbothphysical roots-NPU ancestry inoneepoch/fault/STOREdomain; actualHOSTobserver verifiesboth healthy. Old190184STORE503preleaseRPC afterexactownedretirement/public190SDKfinal0 thennew193V12/state125; noSTOREreplication/replay",
"Native fullCLI/executor rankallocation CPUacceptedbothhosts, process32realfit; TPwithinhost expectation basednative ranklayout, notpercollective GPUtrace. No capacity/isolatedgain/KEEP/stableglobalbound; resourcecategory differs previous independentengines and131TP16PP2",
"Protocol/operator paths unchanged; priorfullAPI E2E139147149-151168175179184190 andpriorjoint133 reused; this boundedfitclientdoesnot claim rerunallfunctions/customIDrace/nativebackgroundprogress"]
out=dict(at=utc(),run_id=r.name,measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",Current=None,source_count=len(pins),native32_owned_idle=True,physical=physical,native_policy=policies,effective_output_tokens=output,native_completed=9,helper_outputs=0,uncredited_outputs=0,pairedcold_HTTP_overlap_s=pilot["cold_HTTP_overlap_s"],direct=direct,clients=clients,public=proof,native_domains=config["native_domains"],SDKclients_init_finalize0=True,oldpublicSDKinit_finalize0=True,newpublicSDKinit0active=True,native_operator_math_changes=0,STORE_replication=False,request_replay=False,model_operations_during_audit=0,artifact_index=ref(j/"artifact_index.json"),limits=limits)
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Joint193 TP8PP4DCP8 native32 fit-functional VALID/"+str(output)+" newoutputs/9complete/fullnativeIDs-wire-SDK-STORE-source, INCONCLUSIVE capacity",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="actualjoint32 fit nativewire-ID-counts-SDK-STORE-physicalepoch")],unknowns=limits,decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,output_tokens=output,native_domains=config["native_domains"])))
