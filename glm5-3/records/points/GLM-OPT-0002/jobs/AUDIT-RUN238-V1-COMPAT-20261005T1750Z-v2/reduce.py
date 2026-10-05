from pathlib import Path
import json,hashlib,sys,subprocess,shlex,urllib.request
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0238"
sys.path.insert(0,str(p/"runs/GLM-RUN-0228/runtime_bundle"))
from phase_runner import utc,atomic_json,same_process
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def metrics(f):
 out={}
 for l in Path(f).read_text().splitlines():
  if l.startswith("vllm:"):
   k=l.split("{")[0].split()[0];out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"and state["completed_stages"]==["prepare","function"]and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text());m=json.loads((r/"manifest.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==177and ref(r/"controller_spec.json")["sha256"]==state["spec_sha256"]==m["spec"]["sha256"]
for x in pins:assert ref(x["path"])["sha256"]==x["sha256"]and Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()
summary=json.loads((r/"compatibility_summary.json").read_text());assert summary["functional_acceptance"]and summary["actual_completed_requests"]==4and summary["effective_output_tokens"]==41and summary["D0_fault_preserved"]
tools=summary["tools"];assert tools["valid"]and [x["name"]for x in tools["requests"]]==["auto","required","named","none"]and tools["effective_output_tokens"]==41and tools["cleanup"]["native_idle"]
trace=[json.loads(l)for l in(p/"runs/GLM-RUN-0125/router_trace.jsonl").read_text().splitlines()];wireproof=[]
for row in tools["requests"]:
 assert row["valid"]and row["status"]==200and row["done"]and len(row["token_ids"])==row["usage"]["completion_tokens"]and row["usage"]["total_tokens"]==row["usage"]["prompt_tokens"]+row["usage"]["completion_tokens"]
 assert hashlib.sha256(json.dumps(row["token_ids"]).encode()).hexdigest()==row["token_ids_sha256"]and ref(row["wire"]["path"])==row["wire"]
 if row["name"]!="none":assert row["tools"]==[dict(name="get_weather",arguments='{"city": "Shanghai"}')]
 else:assert not row["tools"]and row["content_chars"]>0
 header=r.name+"-tool-D1_compat-"+row["name"];leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==header];assert len(leases)==1and leases[0]["replica"]=="D1"
 body=r/("tools_D1_compat/"+row["name"]+".body.json");assert ref(body)["sha256"]==leases[0]["body_sha256"];b=json.loads(body.read_text());assert b["return_token_ids"]and b["chat_template_kwargs"]=={"enable_thinking":False}
 wires=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==leases[0]["lease_id"]];release=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==leases[0]["lease_id"]];assert len(wires)==len(release)==1and wires[0]["audit_error"]is None and release[0]["released"]and not release[0]["backend_failure"]and Path(wires[0]["wire_path"]).read_bytes()==Path(row["wire"]["path"]).read_bytes()
 raw=Path(row["wire"]["path"]).read_bytes()
 if row["stream"]:
  vals=[]
  for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
   data=b"\n".join(l[5:].lstrip(b" ")for l in frame.splitlines()if l.startswith(b"data:"))
   if data and data!=b"[DONE]":vals.append(json.loads(data))
  ids=[t for f in vals for c in f.get("choices",[])for t in c.get("token_ids",[])]
 else:
  value=json.loads(raw);ids=[t for c in value["choices"]for t in c.get("token_ids",[])]
 assert ids==row["token_ids"]
 wireproof.append(dict(name=row["name"],native_owner="D1",body=ref(body),wire=row["wire"],token_ids_sha256=row["token_ids_sha256"],outputs=len(ids)))
delta={}
for key,node in[("D0","node0"),("D1","node1")]:
 a=metrics(r/("compat_before_"+node+".metrics"));b=metrics(r/("compat_after_"+node+".metrics"));delta[key]={k:b[k]-a[k]for k in a if k.endswith("_total")}
 assert b["vllm:num_requests_running"]==b["vllm:num_requests_waiting"]==b["vllm:kv_cache_usage_perc"]==0
assert all(x==0for x in delta["D0"].values())and delta["D1"]["vllm:generation_tokens_total"]==41and delta["D1"]["vllm:prompt_tokens_total"]==646and delta["D1"]["vllm:request_success_total"]==4and delta["D1"]["vllm:num_preemptions_total"]==0
ev=[json.loads(l)for l in(r/"native_tools.stdout").read_text().splitlines()if l.startswith('{"event":')];assert ev[0]["event"]=="task_acl_init"and ev[-1]["event"]=="task_acl_finalize"and ev[0]["returncode"]==ev[-1]["returncode"]==0
exe=json.loads((r/"native_tools_execution.json").read_text());assert exe["status"]=="succeeded"and exe["exit_code"]==0and not exe["native_client_alive"]and exe["signal_attempts"]==[]and not same_process(exe["native_client"])
previous=p/"runs/GLM-RUN-0237";ps=json.loads((previous/"state.json").read_text());assert ps["status"]=="failed"and ps["failure_phase"]=="function"and not same_process(ps["owner"])
assert "exact task client script required"in(previous/"function.log").read_text()and not(previous/"native_tools_execution.json").exists()and not(previous/"native_tools.stdout").exists()
assert not any((x.get("request_header_id")or"").startswith("GLM-RUN-0237-")for x in trace)
for node in["node0","node1"]:
 a=metrics(previous/("compat_before_"+node+".metrics"));b=metrics(r/("compat_before_"+node+".metrics"));assert all(b[k]==v for k,v in a.items()if k.endswith("_total"))
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());physical={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout);ids={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 physical[key]=dict(root_same=True,NPU16_same=True,proof=ref(j/(key+".owner.stdout")))
 if key=="node1":assert v["env"].get("VLLM_USE_V2_MODEL_RUNNER")!="1"
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(url):
 with http.open(url,timeout=15)as x:return x.status,x.read()
health={}
for node,port in[("166",9081),("167",9900)]:
 code,raw=get("http://172.16.10."+node+":"+str(port)+"/health");assert code==200;(j/("native_health_"+node+".wire")).write_bytes(raw);health[node]=ref(j/("native_health_"+node+".wire"))
for node,id,source_run,name in[("D0","resp_glm_run230_D0_new","GLM-RUN-0230","newD0_create"),("D1","resp_glm_run211_D1_new","GLM-RUN-0211","newD1_create")]:
 base=json.loads((p/"runs"/source_run/"client_final_summary.json").read_text());row=next(x for x in base["requests"]if x["name"]==name);url="http://172.16.10."+("166:9081"if node=="D0"else"167:9900")+"/v1/responses/"+id
 status,raw=get(url);assert status==200and json.loads(raw)==json.loads(Path(row["wire"]["path"]).read_text());(j/(node+"_retained_native.wire")).write_bytes(raw)
status,raw=get("http://127.0.0.1:8000/control/replicas");placement=json.loads(raw);(j/"public_placement.json").write_bytes(raw);rows={x["id"]:x for x in placement["replicas"]}
assert rows["D0"]["group_faulted"]and not rows["D1"]["group_faulted"]and all(not x["active_requests"]and not x["draining"]for x in rows.values())
fault=p/"runs/GLM-RUN-0125/response_owners.json.fault";raw=fault.read_bytes();(j/"fault_journal_snapshot.json").write_bytes(raw);faultv=json.loads(raw);assert faultv["open"]and faultv["groups"]["local-228-166"]["faulted"]and not faultv["groups"]["local-211-167"]["faulted"]
policy=Path("/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp228/batch_queue_policy.json");assert json.loads(policy.read_text())==dict(schema_version=1,cohort_id="GLM-COHORT-0228",cap=2,serial=110,diagnostic=False,max_records=0)
limits=["Run238 source177 completed V1D1 auto-required-named-none4/41 nativeIDs-semantic-wire-usage-conservation-SDK0/native32/epochs/STORE230+211 VALID; D0wholegroupfault remains persisted/no clear or model/public/policyactions","Run237 originalFAILED executionfilename guard beforeclientstartup, no nativeclient artifact/SDK/inference andno headerRPC/nativecounterdelta; fixed238tools_client.py, noGPUreplay","Publiccurrentfault excludesD0, not implementedfeaturecompatroute; observedfunctional V1 proof only, V2required236 remainsnative500/grammar cause unresolved","Thinkingbudget/fullResponses139/state-lifecycle/fullAPI gapsopen; finitecap2mixed135.568and131.777 vs sameepochcap394.249 all3SLOFAIL/noKEEP/stablecapacity/globalbound/CurrentNone"]
out=dict(at=utc(),run_id=r.name,source_count=len(pins),measurement_valid=True,functional_acceptance=True,workflow_verdict="VALID",effective_output_tokens=41,new_native_completed=4,native_owner="D1",native_delta=delta,wireproof=wireproof,SDKinit_finalize0=True,physical=physical,native_health=health,native_STORE230and211_retained=True,public_D0_group_faulted=True,fault_journal=ref(j/"fault_journal_snapshot.json"),prior237_execution_guard_invalid=True,prior237_native_requests=0,GPU_replayed=False,models_signalled=0,policywrites=0,public_restarts=0,verdict="INCONCLUSIVE",Current=None,limits=limits)
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run238 V1D1 fourtools41nativeIDs-wire-usage-semantic-SDK0 VALID/native32/STORE230+211 retained/D0fault preserved;237 preclientguardINVALID0requests",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="source177-nativeV1D1-tool4-41IDs-grammar-semantic-wire-counters-SDK0-faultpreserved")],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(dict(valid=True,outputs=41,new_completed=4,owner="D1",native32_same=True)))
