from pathlib import Path
import json,hashlib,sys,subprocess,shlex,urllib.request
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0236"
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
state=json.loads((r/"state.json").read_text());assert state["status"]=="failed"and state["failure_phase"]=="function"and state["completed_stages"]==["prepare"]and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text());m=json.loads((r/"manifest.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==183and ref(r/"controller_spec.json")["sha256"]==state["spec_sha256"]==m["spec"]["sha256"]
for x in pins:assert ref(x["path"])["sha256"]==x["sha256"]and Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()
tools=json.loads((r/"tools_gateway_first/summary.json").read_text());assert not tools["valid"]and tools["status"]=="failed"and len(tools["requests"])==2
auto,bad=tools["requests"];assert auto["valid"]and auto["name"]=="auto"and auto["status"]==200and auto["usage"]==dict(prompt_tokens=160,completion_tokens=12,total_tokens=172)and len(auto["token_ids"])==12and auto["finish_reasons"]=={"0":"tool_calls"}and auto["tools"]==[dict(name="get_weather",arguments='{"city": "Shanghai"}')]
assert bad["name"]=="required"and bad["status"]==500and not bad["valid"]and bad["native_error"]
trace=[json.loads(l)for l in(p/"runs/GLM-RUN-0125/router_trace.jsonl").read_text().splitlines()]
wireproof=[]
for row in[auto,bad]:
 assert ref(row["wire"]["path"])==row["wire"]
 header=r.name+"-tool-gateway_first-"+row["name"];leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==header];assert len(leases)==1and leases[0]["replica"]=="D0"
 body=r/("tools_gateway_first/"+row["name"]+".body.json");assert ref(body)["sha256"]==leases[0]["body_sha256"];assert json.loads(body.read_text())["tool_choice"]==row["name"]
 wires=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==leases[0]["lease_id"]];releases=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==leases[0]["lease_id"]];assert len(wires)==len(releases)==1and releases[0]["released"]and releases[0]["backend_failure"]==(row["name"]=="required")
 assert Path(wires[0]["wire_path"]).read_bytes()==Path(row["wire"]["path"]).read_bytes()
 wireproof.append(dict(name=row["name"],native_owner="D0",body=ref(body),wire=row["wire"],backend_failure=releases[0]["backend_failure"]))
delta={}
for node in["D0","D1"]:
 a=metrics(r/("initial_0_"+node+".metrics"));b=metrics(r/("tools_gateway_first/final_"+node+".metrics"));delta[node]={k:b[k]-a[k]for k in a if k.endswith("_total")}
 assert b["vllm:num_requests_running"]==b["vllm:num_requests_waiting"]==b["vllm:kv_cache_usage_perc"]==0
assert delta["D0"]["vllm:generation_tokens_total"]==16and delta["D0"]["vllm:prompt_tokens_total"]==320and delta["D0"]["vllm:request_success_total"]==2and all(x==0for x in delta["D1"].values())
ev=[json.loads(l)for l in(r/"native_full_api.stdout").read_text().splitlines()if l.startswith('{"event":')];assert ev[0]["event"]=="task_acl_init"and ev[-1]["event"]=="task_acl_finalize"and ev[0]["returncode"]==ev[-1]["returncode"]==0
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());physical={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout);ids={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 physical[key]=dict(root_same=True,NPU16_same=True,proof=ref(j/(key+".owner.stdout")))
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
log=Path("/data/tiankuan/wio/glm52-pd/deploy/logs/local_pp228_166.log").read_bytes();cursor=json.loads((r/"queue_log_cursor.json").read_text());raw=log[cursor["bytes"]:];(j/"native_failure_window.log").write_bytes(raw)
assert b"Failed to advance FSM"in raw and b"grammar rejected tokens [455, 68852, 709]"in raw and b"Terminating request."in raw and b"500 Internal Server Error"in raw and b"GLM_BATCH_QUEUE_CPU_STEP "not in raw
policy=Path("/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp228/batch_queue_policy.json");assert json.loads(policy.read_text())==dict(schema_version=1,cohort_id="GLM-COHORT-0228",cap=2,serial=110,diagnostic=False,max_records=0)
limits=["OriginalRun236 workflowFAILED/measurementINVALID atrequiredtools500; autoactual12nativeIDs valid; requesterror4nativegenerated tokens uncredited, nohelper/performancecredit","NativegrammarFSM reject actual[455,68852,709]/requestterminated, notGPUworkerdeath; roots/NPU32/nativehealth200/nativeSTORE230+211 sameepoch verifiedreadonly","Public V12 failclosed treatsnative500 asgroupfault persistedD0 despitelivehealthyAPI; bothpeersstillpresent, cleanupPOSTduplicate409 was harmless but fixturemustcheck membership","StructuredV2required/MTP-async masking causeunresolved, do notalterkernel/math/guard or blindlyclearfault/replay request; investigatecontrolcompatibility usingresidentV1D1","NativeV2thinkingbudget/fullnative139 gapsopen; cap2mixed repeatsconditional135.568and131.777 vscap394.249 butnoKEEP/stablecapacity/globalbound/CurrentNone"]
out=dict(at=utc(),run_id=r.name,source_count=len(pins),workflow_verdict="INVALID",measurement_valid=False,functional_acceptance=False,auto_tool_function_valid=True,actual_public_effective_outputs=12,actual_new_completed=1,native_generation_delta=16,uncredited_native_failed_request_outputs=4,native_metric_request_success_delta=2,metric_success_includes_error_terminal=True,failed_tool_required_status=500,native_FSM_rejected_tokens=[455,68852,709],SDKinit_finalize0=True,physical=physical,native_health=health,native_STORE230and211_retained=True,native_delta=delta,wireproof=wireproof,public_D0_group_faulted=True,fault_journal=ref(j/"fault_journal_snapshot.json"),native_failure=ref(j/"native_failure_window.log"),GPU_replayed=False,models_signalled=0,policywrites=0,public_restarts=0,verdict="INVALID",Current=None,limits=limits)
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run236 FAILEDrequired500 nativeFSMreject; auto12IDsvalid/native16delta4uncredited; healthy32/STORE230+211 retained butpublicD0groupfault persisted; readonly/noGPUreplay",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="source183-nativeFSM-request500-auto12IDs-health-NPU32-STORE-faultscope")],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(dict(valid=True,original_workflow="INVALID",auto_outputs=12,failed_native_uncredited=4,NPU32_same=True)))
