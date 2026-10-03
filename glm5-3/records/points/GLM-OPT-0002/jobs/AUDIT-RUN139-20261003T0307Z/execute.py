from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0139";old=r.parent/"GLM-RUN-0131"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
owner=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text());assert owner["run_id"]==r.name and owner["status"]=="completed"
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"]
for pin in pins:
 assert Path(pin["path"]).read_bytes()==Path(pin["snapshot"]).read_bytes()and ref(Path(pin["path"]))["sha256"]==pin["sha256"]
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());live={}
for key,o in roots.items():
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=100);(j/(key+".stdout")).write_bytes(z.stdout);(j/(key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 live[key]=dict(root=v["root"],NPU16_same=True)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def values(b):
 return {k:sum(float(x)for x in re.findall(r"^vllm:"+k+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M))for k in ["num_requests_running","num_requests_waiting","generation_tokens_total","prompt_tokens_total","request_success_total","prefix_cache_queries_total","prefix_cache_hits_total","num_preemptions_total"]}
deltas={}
for node,port in[("166",9081),("167",9900)]:
 with http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=10)as response:raw=response.read()
 (j/(node+"_final.metrics")).write_bytes(raw);end=values(raw);assert end["num_requests_running"]==end["num_requests_waiting"]==0
 initial=values((r/("full_api_initial_"+node+".metrics")).read_bytes());deltas[node]={k:end[k]-initial[k]for k in end if k not in["num_requests_running","num_requests_waiting"]}
delta={k:sum(v[k]for v in deltas.values())for k in deltas["166"]}
summary=json.loads((r/"full_api_summary.json").read_text());responses=json.loads((r/"responses_summary.json").read_text());assert summary["functional_acceptance"]and responses["valid"]and responses["new_effective_output_tokens"]==384
assert responses["new_completed_inference_requests"]==11and summary["tool_cases"]==8
tools=[json.loads((r/("tools_"+name)/"summary.json").read_text())for name in["gateway_first","gateway_second"]]
assert all(v["valid"]and len(v["requests"])==4and v["cleanup"]["native_idle"]for v in tools)
effective=384+sum(v["effective_output_tokens"]for v in tools);assert effective==summary["effective_public_output_tokens"]
for tool in tools:
 for row in tool["requests"]:
  assert row["valid"]and ref(Path(row["wire"]["path"]))==row["wire"]and row["usage"]["completion_tokens"]>0
  assert row["name"]=="none"or len(row["tools"])==1and row["tools"][0]["name"]=="get_weather"and json.loads(row["tools"][0]["arguments"])=={"city":"Shanghai"}
attempts=json.loads((r/"attempts.json").read_text());assert len(attempts)==responses["http_attempts"]
for row in attempts:
 assert row["outcome"]=="native_http_valid"and ref(Path(row["wire"]["path"]))==row["wire"]and ref(Path(row["native_wire"]["path"]))==row["native_wire"]and row["native_wire"]["sha256"]==row["wire"]["sha256"]
 assert row["native_owner"]in["D0","D1"]
byid={x["id"]:x for x in attempts}
assert byid["native_invalid_parameter"]["http_status"]==400and byid["unknown_native404"]["http_status"]==404
for name in ["compat_json","compat_sse"]:
 raw=Path(byid[name]["wire"]["path"]).read_bytes()
 if name=="compat_json":value=json.loads(raw)
 else:
  data=[json.loads(b"\n".join(l[5:].lstrip(b" ")for l in frame.splitlines()if l.startswith(b"data:")))for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n")if b"data:"in frame]
  assert [x["sequence_number"]for x in data]==list(range(len(data)))and data[-1]["type"]=="response.completed"
  value=data[-1]["response"]
 assert value["usage"]["output_tokens"]==32and value["usage"]["input_tokens_details"]["cached_tokens"]==0and value["usage"]["input_tokens"]>11000
events=json.loads((r/"state_events.json").read_text());starts=[x for x in events if x["event"]=="gateway_started"];stops=[x for x in events if x["event"]=="gateway_clean_stopped"]
assert len(starts)==2and len(stops)==1and stops[0]["exit_code"]==0
background=next(x for x in events if x["event"]=="native_job_outlives_zero_HTTP_leases");assert background["native"]["num_requests_running"]>0and all(x["active_requests"]==0for x in background["placement"]["replicas"])
cancel=next(x for x in events if x["event"]=="background_cancelled_zero_credit");assert cancel["output_credit"]==0
for name in ["upgrade_retrieve_resp_glm_run138_base","upgrade_retrieve_resp_glm_run138_child","retrieve_after_restart","cancel_after_restart","base_after_restart","other_after_restart","base_after_readd"]:
 assert byid[name]["http_status"]==200and byid[name]["affinity_applied"]
assert (r/"drained_owner.wire").exists()
for name,expected in [("native_full_api",["task_acl_init","task_acl_finalize"]),("first.gateway",["task_acl_init","task_acl_finalize"]),("restart.gateway",["task_acl_init"])]:
 filename=(name+".stdout")if name=="native_full_api"else(name+".log")
 rows=[json.loads(l)for l in(r/filename).read_text().splitlines()if l.startswith('{"event":')]
 assert [x["event"]for x in rows]==expected and all(x["returncode"]==0for x in rows)
assert delta["generation_tokens_total"]>=effective and delta["request_success_total"]==19and delta["num_preemptions_total"]==0
cancel_cost=delta["generation_tokens_total"]-effective
config,_=checked_config(r/"service_config.json");container=json.loads((r/"public_service_owner.json").read_text())
top=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,stat,comm,args"],text=True)
matches=[int(l.split()[0])for l in top.splitlines()[1:]if "native_engines_service_entry.py --config "+str(r/"service_config.json")in l];assert len(matches)==1
pid=matches[0];stat=Path("/proc/"+str(pid)+"/stat").read_text();ident=dict(pid=pid,boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),start_ticks=stat[stat.rfind(")")+2:].split()[19])
assert same_process(ident)and {k:ident[k]for k in ["boot_id","start_ticks"]}=={k:container["identity"][k]for k in ["boot_id","start_ticks"]}
ids=[int(v)for v in next(l for l in Path("/proc/"+str(pid)+"/status").read_text().splitlines()if l.startswith("NSpid:")).split()[1:]];assert ids==[pid,container["pid"]]
assert [v.decode()for v in Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(bytes([0]))if v]==container["argv"]
ports=subprocess.check_output(["ss","-ltnp"],text=True);line=next(l for l in ports.splitlines()if re.search(r":8000\s",l));assert "pid="+str(pid)+","in line
with http.open("http://127.0.0.1:8000/healthcheck",timeout=10)as response:h=json.loads(response.read())
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as response:placement=json.loads(response.read())
assert h["status"]=="ok"and h["request_num"]==0and len(placement["replicas"])==2and all(x["active_requests"]==0and not x["group_faulted"]and not x["draining"]for x in placement["replicas"])
marker=next(json.loads(l.split("GLM_SERVICE_ENTRY_INSTALLED ",1)[1])for l in(r/"restart.gateway.log").read_text().splitlines()if"GLM_SERVICE_ENTRY_INSTALLED "in l)
assert marker["native_domains"]==config["native_domains"]==container["native_domains"]and marker["pid"]==container["pid"]
state_dir=r.parent/"GLM-RUN-0125";fault=json.loads((state_dir/"response_owners.json.fault").read_text());assert fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
owners=json.loads((state_dir/"response_owners.json").read_text())["owners"]
for key in ["resp_glm_run138_base","resp_glm_run138_child","resp_glm_run139_base","resp_glm_run139_other"]:
 row=owners[key];domain=next(x for x in config["native_domains"]if row["replica"]in x["members"]);assert row["group"]==domain["id"]and row["epoch"]==domain["epoch"]
assert owners["resp_glm_run139_base"]["group"]!=owners["resp_glm_run139_other"]["group"]
cross=next(x for x in events if x["event"]=="cross_owner_conflict_rejected_before_RPC");assert cross["status"]==503and ref(Path(cross["body"]["path"]))==cross["body"]and ref(Path(cross["wire"]["path"]))==cross["wire"]
assert byid["compat_chat"]["native_owner"]==byid["compat_json"]["native_owner"]==byid["compat_sse"]["native_owner"]=="D1"
assert json.loads((r/"retire_public.json").read_text())["SDKinit_finalize0"]
atomic_json(j/"public_service_proof.json",dict(at=utc(),host=ident,container=container,argv=container["argv"],marker=marker,listener=line,placement=placement,SDK_init0=True,retained=True,state_dir=str(state_dir)))
index=[ref(f)for f in sorted(r.iterdir())if f.is_file()]
atomic_json(j/"artifact_index.json",index)
out=dict(at=utc(),run_id=r.name,measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",effective_output_tokens=effective,completed_native_requests=19,helper_count=0,cancelled_output_credit=0,uncredited_cancel_native_generation_tokens=cancel_cost,SDK_client_init_finalize0=True,frontend_clean_restart_exit0_SDK0=True,old138_nativeIDs_retained=True,source_count=len(pins),source_unchanged=True,native_epochs=live,NPU32_same=True,API_count=2,geometry=config["engine_geometry"],native_domains=config["native_domains"],placement=config["placement"],native_cohort="GLM-COHORT-0137",fullAPI=summary,native_counter_delta=delta,native_deltas_per_node=deltas,public_host=ident,public8000_retained=True,public_SDK_init0_active=True,idle=True,artifact_index=ref(j/"artifact_index.json"),public_proof=ref(j/"public_service_proof.json"),limits=["FinitefullAPI/native state functional acceptance only; no formalSLO/stability/capacity/KEEP","Two independentlocalTP8PP2DCP8/native16NPUeach/current32active; no isolatedgain/qualitycomparison","Twoownerconflict503preRPC/shape routing/eachengine tools/stateaffinity/cleanproxyrestart verified; no STOREreplication","NativeSTORE not replicated; ownerjournal containsbindings only; actualconcurrentcustomID race andautomaticphysicalfaultmonitoring stillunknown"])
atomic_json(j/"reduction.json",out)
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="needs_decision",summary="Run139 valid fullnativeAPI/tool/state complete/19completed/cleanfrontendSDK0restart/sameNPU32/retainedpublic; noKEEP/capacity",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="complete diagnostic and retained native/public identity audit")],unknowns=out["limits"],decision_request="Proceed fullAPI and discriminating variableload research",next_check_at=None));print(json.dumps(dict(valid=True,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),effective=effective,cancel_cost=cancel_cost,delta=delta,public=ident)))
