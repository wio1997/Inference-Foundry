from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0138";old=r.parent/"GLM-RUN-0137"
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
 initial=values((r/("pilot_initial_"+node+".metrics")).read_bytes());deltas[node]={k:end[k]-initial[k]for k in end if k not in["num_requests_running","num_requests_waiting"]}
delta={k:sum(v[k]for v in deltas.values())for k in deltas["166"]}
reuse=json.loads((r/"reused137_audit.json").read_text());assert reuse["overall_verdict"]=="INVALID"and reuse["effective_output_tokens_reused_not_recredited"]==22and reuse["originalpilot_requests"]==0
semantic=reuse["semantic"];assert semantic["semantic_acceptance"]and semantic["semantic_pass_cases"]==semantic["semantic_cases"]==6
for row in semantic["requests"]:
 for k in["body","wire"]:assert ref(Path(row[k]["path"]))==row[k]
 v=json.loads(Path(row["wire"]["path"]).read_text());c=v["choices"][0];assert c["message"]["content"].strip()==row["expected"]and c["finish_reason"]=="stop"and len(c["token_ids"])==v["usage"]["completion_tokens"]and all(type(x)is int for x in c["token_ids"])
pilot=json.loads((r/"pilot_summary.json").read_text());assert pilot["measurement_valid"]and pilot["functional_acceptance"]and pilot["effective_public_output_tokens"]==192
assert [x["usage"]["prompt_tokens"]for x in pilot["requests"]]==[21,21,81932,81932]
for row in pilot["requests"]:
 raw=Path(row["wire"]["path"]).read_bytes();ids=[];prompt=[];usages=[]
 for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
  data=b"\n".join(v[5:].lstrip()for v in frame.splitlines()if v.startswith(b"data:"))
  if not data or data==b"[DONE]":continue
  v=json.loads(data);prompt+=v.get("prompt_token_ids",[])
  if v.get("usage"):usages.append(v["usage"])
  for c in v.get("choices",[]):ids+=c.get("token_ids",[])
 assert len(ids)==row["expected_output_tokens"]and all(type(v)is int for v in ids)and len(prompt)==row["usage"]["prompt_tokens"]and usages[-1]==row["usage"]
 item=row["wire"];assert ref(Path(item["path"]))==item
 assert row["contract"]["done"]and row["contract"]["finish_reasons"]=={"0":"length"}and not row["contract"]["native_error"]and row["completed"]
smoke=json.loads((r/"public_smoke_summary.json").read_text());assert smoke["functional_acceptance"]and smoke["effective_output_tokens"]==80
attempts=json.loads((r/"public_smoke_attempts.json").read_text());assert [x["status"]for x in attempts]==[200,200,200,200,503,503,200]
for row in attempts:assert ref(Path(row["wire"]["path"]))==row["wire"]
for name in ["native_pilot","native_public_smoke"]:
 rows=[json.loads(l)for l in(r/(name+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert [x["event"]for x in rows]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in rows)
effective=192+80
assert delta["generation_tokens_total"]==effective and delta["request_success_total"]==7 and delta["num_preemptions_total"]==0 and delta["prefix_cache_hits_total"]==0
config,_=checked_config(r/"service_config.json");proof=json.loads((r/"public_service_proof.json").read_text());ident=proof["host"]
assert same_process(ident)
assert [v.decode()for v in Path("/proc/"+str(ident["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
ports=subprocess.check_output(["ss","-ltnp"],text=True);line=next(l for l in ports.splitlines()if re.search(r":8000\s",l));assert "pid="+str(ident["pid"])+","in line
with http.open("http://127.0.0.1:8000/healthcheck",timeout=10)as response:h=json.loads(response.read())
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as response:placement=json.loads(response.read())
assert h["status"]=="ok"and h["request_num"]==0and len(placement["replicas"])==2and all(x["active_requests"]==0and not x["group_faulted"]and not x["draining"]for x in placement["replicas"])
trace=r.parent/"GLM-RUN-0125/router_trace.jsonl";events=[json.loads(l)for l in trace.read_text().splitlines()]
own=[e for e in events if e.get("event")=="lease_acquired"and e.get("created_at","")>""]
assert proof["container_marker"]["native_domains"]==config["native_domains"]
ack=[json.loads(l)for l in(r/"public.gateway.log").read_text().splitlines()if l.startswith('{"event":')];assert len(ack)==1and ack[0]["event"]=="task_acl_init"and ack[0]["returncode"]==0
for f in [r/"pilot_final.json",r/"public_smoke_final.json"]:assert json.loads(f.read_text())["NPU32_same"]and json.loads(f.read_text())["idle"]
cpu={}
for node in["166","167"]:
 rows=[json.loads(l)for l in(old/("CPU_config_"+node+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert rows[0]["event"]=="task_acl_init"and rows[-1]["event"]=="task_acl_finalize"and rows[0]["returncode"]==rows[-1]["returncode"]==0
 c=next(x for x in rows if x["event"]=="full_native_CLI_API_config_valid");assert c["TP"]==c["DCP"]==8and c["PP"]==2and c["world_size"]==c["local_world_size"]==16and c["nnodes"]==1and c["node_rank"]==0and c["NPU_workers_started"]==c["models"]==c["requests"]==0;cpu[node]=c
assert json.loads((old/"retire_public.json").read_text())["SDK_final0"]
for node in["166","167"]:
 rows=[json.loads(l)for l in(old/("cleanup_"+node+".stdout")).read_text().splitlines()if l.startswith("{")]
 assert rows[-1]["event"]=="cleanup_complete"and rows[-1]["npu_workers"]==rows[-1]["signals_to_unknown"]==rows[-1]["signals_to_inactive"]==0
selected={}
for node in["166","167"]:
 args=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/local_engines137_"+node+".log"]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode()
 rows=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in z.stdout.decode(errors="replace").splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in l];assert rows
 c=rows[-1];assert c["cohort"]=="GLM-COHORT-0137"and c["serial"]==c["prefill_cadence"]==1and c["budget_tokens"]==4096and c["prefill_threshold_tokens"]==1024and c["fallback"]is False;selected[node]=c
index=[ref(f)for f in sorted(r.iterdir())if f.is_file()]
atomic_json(j/"artifact_index.json",index)
out=dict(at=utc(),run_id=r.name,measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",effective_output_tokens=effective,completed_native_requests=7,helper_count=0,SDK_client_init_finalize0=True,semantic137_reused_not_recredited=semantic,partial137_reused_audit=ref(r/"reused137_audit.json"),source_count=len(pins),source_unchanged=True,native_epochs=live,NPU32_same=True,API_count=2,geometry=config["engine_geometry"],native_domains=config["native_domains"],placement=config["placement"],native_cohort="GLM-COHORT-0137",native_selected_policy=selected,CPU_full_native_both=cpu,pilot=pilot,public_smoke=smoke,native_counter_delta=delta,native_deltas_per_node=deltas,public_host=ident,public8000_retained=True,public_SDK_init0_active=True,idle=True,artifact_index=ref(j/"artifact_index.json"),limits=["Independentlocal16NPUeach functional diagnostic; no stableSLO/capacity/KEEP/globalbound orisolatedgain","DirectfullIDs/promptIDs/usage/length-DONE andsix137exactanswers reusednotrecredited; publictypedJSON/previous/retrieve80, fullSSE/tools/background/restart/newdomainconflict coverage next","Ownerjournal125 retains pointers; replaced131physicalSTORE data unreplicated/old125failclosed; no physicalfaultmonitoring/concurrentnativecustomID claim"])
atomic_json(j/"reduction.json",out)
atomic_json(j/"public_service_proof.json",dict(at=utc(),host=ident,argv=proof["argv"],container_marker=proof["container_marker"],listener=line,placement=placement,SDK_init0=True,retained=True,state_dir=str(r.parent/"GLM-RUN-0125"),native_domains=config["native_domains"]))
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="needs_decision",summary="Run138 independentnative engines functional diagnostic complete/SDK0/all32same/twodomains/public8000/272new/22semantic137reused; no capacity claim",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="newnativeboth fullCPU/ownedcleanup/fit/semantic/directcold/public/SDK/epoch/cost")],unknowns=out["limits"],decision_request="Proceed newdomain fullAPI and discriminating variableload",next_check_at=None));print(json.dumps(dict(valid=True,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),effective=effective,delta=delta,public=ident)))
