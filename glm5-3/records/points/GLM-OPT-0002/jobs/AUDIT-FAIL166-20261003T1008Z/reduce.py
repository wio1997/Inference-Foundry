from pathlib import Path
import json,subprocess,sys,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0166";roots=json.loads((r/"restored/standalone_root_identities.json").read_text());members=json.loads((r/"restored/standalone_native_members.json").read_text());http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])
def ref(f):
 f=Path(f);b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def probe(label):
 out={}
 for key,o in roots.items():
  args=["python3","-c",(r/"live_probe.py").read_text()]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=100);(j/(label+"_"+key+".stdout")).write_bytes(z.stdout);(j/(label+"_"+key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
  expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 for node,port in[("166",9081),("167",9900)]:
  b=http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=15).read();(j/(label+"_"+node+".metrics")).write_bytes(b);d={}
  for l in b.decode().splitlines():
   if not l or l.startswith("#"):continue
   k=l.split("{")[0].split()[0]
   if k in ["vllm:request_success_total","vllm:generation_tokens_total","vllm:prompt_tokens_total","vllm:prefix_cache_queries_total","vllm:prefix_cache_hits_total","vllm:num_preemptions_total","vllm:num_requests_running","vllm:num_requests_waiting"]:d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
  out[node]=d
 return out


assert s["completed_stages"]==["prepare","fault","restore"] and s["failure_phase"]=="pilot"
for n in ["prepare","fault","restore"]:assert json.loads((r/(n+".phase.json")).read_text())["exit_code"]==0
assert json.loads((r/"pilot.phase.json").read_text())["exit_code"]==1
assert not (r/"switch.phase.json").exists() and not (r/"finalize.phase.json").exists()
spec=json.loads((r/"controller_spec.json").read_text())
for row in spec["stages"][0]["sources"]:assert Path(row["path"]).read_bytes()==Path(row["snapshot"]).read_bytes()and hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()==row["sha256"]

rows=[]
for i in [0,1]:
 v=json.loads((r/("pilot_"+str(i))/"pilot_summary.json").read_text())
 assert not v["measurement_valid"] and v["error_type"]=="TimeoutError" and len(v["requests"])==2
 short,full=v["requests"]
 assert short["completed"] and short["effective_public_output_credit"]==32 and short["usage"]==dict(prompt_tokens=21,total_tokens=53,completion_tokens=32) and short["contract"]["done"] and short["contract"]["finish_reasons"]=={"0":"length"}
 assert not full["completed"] and full["effective_public_output_credit"]==0 and full["error_type"]=="TimeoutError"
 details=[]
 for row in [short,full]:
  b=Path(row["wire"]["path"]).read_bytes();assert ref(row["wire"]["path"])==row["wire"]
  ids=[];prompt=None;native_ids=set();done=False;usage=None;parse_errors=[]
  for l in b.decode().splitlines():
   if not l.startswith("data: "):continue
   if l=="data: [DONE]":done=True;continue
   try:x=json.loads(l[6:])
   except ValueError as e:parse_errors.append(str(e));continue
   if x.get("id"):native_ids.add(x["id"])
   if "prompt_token_ids" in x:prompt=x["prompt_token_ids"]
   if x.get("usage"):usage=x["usage"]
   for c in x.get("choices",[]):ids.extend(c.get("token_ids") or [])
  if row is short:assert len(native_ids)==1 and len(ids)==32 and len(prompt)==21 and done and not parse_errors
  details.append(dict(request=row,prompt_token_ids=None if prompt is None else len(prompt),output_token_ids=len(ids),native_ids=sorted(native_ids),DONE=done,usage=usage,parse_errors=parse_errors))
 a=[json.loads(l) for l in(r/("D1_pilot_"+str(i)+".stdout")).read_text().splitlines() if l.startswith('{"event":')]
 assert [x["event"] for x in a]==["task_acl_init","task_acl_finalize"] and all(x["returncode"]==0 for x in a)
 rows.append(dict(client=i,SDKinit_finalize0=True,requests=details))
actual=probe("terminal")
def count(f):
 d={}
 for l in Path(f).read_text().splitlines():
  if l and not l.startswith("#"):
   k=l.split("{")[0].split()[0]
   if k in actual["166"]:d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
 return d
before={n:count(r/("pilot_before_"+n+".metrics"))for n in["166","167"]}
delta={n:{k:actual[n][k]-before[n][k]for k in actual[n]}for n in actual}

assert all(v==0 for v in delta["166"].values())
d=delta["167"]
assert d["vllm:request_success_total"]==2 and d["vllm:prefix_cache_hits_total"]==d["vllm:num_preemptions_total"]==0
# Failed engine may still retain two Running requests; snapshots do not imply health or idle.
args=["ssh","-o","BatchMode=yes","root@172.16.10.167","cat /data/tiankuan/wio/glm52-pd/deploy/logs/local_pp166_167.log"]
z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode();(j/"native_terminal.log").write_bytes(z.stdout)
lines=z.stdout.decode(errors="replace").splitlines()
excerpt=[l for l in lines if any(t in l for t in ["sync_and_slice_intermediate_tensors","has incorrect dimension","size of tensor","WorkerProc hit","shm_broadcast.py","model_runner_v1.py:"])]
assert any("Shape [5, 6144]" in l and "[16, 6144]" in l for l in excerpt)
atomic_json(j/"native_error_excerpt.json",dict(source=ref(j/"native_terminal.log"),selected_lines=excerpt,limits=["Native PP intermediate-copy shape error is observed; deeper cause requires source and shape-transport investigation."]))
# Native source capture is a later read-only observation, not a retroactive source pin.
src_script='import pathlib,json,hashlib,ast; paths=["/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py","/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py","/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py","/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py"]; out=[]\nfor name in paths:\n p=pathlib.Path(name);b=p.read_bytes();txt=b.decode();tree=ast.parse(txt);sn=[]\n for n in ast.walk(tree):\n  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ["sync_and_slice_intermediate_tensors","sync_and_gather_intermediate_tensors","execute_model","_preprocess"]:\n   sn.append(dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source="\\n".join(txt.splitlines()[n.lineno-1:n.end_lineno])))\n out.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),functions=sn))\nprint(json.dumps(out))'
z=subprocess.run(["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["docker","exec","-i","glm52-single","python3","-"])],input=src_script.encode(),capture_output=True,timeout=90)
(j/"source_observation.stderr").write_bytes(z.stderr);z.check_returncode();(j/"native_source_observation.json").write_bytes(z.stdout)
h=json.loads(http.open("http://127.0.0.1:8000/healthcheck",timeout=10).read());snap=json.loads(http.open("http://127.0.0.1:8000/control/replicas",timeout=10).read());assert h["request_num"]==0
atomic_json(j/"public_state.json",dict(health=h,replicas=snap))
assert any(x["group_faulted"]for x in snap["replicas"])

out=dict(at=utc(),run_id=r.name,execution_verdict="INVALID",readonly_terminal_acceptance=True,failure="Both NEWcold81932 requests timeout after native PP1 intermediate copy shape mismatch [5,6144] versus [16,6144]; two short32 complete",source_pins=len(spec["stages"][0]["sources"]),native32_same_post_restore=True,native_counter_delta=delta,native_metrics=actual,effective_completed_outputs=64,effective_completed_requests=2,clients=rows,SDKinit_finalize0=True,PP4_native_startup_Graph_API_ready=True,capacity_or_complete_functional_credit=False,old_public_D0_survivor=True,oldD1_epoch_quarantined=True,original_source_raw_preserved=True,native_error_excerpt=ref(j/"native_error_excerpt.json"),native_source_observation=ref(j/"native_source_observation.json"),limits=["Running2 and native partial generation are not valid completed outputs; no new public switch, no KEEP or performance capacity claim; process presence does not prove failed engine health"])
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run166 INVALID: PP4 startup and two short32 complete, both cold81932 timeout after PP1 native intermediate-copy shape mismatch. SDK0/source106/native32 roots retained/D0 unchanged/oldpublic D0 survivor; no full functional or capacity credit.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="Failure actual native shape mismatch and partial64, preserved source/raw")],unknowns=["Exact shape mismatch cause and safe control-shape recovery", "Failed D1 requires owned recovery before new inference"],decision_request=None,next_check_at=None));print(json.dumps(out))
