from pathlib import Path
import json,subprocess,sys,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0163";roots=json.loads((r/"restored/standalone_root_identities.json").read_text());members=json.loads((r/"restored/standalone_native_members.json").read_text());http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
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
  assert d["vllm:num_requests_running"]==d["vllm:num_requests_waiting"]==0;out[node]=d
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
 assert not v["measurement_valid"] and v["error_type"]=="FileNotFoundError" and "canonical_81932.body.json"in v["error"]
 assert len(v["requests"])==1
 row=v["requests"][0];assert row["completed"]and row["effective_public_output_credit"]==32 and row["usage"]==dict(prompt_tokens=21,total_tokens=53,completion_tokens=32)and row["contract"]["done"]and row["contract"]["finish_reasons"]=={"0":"length"}
 b=Path(row["wire"]["path"]).read_bytes();assert ref(row["wire"]["path"])==row["wire"]
 ids=[];prompt=None;native_ids=set()
 for l in b.decode().splitlines():
  if not l.startswith("data: ")or l=="data: [DONE]":continue
  x=json.loads(l[6:]);native_ids.add(x["id"])
  if "prompt_token_ids"in x:prompt=x["prompt_token_ids"]
  for c in x.get("choices",[]):ids.extend(c.get("token_ids")or[])
 assert len(native_ids)==1and len(ids)==32and len(prompt)==21
 a=[json.loads(l)for l in(r/("D1_pilot_"+str(i)+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert [x["event"]for x in a]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in a)
 rows.append(dict(request=row,SDKinit_finalize0=True,prompt_token_ids=21,output_token_ids=32,native_id=next(iter(native_ids))))
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
assert all(v==0for v in delta["166"].values())
d=delta["167"];assert d["vllm:generation_tokens_total"]==64 and d["vllm:prompt_tokens_total"]==d["vllm:prefix_cache_queries_total"]==42 and d["vllm:request_success_total"]==2 and d["vllm:prefix_cache_hits_total"]==d["vllm:num_preemptions_total"]==d["vllm:num_requests_running"]==d["vllm:num_requests_waiting"]==0
h=json.loads(http.open("http://127.0.0.1:8000/healthcheck",timeout=10).read());snap=json.loads(http.open("http://127.0.0.1:8000/control/replicas",timeout=10).read());assert h["request_num"]==0
atomic_json(j/"public_state.json",dict(health=h,replicas=snap))
assert any(x["group_faulted"]for x in snap["replicas"])
out=dict(at=utc(),run_id=r.name,execution_verdict="INVALID",readonly_terminal_acceptance=True,failure="GPT fixture construction omitted canonical_81932.body.json; after two successful short32, before cold requests/public switch",source_pins=len(spec["stages"][0]["sources"]),native32_same_post_restore=True,native_counter_delta=delta,native_metrics=actual,new_native_inference_outputs=64,new_native_requests=2,clients=rows,SDKinit_finalize0=True,PP38_40_native_ready=True,capacity_or_complete_functional_credit=False,old_public_D0_survivor=True,oldD1_epoch_quarantined=True,original_source_raw_preserved=True)
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run163 originalINVALID missing longinput fixture; actualnativePP38,40 startup/Graphready and two short32 complete; SDK0/source103/native32ownedidle/D0unchanged/oldpublicD0survivor; no completefunctional orcapacitycredit.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="Failurepartialshort64/rawpreserved")],unknowns=["Cold81932/API-STOREpublic/newPPperformance pending"],decision_request=None,next_check_at=None));print(json.dumps(out))
