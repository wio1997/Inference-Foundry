from pathlib import Path
import json,sys,time,hashlib,re,subprocess,shlex,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0070";deadline=time.monotonic()+2700
while True:
 s=json.loads((r/"state.json").read_text())
 if s["status"]!="running":break
 assert same_process(s["owner"])and time.monotonic()<deadline;time.sleep(10)
assert s["status"]=="completed"and not same_process(s["owner"])
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"]
for v in pins:assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
matrix=json.loads((r/"matrix_summary.json").read_text());assert matrix["functional_acceptance"]and matrix["new_requests"]==15 and matrix["effective_public_output_tokens"]==2304
def metrics(f):
 a={}
 for l in f.read_text().splitlines():
  if not l or l.startswith("#"):continue
  name=l.split("{")[0].split()[0]
  if any(v in name for v in["prefix_cache_hits_total","prefix_cache_queries_total","prompt_tokens_total","generation_tokens_total","request_success_total","num_preemptions_total","num_requests_running","num_requests_waiting","kv_cache_usage_perc"]):a[name]=a.get(name,0)+float(l.rsplit(" ",1)[1])
 return a
rows=[]
for window in matrix["windows"]:
 w=Path(window["window"]);a=json.loads((w/"mixed_summary.json").read_text());plan=json.loads((w/"arrival_plan.json").read_text());by={v["id"]:v for v in plan["cases"]};rr=[]
 for v in a["requests"]:
  c=by[v["id"]];raw=Path(v["wire"]["path"]).read_bytes();assert len(raw)==v["wire"]["bytes"]and hashlib.sha256(raw).hexdigest()==v["wire"]["sha256"]
  obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True)
  for n in range(0,len(raw),733):obs.feed(raw[n:n+733])
  con=obs.contract();assert con["done"]and not con["unknown"]and not con["native_error"]and con["finish_reasons"]=={"0":"length"}and con["usage"]==v["usage"]
  frames=[]
  for part in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
   data=b"\n".join(l[5:].removeprefix(b" ")for l in part.splitlines()if l.startswith(b"data:"))
   if data and data!=b"[DONE]":frames.append(json.loads(data))
  ids=[t for f in frames for ch in f.get("choices",[])for t in ch.get("token_ids")or[]];assert ids==v["committed_token_ids"]and len(ids)==c["expected_outputs"]
  body=json.loads((w/c["body"]).read_text());assert body["cache_salt"]==w.name+"-"+c["id"]and not body.get("kv_transfer_params")
  points=v["native_token_chunk_points"];assert points[-1]["committed_token_count"]==len(ids)and all(points[n]["elapsed_s"]>=points[n-1]["elapsed_s"]and points[n]["committed_token_count"]>points[n-1]["committed_token_count"]for n in range(1,len(points)))
  rr.append({k:v[k]for k in["id","owner","actual_dispatch_s","ttft_s","wall_s","usage","max_native_chunk_gap_s","wire"]})
 counters={}
 for key in["D0","D1"]:
  b=metrics(w/("before_"+key+".metrics"));e=metrics(w/("after_"+key+".metrics"));delta={k:e.get(k,0)-v for k,v in b.items()if k.endswith("_total")}
  expected=sum(v["usage"]["completion_tokens"]for v in rr if v["owner"]==key);prompts=sum(v["usage"]["prompt_tokens"]for v in rr if v["owner"]==key)
  assert delta.get("vllm:generation_tokens_total",0)==expected and delta.get("vllm:prompt_tokens_total",0)==prompts and delta.get("vllm:prefix_cache_queries_total",0)>=prompts
  assert delta.get("vllm:prefix_cache_hits_total",0)==delta.get("vllm:external_prefix_cache_hits_total",0)==delta.get("vllm:num_preemptions_total",0)==0
  assert all(e.get(k,0)==0 for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"])
  counters[key]=dict(delta=delta,idle=True,cold_confirmed=True)
 rows.append(dict(policy=window["policy"],elapsed_s=a["elapsed_s"],requests=rr,native_counters=counters,actual_outputs=768))
owners=json.loads((r/"adopted_model_identities.json").read_text());members=json.loads((r/"native_member_identities.json").read_text());physical={};native={}
for key,o in owners.items():
 code="""import pathlib,json,sys,subprocess,re
a=json.load(sys.stdin);o=a['owner'];boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for x in a['targets']+[dict(pid=o['pid'],identity=o['identity'])]:
 p=pathlib.Path('/proc/'+str(x['pid']));b=(p/'stat').read_text();v=b[b.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity']
assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);ids={int(v)for v in re.findall(r'^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker',raw,re.M)};assert ids=={x['pid']for x in a['targets']}
policy=json.loads(pathlib.Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run69/issue_budget_policy.json').read_text());assert policy==a['policy'];print(json.dumps(dict(API_same=True,NPU16_same=True,npu_pids=sorted(ids),policy=policy)))
"""
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 targets=[v for v in members[key]["owned_targets"]if v["pid"]in members[key]["npu_worker_pids"]]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,targets=targets,policy=matrix["terminal_policy"])).encode(),capture_output=True,timeout=90);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();physical[key]=json.loads(z.stdout)
 logfile=next(v["log"]for v in json.loads((r.parent/"GLM-RUN-0069/planned_launch.json").read_text())if v["node"]==o["host"]);args=["cat",logfile]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=90);z.check_returncode();f=j/(key+".native.log");f.write_bytes(z.stdout);ls=z.stdout.decode(errors="replace").splitlines()
 installed=[json.loads(l.split("GLM_ISSUE_BUDGET_INSTALLED ",1)[1])for l in ls if"GLM_ISSUE_BUDGET_INSTALLED "in l];selected=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in ls if"GLM_ISSUE_BUDGET_SELECTED "in l];assert len(installed)==1 and installed[0]["native_base"]=="AsyncScheduler"and installed[0]["allocated_scheduler_max"]==16384
 assert [(v["budget_tokens"],v["prefill_threshold_tokens"],v["prefill_cadence"],v["serial"])for v in selected]==[(4096,2048,1,1),(4096,0,1,2),(4096,2048,1,3),(4096,2048,2,4)]and all(not v["fallback"]for v in selected)
 assert not any(any(x in l for x in["ERROR","Traceback","OutOfMemoryError"])for l in ls)
 native[key]=dict(raw=dict(path=str(f),bytes=len(z.stdout),sha256=hashlib.sha256(z.stdout).hexdigest()),installed=installed,selected=selected,native_model_errors=0)
out=dict(at=utc(),verdict="INCONCLUSIVE",measurement_valid=True,functional_acceptance=True,source_pins=len(pins),new_requests=15,effective_public_output_tokens=2304,windows=rows,physical=physical,native=native,models_started=0,model_signals=0,GPUallocation16384_unchanged=True,terminal_policy=matrix["terminal_policy"],limits=matrix["limits"],conclusion="SameD69cohort nativeCPU budget4096/threshold0c1,2048c1,2048c2 cold mixed3window tradeoff withboundedhot policy/unchangedAPI+32physicalepochs; actualnumbers inwindows, no formalTPOT/SLO/capacity/KEEP.")
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,results=out,reduction=str(j/"reduction.json"));atomic_json(r/"manifest.json",m);(r/"summary.md").write_text("# "+r.name+"\n\n"+out["conclusion"]+"\n\n"+json.dumps(dict(outputs=2304,windows=[dict(budget=w["policy"]["budget_tokens"],threshold=w["policy"]["prefill_threshold_tokens"],cadence=w["policy"]["prefill_cadence"],TTFT=[v["ttft_s"]for v in w["requests"]],max_chunks=[v["max_native_chunk_gap_s"]for v in w["requests"]])for w in rows]))+"\n")
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run70 15nativecoldmixedrequests/2304commits/3boundedhot threshold/cadence settings/sameD69API2NPU32epochs/GPU16k unchanged; noKEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="originalwires/IDs/coldcounterdeltas/epochs/nativepolicy selectedlogs")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(out))

