from pathlib import Path
import json,sys,hashlib,subprocess,shlex,re,urllib.request,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0067";deadline=time.monotonic()+4200
while True:
 s=json.loads((r/"state.json").read_text())
 if s["status"]!="running":break
 assert same_process(s["owner"]),"controller disappeared without terminal state"
 assert time.monotonic()<deadline,"audit wait deadline"
 time.sleep(15)
if s["status"]!="completed":
 evidence=[];errors={}
 for x in json.loads((r/"planned_launch.json").read_text()):
  args=["tail","-n","100",x["log"]]
  if x["node"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,capture_output=True,timeout=45);f=j/(x["node"]+".failure.tail");f.write_bytes(z.stdout);errors[x["node"]]=z.stdout.decode(errors="replace")[-8000:];evidence.append(dict(id=x["node"]+"_failure_tail",path=str(f),bytes=len(z.stdout),sha256=hashlib.sha256(z.stdout).hexdigest(),locator="native terminal error context"))
 atomic_json(j/"failure_reduction.json",dict(at=utc(),state=s,errors=errors,limits="Nativefailedstage; no functional/performance/capacitycredit, originalphase preserved"))
 atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="needs_decision",summary="Run67 failedstage preserved; nativefailure context only, no performance credit",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=evidence,unknowns=["Native GPU/E2E failure requires rootcause review"],decision_request="GPT review native first failure and design newRun, no replay",next_check_at=None));print(json.dumps(dict(state=s,errors=errors)));raise SystemExit(0)
assert not same_process(s["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for x in spec["stages"][0]["sources"]:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]
a=json.loads((r/"mixed_summary.json").read_text());assert a["functional_acceptance"]and a["actual_new_requests"]==5 and a["effective_public_output_tokens"]==768
plan=json.loads((r/"arrival_plan.json").read_text());by={v["id"]:v for v in plan["cases"]};rows=[]
for row in a["requests"]:
 c=by[row["id"]];b=Path(row["wire"]["path"]).read_bytes();assert len(b)==row["wire"]["bytes"]and hashlib.sha256(b).hexdigest()==row["wire"]["sha256"]
 obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True)
 for n in range(0,len(b),733):obs.feed(b[n:n+733])
 con=obs.contract();assert con["done"]and not con["unknown"]and not con["native_error"]and con["finish_reasons"]=={"0":"length"}and con["usage"]==row["usage"]
 frames=[]
 for part in b.replace(b"\r\n",b"\n").split(b"\n\n"):
  d=b"\n".join(l[5:].removeprefix(b" ")for l in part.splitlines()if l.startswith(b"data:"))
  if d and d!=b"[DONE]":frames.append(json.loads(d))
 ids=[v for f in frames for ch in f.get("choices",[])for v in ch.get("token_ids")or[]];assert ids==row["committed_token_ids"]and len(ids)==c["expected_outputs"]
 points=row["native_token_chunk_points"];assert points[-1]["committed_token_count"]==len(ids)and all(points[n]["elapsed_s"]>=points[n-1]["elapsed_s"]and points[n]["committed_token_count"]>points[n-1]["committed_token_count"]for n in range(1,len(points)))
 body=json.loads((r/c["body"]).read_text());assert body["cache_salt"]==r.name+"-"+c["id"]and not body.get("kv_transfer_params")
 rows.append({k:row[k]for k in["id","owner","actual_dispatch_s","ttft_s","wall_s","usage","effective_public_output_credit","max_native_chunk_gap_s","wire"]})
def metrics(f):
 out={}
 for l in f.read_text().splitlines():
  if not l or l.startswith("#"):continue
  name=l.split("{")[0].split()[0]
  if any(v in name for v in["prefix_cache_hits_total","prefix_cache_queries_total","prompt_tokens_total","generation_tokens_total","request_success_total","num_preemptions_total","num_requests_running","num_requests_waiting","kv_cache_usage_perc"]):out[name]=out.get(name,0)+float(l.rsplit(" ",1)[1])
 return out
deltas={}
for key in["D0","D1"]:
 b=metrics(r/("before_"+key+".metrics"));e=metrics(r/("after_"+key+".metrics"));d={k:e.get(k,0)-v for k,v in b.items()if k.endswith("_total")};deltas[key]=dict(delta=d,after_gauges={k:v for k,v in e.items()if not k.endswith("_total")})
 expected=sum(v["usage"]["completion_tokens"]for v in rows if v["owner"]==key);assert d.get("vllm:generation_tokens_total",0)==expected and all(e.get(k,0)==0 for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"])
owners=json.loads((r/"adopted_model_identities.json").read_text());old=json.loads((r/"native_member_identities.json").read_text());physical={}
for key,o in owners.items():
 code="""import json,sys,pathlib,subprocess,re
a=json.load(sys.stdin);o=a['owner'];boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for x in a['targets']:
 b=pathlib.Path('/proc/'+str(x['pid'])+'/stat').read_text();f=b[b.rfind(')')+2:].split();assert f[0]not in['Z','X']and dict(boot_id=boot,start_ticks=f[19])==x['identity']
b=pathlib.Path('/proc/'+str(o['pid'])+'/stat').read_text();f=b[b.rfind(')')+2:].split();assert f[0]not in['Z','X']and dict(boot_id=boot,start_ticks=f[19])==o['identity']
assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);ids={int(v)for v in re.findall(r'^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker',raw,re.M)};assert ids=={x['pid']for x in a['targets']};print(json.dumps(dict(API_same=True,NPU16_same=True,npu_pids=sorted(ids))))
"""
 targets=[v for v in old[key]["owned_targets"]if v["pid"]in old[key]["npu_worker_pids"]];args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,targets=targets)).encode(),capture_output=True,timeout=90);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();physical[key]=json.loads(z.stdout)

native={};cleanup={};prior=json.loads((r.parent/"GLM-RUN-0065/native_member_identities.json").read_text())
for x in json.loads((r/"planned_launch.json").read_text()):
 node=x["node"];key="D"+str(x["rank"]);args=["cat",x["log"]]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=90);f=j/(key+".native.log");f.write_bytes(z.stdout);z.check_returncode();ls=z.stdout.decode(errors="replace").splitlines()
 installed=[json.loads(l.split("GLM_ISSUE_BUDGET_INSTALLED ",1)[1])for l in ls if"GLM_ISSUE_BUDGET_INSTALLED "in l]
 selected=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in ls if"GLM_ISSUE_BUDGET_SELECTED "in l]
 assert len(installed)==1 and installed[0]["native_base"]=="AsyncScheduler"and installed[0]["allocated_scheduler_max"]==16384 and installed[0]["default_budget"]==4096
 assert selected and all(v["budget_tokens"]==4096 and v["serial"]==1 and not v["fallback"]for v in selected)
 bad=[l for l in ls if any(n in l for n in["ERROR","Traceback","OutOfMemoryError"])];assert not bad,bad[:3]
 skip=[json.loads(l.split("GLM_UNUSED_SFA_WORKSPACE_SKIPPED ",1)[1])for l in ls if"GLM_UNUSED_SFA_WORKSPACE_SKIPPED "in l]
 assert len(skip)==32 and all(v["actual_elements"]==0 and v["native_math_changes"]==0 for v in skip)
 assert len([l for l in ls if"GLM_DP_METADATA_INSTALLED "in l])==16 and any("Graph capturing finished"in l for l in ls)
 assert old[key]["API_owner"]!=prior[key]["API_owner"]
 original={v["pid"]:v["identity"]for v in prior[key]["owned_targets"]}
 assert all(original.get(v["pid"])!=v["identity"]for v in old[key]["owned_targets"]if v["pid"]in old[key]["npu_worker_pids"])
 native[key]=dict(raw=dict(path=str(f),bytes=len(z.stdout),sha256=hashlib.sha256(z.stdout).hexdigest()),issue_budget_installed=installed,issue_budget_selected=selected,zero_unused_SFA=len(skip),metadata=16,graph_complete=True,errors=bad)
 journal=r/("cleanup_"+node+".stdout");events=[json.loads(l)for l in journal.read_text().splitlines()if l.startswith("{")]
 assert events[-1]["all_original_D_inactive"]and events[-1]["signals_to_unknown"]==events[-1]["signals_to_inactive"]==0
 for e in events:
  if e.get("event")=="signal_attempt":assert e["pid"]in original and e["identity"]==original[e["pid"]]
 cleanup[node]=dict(raw=dict(path=str(journal),bytes=journal.stat().st_size,sha256=hashlib.sha256(journal.read_bytes()).hexdigest()),sent=[v for v in events if v.get("event")=="signal_sent"],original_D65_inactive=True)
baseline=json.loads((r.parent/"GLM-RUN-0066/reduction_brief.json").read_text());bb={v["id"]:v for v in baseline["requests"]}
comparison=[dict(id=v["id"],TTFT_67=v["ttft_s"],TTFT_66=bb[v["id"]]["ttft_s"],wall_67=v["wall_s"],wall_66=bb[v["id"]]["wall_s"],max_chunk_gap_67=v["max_native_chunk_gap_s"],max_chunk_gap_66=bb[v["id"]]["max_native_chunk_gap_s"])for v in rows]

out=dict(at=utc(),verdict="INCONCLUSIVE",measurement_valid=True,functional_acceptance=True,source_pins=len(spec["stages"][0]["sources"]),actual_new_requests=5,effective_public_output_tokens=768,elapsed_s=a["elapsed_s"],requests=rows,native_counter_deltas=deltas,physical=physical,native=native,cleanup=cleanup,comparison_with66=comparison,new_models=2,model_phase="D65 exactcleanup/newD67 install, mixed phase no model operations",CPU_issue_budget=4096,GPU_allocation=16384,conclusion="Native mixed4096issuebudget diagnostic; actualfunctional/chunklatency vs66 tobecompared, no formalSLO/capacity/KEEP.",limits=a["limits"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=out,reduction=str(j/"reduction.json"));atomic_json(r/"manifest.json",m);(r/"summary.md").write_text("# "+r.name+"\n\n"+json.dumps(dict(requests=5,outputs=768,TTFT=[v["ttft_s"]for v in rows],max_chunks=[v["max_native_chunk_gap_s"]for v in rows],new_D67=True))+"\n\n"+out["conclusion"]+"\n")
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run67VALIDfinite5mixednative requests/768commits/4096CPUissuebudget/nativeAsync/GPU16384/source/32newphysical/idle/counters; noKEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="nativewires/chunk-timing/arrival/counters/32NPUepochs")],unknowns=a["limits"],decision_request=None,next_check_at=None));print(json.dumps(out))

