from pathlib import Path
import json,sys,time,hashlib,re,subprocess,shlex,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0108";deadline=time.monotonic()+6500
while True:
 state=json.loads((r/"state.json").read_text())
 if state["status"]!="running":break
 assert same_process(state["owner"])and time.monotonic()<deadline;time.sleep(10)
assert not same_process(state["owner"])
semantic=json.loads((r/"semantic_summary.json").read_text())if(r/"semantic_summary.json").exists()else None
def ref(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
spec=json.loads((r/"controller_spec.json").read_text())
for v in spec["stages"][0]["sources"]:
 raw=Path(v["path"]).read_bytes();assert raw==Path(v["snapshot"]).read_bytes()and hashlib.sha256(raw).hexdigest()==v["sha256"]
logs={};native_errors={};planned=json.loads((r/"planned_launch.json").read_text())
for x in planned:
 args=["cat",x["log"]]
 if x["node"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=90);f=j/(x["node"]+".native.log");f.write_bytes(z.stdout)
 logs[x["node"]]=dict(raw=ref(f),returncode=z.returncode)
 errors=[dict(line=i,text=l[:2000])for i,l in enumerate(z.stdout.decode(errors="replace").splitlines(),1)if any(k in l for k in["OutOfMemoryError","Engine core initialization failed","RuntimeError:","ValueError:","Traceback (most recent call last)","SIGABRT","ValidationError:","GLM_UNUSED_SFA_WORKSPACE_READ"])]
 native_errors[x["node"]]=errors[:30]
if state["status"]!="completed":
 phase=r/(str(state.get("failure_phase")or state["active_stage"])+".phase.json")
 attempts=json.loads((r/"attempts.json").read_text())if(r/"attempts.json").exists()else[]
 out=dict(at=utc(),functional_acceptance=False,verdict="REJECT"if any(native_errors.values())or(semantic and not semantic["semantic_acceptance"])else"INCONCLUSIVE",state=state,phase=json.loads(phase.read_text())if phase.exists()else None,source_pins=len(spec["stages"][0]["sources"]),native_logs=logs,native_errors=native_errors,attempts=attempts,startup_owners=json.loads((r/"startup_model_identities.json").read_text())if(r/"startup_model_identities.json").exists()else None,effective_public_output_tokens=sum(a.get("effective_public_output_credit",0)for a in attempts),limits=["Actualfailedphase/raw retained; no fit/fullE2E/KEEP proof; source or observationfixture failure may require GPTINVALID judgment","No audit-inducedmodel/native signals; exactcurrentresource reconciliation stillrequired"])
else:
 summary=json.loads((r/"pilot_summary.json").read_text());assert summary["functional_acceptance"]and len(summary["requests"])==2 and summary["effective_public_output_tokens"]==96
 for name in["deploy","semantic","pilot"]:
  phase=json.loads((r/(name+".phase.json")).read_text());assert phase["status"]=="succeeded"and phase["exit_code"]==0 and not phase["timed_out"]
 rows=[]
 for x in summary["requests"]:
  raw=Path(x["wire"]["path"]).read_bytes();assert len(raw)==x["wire"]["bytes"]and hashlib.sha256(raw).hexdigest()==x["wire"]["sha256"]
  obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True)
  for n in range(0,len(raw),733):obs.feed(raw[n:n+733])
  con=obs.contract();assert con["done"]and not con["unknown"]and not con["native_error"]and con["usage"]==x["usage"]and con["finish_reasons"]=={"0":"length"}
  frames=[]
  for part in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
   d=b"\n".join(l[5:].removeprefix(b" ")for l in part.splitlines()if l.startswith(b"data:"))
   if d and d!=b"[DONE]":frames.append(json.loads(d))
  ids=[t for f in frames for ch in f.get("choices",[])for t in ch.get("token_ids")or[]];assert len(ids)==x["usage"]["completion_tokens"]==x["expected_output_tokens"]
  prompts=[f["prompt_token_ids"]for f in frames if f.get("prompt_token_ids")is not None];assert prompts and len(prompts[0])==x["usage"]["prompt_tokens"]
  body=(r/(x["name"]+".body.json")).read_bytes();v=json.loads(body);assert hashlib.sha256(body).hexdigest()==x["request_body_sha256"]and v["cache_salt"].startswith(r.name)and not v.get("kv_transfer_params")
  rows.append({k:x[k]for k in["name","owner","kind","usage","ttft_s","wall_s","wire","effective_public_output_credit"]})
 def metrics(path):
  out={}
  for l in path.read_text().splitlines():
   if not l or l.startswith("#"):continue
   name=l.split("{")[0].split()[0]
   if any(k in name for k in["prompt_tokens_total","generation_tokens_total","request_success_total","prefix_cache_queries_total","prefix_cache_hits_total","num_preemptions_total","num_requests_running","num_requests_waiting","kv_cache_usage_perc"]):out[name]=out.get(name,0)+float(l.rsplit(" ",1)[1])
  return out
 counters={}
 for key in["D0","D1"]:
  before=metrics(r/("initial_"+key+".metrics"));after=metrics(r/("D1_after_"+key+".metrics"));delta={k:after.get(k,0)-v for k,v in before.items()if k.endswith("_total")}
  assert delta["vllm:prompt_tokens_total"]==(81953 if key=="D1"else 0) and delta["vllm:generation_tokens_total"]==(96 if key=="D1"else 0) and delta["vllm:request_success_total"]==(2 if key=="D1"else 0)
  assert delta.get("vllm:prefix_cache_hits_total",0)==delta.get("vllm:external_prefix_cache_hits_total",0)==delta.get("vllm:num_preemptions_total",0)==0
  assert all(after.get(k,0)==0 for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"])
  if key=="D1":
   cb=metrics(r/(key+"_before_"+key+".metrics"));ce=after;assert ce["vllm:prompt_tokens_total"]-cb["vllm:prompt_tokens_total"]==81932 and ce["vllm:generation_tokens_total"]-cb["vllm:generation_tokens_total"]==64
  counters[key]=dict(delta=delta,cold_verified=True,idle=True)
 CPU=[];install={}
 for node in["166","167"]:
  prior=r if node=="167"else r.parent/"GLM-RUN-0089"
  f=prior/("CPU_fullCLI_"+node+".stdout");acks=[json.loads(l)for l in f.read_text().splitlines()if l.startswith('{"event":')]
  assert len(acks)==(3 if node=="166"else 4) and acks[0]["returncode"]==acks[-1]["returncode"]==0
  proof=acks[1]["DSA_CP_native_proof"]
  if node=="167":assert proof["enable_expert_parallel"]is False and proof["native_comm_method_allgather_for_tokens"]==[4,8,32,4096]
  assert not proof["enabled"]and proof["SP_MoE"]is False and not proof["old_zero_workspace_guard_eligible"]and proof["shared_expert_overlap"]is False and not proof["o_proj_tp_enabled"]
  assert acks[1]["kv_role"]==("kv_producer"if node=="166"else"kv_consumer")and acks[1]["DP"]==1 and acks[1]["TP"]==acks[1]["DCP"]==(16 if node=="166"else 8)
  assert acks[1]["speculative_tokens"]==(5 if node=="166"else 3)
  assert acks[1]["capture_geometry"]["sizes"]==([6,12,24,48]if node=="166"else[4,8,16,32])
  CPU.append(dict(node=node,raw=ref(f),ack=acks[1],reused_P89=node=="166"))
  text=(j/(node+".native.log")).read_text(errors="replace");assert not native_errors[node]
  assert text.count("GLM_DP_METADATA_INSTALLED ")==0 and text.count("GLM_UNUSED_SFA_WORKSPACE_GUARD_INSTALLED ")==0 and "GLM_UNUSED_SFA_WORKSPACE_SKIPPED "not in text and "Graph capturing finished"in text
  assert "DSA-CP is enabled"not in text
  assert "Sequence-parallel MoE is enabled"not in text
  budget=[json.loads(l.split("GLM_ISSUE_BUDGET_INSTALLED ",1)[1])for l in text.splitlines()if"GLM_ISSUE_BUDGET_INSTALLED "in l]
  assert len(budget)==1 and budget[0]["native_base"]=="AsyncScheduler"and budget[0]["allocated_scheduler_max"]==4096
  expected=dict(cohort="GLM-COHORT-0089"if node=="166"else"GLM-COHORT-0108",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1,policy_error=None,fallback=False,native_max=4096)
  selected=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in text.splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in l]
  if node=="167":assert expected in selected
  else:expected=dict(policy_file_serial=3,native_schedule_unexercised_by_Run108=True,last_historical_selected=selected[-1]if selected else None)
  install[node]=dict(coupled_metadata_notinstalled=True,workspace_guard_notinstalled_native_allocation=True,DSACP_disabled=True,SP_enabled=False,Graph=True,native_async=True,selected_policy=expected)
 members=json.loads((r/"native_member_identities.json").read_text());API=json.loads((r/"adopted_model_identities.json").read_text());physical={}
 assert "--enable-expert-parallel"not in API["D1"]["argv"]and API["D1"]["expert_parallel_enabled"]is False and API["D1"]["native_ep_group_world_size"]==8
 check="""import pathlib,json,sys,subprocess,re
a=json.load(sys.stdin);o=a['owner'];boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for x in a['targets']+[dict(pid=o['pid'],identity=o['identity'])]:
 p=pathlib.Path('/proc/'+str(x['pid']));b=(p/'stat').read_text();v=b[b.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity']
assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
env=dict(v.decode().split('=',1)for v in pathlib.Path('/proc/'+str(o['pid'])+'/environ').read_bytes().split(bytes([0]))if b'='in v)
if o['host']=='167':assert env['VLLM_PP_LAYER_PARTITION']=='42,36'and '--enable-expert-parallel'not in o['argv']
raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);ids={int(v)for v in re.findall(r'^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker',raw,re.M)};assert ids=={x['pid']for x in a['targets']}
policy=json.loads((pathlib.Path(o['argv'][1]).parent/'issue_budget_policy.json').read_text());assert policy==a['policy'];print(json.dumps(dict(API_same=True,NPU16_same=True,npu_pids=sorted(ids),policy=policy)))
"""
 for key,o in API.items():
  args=["python3","-c",check]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  targets=[v for v in members[key]["owned_targets"]if v["pid"]in members[key]["npu_worker_pids"]];z=subprocess.run(args,input=json.dumps(dict(owner=o,targets=targets,policy=dict(schema_version=1,cohort_id="GLM-COHORT-0089"if key=="D0"else"GLM-COHORT-0108",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3 if key=="D0"else 1))).encode(),capture_output=True,timeout=90);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();physical[key]=json.loads(z.stdout)
 cleanup={}
 for node in["167"]:
  f=r/("cleanup_"+node+".stdout");events=[json.loads(l)for l in f.read_text().splitlines()if l.startswith("{")]
  assert events[0]["event"]=="ownership_preflight"and events[-1]["event"]=="cleanup_complete"and events[-1]["npu_workers"]==0 and events[-1]["signals_to_unknown"]==events[-1]["signals_to_inactive"]==0
  cleanup[node]=dict(raw=ref(f),exact107_D_owner=True,allNPUempty_before_new=True,signals_to_unknown=0,signals=[x for x in events if x["event"]=="signal_sent"])
 out=dict(at=utc(),functional_acceptance=True,verdict="INCONCLUSIVE",new_models=1,retained_P89=True,new_physical_cohort="retainedP89+newD108 DP1TP8PP2DCP8 EPdisabled_nativeAllGather partition42,36/PnoSP_DnoSP_DSACPoff",effective_public_output_tokens=96,requests=rows,counters=counters,CPU_actual_preflight=CPU,native_install=install,physical_same=physical,priorD107_exact_cleanup=cleanup,native_logs=logs,source_pins=len(spec["stages"][0]["sources"]),limits=["Finite1newDshort32+1NEWcold81932to64 directnative independentEP16 functional diagnostic, no fullgateway/state/tools/PD/SLO/stablecapacity/KEEP","D108nativeMTP3SPFALSE_DSACPoff/DP1EP16 replacesD107, KV29000..29015 belowOSephemeralrange; nativeNoSPlegalcapture4,8,16,32 comparedpriornoSP6,12,24,48. P89exactAPI/NPU16/policy3 retained. RestnativeAsync/GPU4096/KV3GiB/HCCL768/budget4096t1024c1 unchanged; versus79classchange noisolatedspeedup","Nativeoperators/math/W8A8/MLAworkspace retained; AtomicMQWorker/no coupledDPmetadata; no rootcause/quality/capacity promotion"])
out["semantic_diagnostic"]=semantic
if state["status"]=="completed":
 old=json.loads((r.parent/"GLM-RUN-0089/adopted_model_identities.json").read_text())
 assert API["D0"]==old["D0"]
 oldmembers=json.loads((r.parent/"GLM-RUN-0089/native_member_identities.json").read_text())
 assert members["D0"]["npu_worker_pids"]==oldmembers["D0"]["npu_worker_pids"]
 expected_ids={v["pid"]:v["identity"]for v in oldmembers["D0"]["owned_targets"]if v["pid"]in oldmembers["D0"]["npu_worker_pids"]}
 assert all(expected_ids[v["pid"]]==v["identity"]for v in members["D0"]["owned_targets"]if v["pid"]in members["D0"]["npu_worker_pids"])
 out["retained_P89_same_epoch_NPU16"]=True
if state["status"]=="completed":
 assert semantic and semantic["semantic_acceptance"]and semantic["semantic_pass_cases"]==semantic["semantic_cases"]==3
 for row in semantic["requests"]:
  f=Path(row["wire"]["path"]);raw=f.read_bytes();assert len(raw)==row["wire"]["bytes"]and hashlib.sha256(raw).hexdigest()==row["wire"]["sha256"]
  v=json.loads(raw);assert not v.get("error")and v["choices"][0]["message"]["content"].strip()==row["expected"]
  assert v["usage"]==row["usage"]and len(v["choices"][0]["token_ids"])==v["usage"]["completion_tokens"]
  f=Path(row["body"]["path"]);assert ref(f)==row["body"]
  assert row["http_status"]==200 and row["semantic_pass"]
 out["semantic_output_tokens"]=semantic["effective_public_output_tokens"]
 out["pilot_output_tokens"]=out["effective_public_output_tokens"]
 out["effective_public_output_tokens"]+=out["semantic_output_tokens"]
else:
 out["functional_acceptance"]=False
 if semantic and not semantic["semantic_acceptance"]:out["verdict"]="REJECT"
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);raw=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run108 "+state["status"]+" newD108TP8PP2nativeNoSP_EP8perPP/P89retained diagnostic raw/source/phase/semantic audit; functional="+str(out["functional_acceptance"])+"/"+str(out["effective_public_output_tokens"])+" outputs; noKEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),locator="actualphase/source/nativeerror-or-wire/tokenIDs/counters/DSACP/config/epoch")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(status=state["status"],functional=out["functional_acceptance"],outputs=out["effective_public_output_tokens"],native_errors=native_errors)))

