from pathlib import Path
import sys,json,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0216"
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def metric(raw):
 out={}
 for l in raw.decode().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{")[0].split()[0]
  if k.startswith("vllm:"):out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
state=json.loads((r/"state.json").read_text());assert state["status"]=="failed"and not same_process(state["owner"])and state["completed_stages"]==["prepare","fault"]
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==181and all(st["sources"]==pins for st in spec["stages"])
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(x["path"])["sha256"]==x["sha256"]
phase=json.loads((r/"restore.phase.json").read_text());assert phase["status"]=="failed"and phase["exit_code"]==1and not phase["timed_out"]
error=(r/"restore.log").read_text();assert "KeyError: 'VLLM_USE_V2_MODEL_RUNNER'"in error and "D0_ready\": true"in error
original=(r/"live_probe.py").read_text();assert '"VLLM_USE_V2_MODEL_RUNNER"'not in original
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());plans=json.loads((r/"standalone_launch.json").read_text())
ready=json.loads((r/"D0_ready_members.stdout").read_text());roots["node0"]=ready["root"];members["node0"]=ready;plans["node0"]=json.loads((r/"candidate_plan.json").read_text())
log=Path(plans["node0"]["log"]);(j/"native216_resident.stdout").write_bytes(log.read_bytes())
physical={};policies={};rankmaps={}
physical={};policies={};rankmaps={}
plans=json.loads((folder/"standalone_launch.json").read_text())
for key,o in roots.items():
 node=o["host"];TP=8 if node=="166"else 4;PP=2 if node=="166"else 4;K=2 if node=="166"else 1
 args=["python3","-c",(j/"live_probe.py").read_text()]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 bypid={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(bypid[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 a=o["argv"]
 for flag,value in [("--nnodes","1"),("--tensor-parallel-size",str(TP)),("--pipeline-parallel-size",str(PP)),("--decode-context-parallel-size",str(TP)),("--max-num-batched-tokens","8192")]:assert a[a.index(flag)+1]==value
 assert json.loads(a[a.index("--speculative-config")+1])["num_speculative_tokens"]==K
 assert v["env"]["VLLM_PP_LAYER_PARTITION"]==("42,36"if node=="166"else "22,20,20,16")
 physical[key]=dict(root_same=True,actual_NPU16=True,root=v["root"],proof=ref(j/(key+".owner.stdout")),TP=TP,PP=PP,DCP=TP,K=K)
 path="/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp216"if node=="166"else "local_pp211")
 log=plans[key]["log"]
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(path)+");print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),files={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in p.glob('*.py')})))"
 args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);(j/("native_source_"+node+".stdout")).write_bytes(z.stdout);z.check_returncode();v=json.loads(z.stdout)
 assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0216"if node=="166"else "GLM-COHORT-0211",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 src=r/"plugin_src"if node=="166"else p/"runs/GLM-RUN-0211/plugin_src"
 for name,sha in v["files"].items():assert ref(src/name)["sha256"]==sha
 policies[node]=v
 logf=j/"native216_resident.stdout"if node=="166"else p/"runs/GLM-RUN-0211/D1_native_resident.stdout"
 text=logf.read_text();ev=[json.loads(l)for l in text.splitlines()if l.startswith('{"event":')]
 assert ev[0]["event"]=="task_acl_init"and ev[0]["returncode"]==0and not any(x["event"]=="task_acl_finalize"for x in ev)
 assert "Graph capturing finished"in text
 primary=json.loads((p/"runs/GLM-RUN-0190/restored/PP_empty_guard_workers.json").read_text())["markers"][0]
 markers=[json.loads(l.split("GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED ",1)[1])for l in text.splitlines()if "GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED "in l];assert len(markers)==16and all(x==primary for x in markers)
 entries=[]
 for line in text.splitlines():
  match=re.search(r"\(Worker pid=(\d+)\).*world_size=16 rank=(\d+) local_rank=(\d+)",line)
  if match:entries.append(dict(container_pid=int(match[1]),rank=int(match[2]),local_rank=int(match[3])))
 assert len(entries)==16and {x["rank"]for x in entries}==set(range(16))and all(x["rank"]==x["local_rank"]for x in entries)
 mapcode="import pathlib,json,sys;rows=[];a=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()\nfor x in a:\n p=pathlib.Path('/proc/'+str(x['pid']));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity'];ns=next(l for l in(p/'status').read_text().splitlines()if l.startswith('NSpid:')).split()[1:];rows.append(dict(host_pid=x['pid'],identity=x['identity'],container_pid=int(ns[-1])))\nprint(json.dumps(rows))"
 args=["python3","-c",mapcode]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 targets=[x for x in members[key]["owned_targets"]if x["pid"]in members[key]["npu_worker_pids"]]
 z=subprocess.run(args,input=json.dumps(targets).encode(),capture_output=True,timeout=60);z.check_returncode();(j/(key+".NSpid.stdout")).write_bytes(z.stdout);nspids=json.loads(z.stdout)
 assert {x["container_pid"]for x in nspids}=={x["container_pid"]for x in entries}
 mapped=[]
 for x in entries:
  hostrow=next(y for y in nspids if y["container_pid"]==x["container_pid"]);PPindex=x["rank"]//TP;TPindex=x["rank"]%TP
  label="Worker_PP"+str(PPindex)+"_TP"+str(TPindex)+"_DCP"+str(TPindex)+" pid="+str(x["container_pid"])
  assert label in text;mapped.append(dict(x,**hostrow,PP=PPindex,TP=TPindex,DCP=TPindex))
 rankmaps[node]=dict(source=ref(logf),actual_rank_HOST_NPU_map=mapped,NSpid_proof=ref(j/(key+".NSpid.stdout")),limits="init labels and current HOST NPU PID identity, not percollective GPU activity")

http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
native={}
for key,o in roots.items():
 with http.open("http://172.16.10."+o["host"]+":"+str(9081 if o["host"]=="166"else 9900)+"/health",timeout=15)as res:assert res.status==200
 with http.open("http://172.16.10."+o["host"]+":"+str(9081 if o["host"]=="166"else 9900)+"/metrics",timeout=15)as res:b=res.read()
 (j/(key+".metrics")).write_bytes(b);v=metric(b);assert v["vllm:num_requests_running"]==v["vllm:num_requests_waiting"]==v["vllm:kv_cache_usage_perc"]==0
 if key=="node0":assert v["vllm:generation_tokens_total"]==v["vllm:request_success_total"]==v["vllm:prompt_tokens_total"]==0
 else:
  before=metric((r/"before_167.metrics").read_bytes());assert all(v[k]==n for k,n in before.items()if k.endswith("_total"))
 native[key]=ref(j/(key+".metrics"))
root=roots["node0"];env=dict(x.decode().split("=",1)for x in Path("/proc/"+str(root["pid"])+"/environ").read_bytes().split(bytes([0]))if b"="in x);assert env["VLLM_USE_V2_MODEL_RUNNER"]=="1"
text=(j/"native216_resident.stdout").read_text();assert text.count("npu model runner v2 is in developing")==16and "Graph capturing finished"in text and "8/8"in text
capture_lines=[l for l in text.splitlines()if any(k in l for k in["Graph capturing finished","GPU KV cache size","Capturing CUDA graphs"])]
for mode in["before","survivor"]:
 c=json.loads((r/("client_"+mode+"_summary.json")).read_text());assert c["valid"]and c["effective_output_tokens"]==0and c["native_delta"]["D1"]["generation_tokens_total"]==0
 events=[json.loads(l)for l in(r/("client_"+mode+".stdout")).read_text().splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
 retained=next(x for x in c["requests"]if x["name"]=="D1_retained")
 source=json.loads((p/"runs/GLM-RUN-0211/client_final_summary.json").read_text());source=next(x for x in source["requests"]if x["name"]=="newD1_create")
 assert Path(retained["wire"]["path"]).read_bytes()==Path(source["wire"]["path"]).read_bytes()
 if mode=="survivor":
  for name in["retired214_get","retired214_previous"]:
   row=next(x for x in c["requests"]if x["name"]==name);assert row["status"]==503and row["rejected_before_lease_and_RPC"]
with http.open("http://127.0.0.1:8000/control/replicas",timeout=15)as res:placement=json.loads(res.read())
byid={x["id"]:x for x in placement["replicas"]};assert byid["D0"]["group_faulted"]and not byid["D1"]["group_faulted"]and all(not x["active_requests"]for x in placement["replicas"])
atomic_json(j/"adopted_roots.json",roots);atomic_json(j/"adopted_members.json",members);atomic_json(j/"adopted_launch.json",plans)
out=dict(at=utc(),run_id=r.name,driver_verdict="INVALID",partial_native_fit_VALID=True,source_count=181,active_NPU_ranks=32,physical=physical,rankinit=rankmaps,policies=policies,native=native,capture_lines=capture_lines,capture_source=ref(j/"native216_resident.stdout"),V2_flag_actual=True,V2_warning16=True,new_native_output_tokens=0,new_native_requests=0,retained_D1_all_native_counters=True,old214STORE503preleaseRPC=True,retained_STORE211_exact=True,SDKclient_before_survivor_init_finalize0=True,original_failure="Live_probe filters env and omitted VLLM_USE_V2_MODEL_RUNNER; driverKeyErrorafterready; modelnativefit success, no original source/raw edited",prototype_only=True,V2_thinking_token_budget_gap_open=True,full_native_request_equivalence=False,adopted_roots=ref(j/"adopted_roots.json"),adopted_members=ref(j/"adopted_members.json"),adopted_launch=ref(j/"adopted_launch.json"),Current=None,limits=["Readonly partialnativefit notfunction/performance/fullrequestequivalence/KEEP","Run216 remainsfailed/INVALIDdriver; nextRunadopts samehealthyV2model without reload orprior client replay","V1PPemptyguard imported notV2executionproof; nativeV2metadata/requestbehavior pending realE2E; thinkingbudgetgap open"])
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="216 frozen driverINVALID env-whitelistfixture; readonly nativeV2 actualfit Graph8/native32/zeroinference retainedD1/STORE211 SDK0 VALIDpartial, nofunctionperformanceKEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="partial-nativefit32-source181-exactowners-Graph8-frozenfailure")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(partial_fit_VALID=True,driverINVALID=True,Graph8=True,outputs=0,roots=roots)))
