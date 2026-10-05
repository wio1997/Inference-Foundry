from pathlib import Path
import sys,json,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0219";folder=r/"restored";state125=p/"runs/GLM-RUN-0125"
sys.path.insert(0,str(r/"runtime_bundle"))
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def metric(f):
 out={}
 for l in Path(f).read_text().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{")[0].split()[0]
  if k.startswith("vllm:"):out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==175and all(x["sources"]==pins for x in spec["stages"])
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(x["path"])["sha256"]==x["sha256"]
for n in["prepare","restore","switch","diagnose"]:
 v=json.loads((r/(n+".phase.json")).read_text());assert v["status"]=="succeeded"and v["exit_code"]==0and not v["timed_out"]
c=json.loads((r/"diagnostic_client_summary.json").read_text());assert c["HTTP_status"]==200and c["semantic_pass"]and c["effective_output_tokens"]==2
for k in["body","wire"]:assert ref(c[k]["path"])==c[k]
value=json.loads(Path(c["wire"]["path"]).read_text());choice=value["choices"][0];u=value["usage"]
assert choice["message"]["content"].strip()=="4"and choice["finish_reason"]=="stop"and len(choice["token_ids"])==u["completion_tokens"]==2and u["prompt_tokens"]==19and u["total_tokens"]==21
prior=json.loads((p/"runs/GLM-RUN-0217/client_final_add2_3.body").read_text());body=json.loads(Path(c["body"]["path"]).read_text())
prior.pop("cache_salt");body.pop("cache_salt");assert prior==body
trace=[json.loads(l)for l in(state125/"router_trace.jsonl").read_text().splitlines()]
lease=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==r.name+"-diagnostic-add2"];assert len(lease)==1and lease[0]["replica"]=="D0";lease=lease[0]
wire=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];released=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]]
assert len(wire)==len(released)==1and wire[0]["audit_error"]is None and released[0]["released"]and not released[0]["backend_failure"]
assert wire[0]["wire_sha256"]==c["wire"]["sha256"]and Path(wire[0]["wire_path"]).read_bytes()==Path(c["wire"]["path"]).read_bytes()
events=[json.loads(l)for l in(r/"diagnostic_client.stdout").read_text().splitlines()if l.startswith('{"event":')]
assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and all(x["returncode"]==0for x in events)
config,_=checked_config(folder/"service_config.json");old,_=checked_config(r/"service_config.json");epochs={x["id"]:x["epoch"]for x in config["native_domains"]};beforeepochs={x["id"]:x["epoch"]for x in old["native_domains"]}
assert epochs["local-211-167"]==beforeepochs["local-211-167"]and epochs["local-219-166"]!=beforeepochs["local-216-166"]
roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text());physical={};policies={};rankmaps={}
plans=json.loads((folder/"standalone_launch.json").read_text())
for key,o in roots.items():
 node=o["host"];TP=8 if node=="166"else 4;PP=2 if node=="166"else 4;K=2 if node=="166"else 1
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 bypid={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(bypid[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 a=o["argv"]
 for flag,value in [("--nnodes","1"),("--tensor-parallel-size",str(TP)),("--pipeline-parallel-size",str(PP)),("--decode-context-parallel-size",str(TP)),("--max-num-batched-tokens","8192")]:assert a[a.index(flag)+1]==value
 assert json.loads(a[a.index("--speculative-config")+1])["num_speculative_tokens"]==K
 assert v["env"]["VLLM_PP_LAYER_PARTITION"]==("42,36"if node=="166"else "22,20,20,16")
 physical[key]=dict(root_same=True,actual_NPU16=True,root=v["root"],proof=ref(j/(key+".owner.stdout")),TP=TP,PP=PP,DCP=TP,K=K)
 path="/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp219"if node=="166"else "local_pp211")
 log=plans[key]["log"]
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(path)+");print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),files={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in p.glob('*.py')})))"
 args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);(j/("native_source_"+node+".stdout")).write_bytes(z.stdout);z.check_returncode();v=json.loads(z.stdout)
 assert v["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0219"if node=="166"else "GLM-COHORT-0211",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 src=r/"plugin_src"if node=="166"else p/"runs/GLM-RUN-0211/plugin_src"
 for name,sha in v["files"].items():assert ref(src/name)["sha256"]==sha
 policies[node]=v
 logf=r/"D0_native_resident.stdout"if node=="166"else p/"runs/GLM-RUN-0211/D1_native_resident.stdout"
 text=logf.read_text();ev=[json.loads(l)for l in text.splitlines()if l.startswith('{"event":')]
 assert ev[0]["event"]=="task_acl_init"and ev[0]["returncode"]==0and not any(x["event"]=="task_acl_finalize"for x in ev)
 assert "npu model runner v2 is in developing"in text if node=="166"else "Graph capturing finished"in text
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

env=dict(x.decode().split("=",1)for x in Path("/proc/"+str(roots["node0"]["pid"])+"/environ").read_bytes().split(bytes([0]))if b"="in x)
assert env["VLLM_USE_V2_MODEL_RUNNER"]=="1"and env["ASCEND_LAUNCH_BLOCKING"]=="1"
a=roots["node0"]["argv"];assert "--enforce-eager"in a and json.loads(a[a.index("--compilation-config")+1])==dict(cudagraph_mode="NONE")
raw=Path(plans["node0"]["log"]).read_bytes();(j/"native219_resident.stdout").write_bytes(raw);text=raw.decode(errors="replace")
markers=[json.loads(l.split("GLM_SFA_DCP_METADATA_DIAGNOSTIC_INSTALLED ",1)[1])for l in text.splitlines()if "GLM_SFA_DCP_METADATA_DIAGNOSTIC_INSTALLED "in l];assert len(markers)==16and all(x["math_changes"]==0for x in markers)
metadata=[json.loads(l.split("GLM_SFA_DCP_METADATA_DIAGNOSTIC ",1)[1])for l in text.splitlines()if "GLM_SFA_DCP_METADATA_DIAGNOSTIC "in l];assert metadata
assert all(x["query_GPU"]==x["query_CPU"]and x["query_sum"]==x["repeat_output_size"]and x["num_actual_tokens"]<=x["query_sum"]for x in metadata)
atomic_json(j/"runtime_metadata.json",metadata)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for node,port in[("166",9081),("167",9900)]:
 with http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=15)as res:b=res.read()
 f=j/("terminal_"+node+".metrics");f.write_bytes(b);v=metric(f);assert v["vllm:num_requests_running"]==v["vllm:num_requests_waiting"]==v["vllm:kv_cache_usage_perc"]==0
 if node=="166":
  assert v["vllm:generation_tokens_total"]==2and v["vllm:request_success_total"]==1and v["vllm:prompt_tokens_total"]==19and v["vllm:num_preemptions_total"]==0
  assert v["vllm:spec_decode_num_draft_tokens_total"]==2*v["vllm:spec_decode_num_drafts_total"]and v["vllm:spec_decode_num_drafts_total"]>0
 else:
  before=metric(r/"before_167.metrics");assert all(v[k]==n for k,n in before.items()if k.endswith("_total"))
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and public["SDK_init0"]and owner_alive(json.loads((folder/"identity_observer/process_owner.json").read_text()))
obs=observe(folder/"service_config.json");assert all(x["status"]=="healthy"for x in obs["groups"]);atomic_json(j/"HOST_observer_now.json",obs)
retired=json.loads((r/"retire217_public.json").read_text());assert retired["SDKinit_finalize0"]and retired["models_signalled"]==0and(p/"runs/GLM-RUN-0217/restored/identity_observer/terminal.json").exists()
out=dict(at=utc(),run_id=r.name,measurement_valid=True,diagnostic_semantic_pass=True,full_functional_acceptance_unknown=True,verdict="INCONCLUSIVE",source_count=175,physical=physical,rankinit=rankmaps,policies=policies,active_NPU_ranks=32,outputs=2,prompt_tokens=19,new_native_completed=1,K2draft_positive=True,V2flag_actual=True,blocking_actual=True,targetGraph_NONE=True,runtime_metadata=ref(j/"runtime_metadata.json"),metadata_rows=len(metadata),query_extent_all_matched=True,prior217_body_same_except_coldsalt=True,native_wire_same=True,SDKclient_init_finalize0=True,D1_all_totals_retained=True,D1epoch_unchanged=True,new_native_epochs=epochs,public=public,public217_SDKfinal0=True,public219_SDKinit0active=True,performance_claim=False,Current=None,limits=["One eagerblocking true E2E semantic4/19prompt/2tokens/native32/source175/SDK0; notfullAPI/functioncomplete orKEEP","Eager runtimequeryGPU/CPU sum=inputextent; padding hypothesis notproven untilsameGraph observed","Native Graph+blocking guard preserved; nextsameGraph metadata observation no modelmath/operator/guardbypass","KnownV2thinking_token_budget gap open; comparison eager+blocking+epoch+observation changes no isolatedcausality/performanceclaim"])
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="219 trueE2E eagerblocking first19prompt semantic4/native2tokens1request/K2draft/SDK0/native32-source175 VALID diagnostic; eagerqueryextentmatches, fullAPI/Graphcause/performance unknown",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="native32-source175-actualeagerblocking-query-19prompt2tokens1request-SDK0")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(diagnostic_VALID=True,outputs=2,NPU32=True,epochs=epochs)))
