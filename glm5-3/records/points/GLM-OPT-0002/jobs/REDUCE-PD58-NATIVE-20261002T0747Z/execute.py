import sys,json,subprocess,shlex,hashlib,re,urllib.request
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0058";http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def ref(f):
 d=f.read_bytes();return{"path":str(f),"bytes":len(d),"sha256":hashlib.sha256(d).hexdigest()}
def cmd(node,args):
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 p=subprocess.run(args,capture_output=True,timeout=100);return p
state=json.loads((r/"state.json").read_text());assert state["status"]in["failed","completed"]and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==32
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(Path(x["path"]))["sha256"]==x["sha256"]
planned=json.loads((r/"planned_launch.json").read_text());logs={};native=[]
for x in planned:
 p=cmd(x["node"],["cat",x["log"]]);key=x["role"]+str(x["rank"])
 if p.returncode!=0:logs[key]={"present":False,"stderr":p.stderr.decode(errors="replace")};continue
 f=j/(key+".native.log");f.write_bytes(p.stdout);lines=p.stdout.decode(errors="replace").splitlines()
 receipts={mark:[l for l in lines if mark+" "in l]for mark in["GLM_DP_METADATA_INSTALLED","GLM_ATOMIC_MQ_BIND_INSTALLED","GLM_UNUSED_SFA_WORKSPACE_GUARD_INSTALLED","GLM_UNUSED_SFA_WORKSPACE_SKIPPED"]}
 logs[key]={"ref":ref(f),"present":True,"receipts":receipts,"KV_capacity":[l for l in lines if "GPU KV cache size:"in l],"weights":[l for l in lines if "Model loading took"in l or"model weights take"in l],"first_errors":[l for l in lines if any(z in l for z in [" ERROR ","OutOfMemoryError","Address already in use","KeyError:","RuntimeError:","AssertionError:"])][:20]}
for node in["166","167"]:
 for name,args in [("processes",["docker","top","glm52-single","-eo","pid,ppid,comm,args"]),("sockets",["ss","-ltnp"]),("npu-smi",["npu-smi","info"])]:
  p=cmd(node,args);p.check_returncode();(j/(node+"."+name)).write_bytes(p.stdout)
 index=json.loads((j.parents[1]/"jobs/PD2-CPU-DIAGNOSTIC-20261002T0158Z/source_index.json").read_text())
 for x in index:
  p=cmd(node,["docker","exec","glm52-single","sha256sum",x["native_path"]]);p.check_returncode();assert p.stdout.decode().split()[0]==x["sha256"]
owners=json.loads((r/"startup_model_identities.json").read_text())if(r/"startup_model_identities.json").exists()else json.loads((r/"prior_model_owners.json").read_text());current={}
for key,o in owners.items():
 code="import pathlib,json;p=pathlib.Path('/proc/"+str(o["pid"])+"');d={'present':p.exists()};\nif p.exists():\n s=(p/'stat').read_text();d.update(boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=s[s.rfind(')')+2:].split()[19],argv=[x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x]);\nprint(json.dumps(d))"
 p=cmd(o["host"],["python3","-c",code]);p.check_returncode();d=json.loads(p.stdout);current[key]=d
 if d["present"]:assert {k:d[k]for k in["boot_id","start_ticks"]}==o["identity"]and d["argv"]==o["argv"]
rows=json.loads((r/"attempts.json").read_text())if(r/"attempts.json").exists()else[];valid_rows=[]
for row in rows:
 f=r/(row["name"]+".body.json");assert ref(f)["sha256"]==row["request_body_sha256"]
 f=Path(row["wire"]["path"]);assert ref(f)["sha256"]==row["wire"]["sha256"]
 if not row["completed"]:continue
 body=json.loads((r/(row["name"]+".body.json")).read_text());raw=f.read_bytes()
 if body.get("stream"):
  obs=NativeSSEObserver(collect_contract=True);obs.feed(raw);c=obs.contract();assert c["done"]and not c["native_error"]and not c["unknown"]and c["finish_reasons"]=={"0":"length"};u=c["usage"]
 else:
  v=json.loads(raw);assert not v.get("error")and len(v["choices"])==1 and v["choices"][0]["finish_reason"]=="length";u=v["usage"]
 assert u==row["usage"]and u["completion_tokens"]==row["expected_output_tokens"]and u["total_tokens"]==u["prompt_tokens"]+u["completion_tokens"]
 if row["kind"]=="internal_P_KV_helper":
  assert row["effective_public_output_credit"]==0 and u["completion_tokens"]==1
 else:assert row["effective_public_output_credit"]==u["completion_tokens"]
 valid_rows.append(row)
def metric(f,key):
 lines=re.findall(r"^vllm:"+re.escape(key)+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",f.read_text(),re.M);return sum(float(x)for x in lines)if lines else None
metrics={};idle={};transfers={}
envs={}
for key,o in owners.items():
 if not current[key]["present"]:continue
 expected=next(x["env"]for x in planned if x["role"]+str(x["rank"])==key)
 env_code="import pathlib,json;e=dict(z.decode().split('=',1)for z in pathlib.Path('/proc/"+str(o["pid"])+"/environ').read_bytes().split(bytes([0]))if b'='in z);print(json.dumps({k:e.get(k)for k in ['HCCL_NPU_SOCKET_PORT_RANGE','HCCL_HOST_SOCKET_PORT_RANGE','HCCL_BUFFSIZE']}))"
 z=cmd(o["host"],["python3","-c",env_code]);z.check_returncode();envs[key]=json.loads(z.stdout);assert envs[key]==expected
if state["status"]=="completed":
 assert len(rows)==len(valid_rows)==11 and all(x["present"]for x in current.values())
 for key,o in owners.items():
  for ep in["health","metrics"]:
   with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/"+ep,timeout=10)as z:assert z.status==200;(j/(key+"."+ep)).write_bytes(z.read())
  idle[key]={k:metric(j/(key+".metrics"),k)for k in["num_requests_running","num_requests_waiting"]};assert set(idle[key].values())=={0.0}
  assert len(logs[key]["receipts"]["GLM_DP_METADATA_INSTALLED"])==16 and len(logs[key]["receipts"]["GLM_ATOMIC_MQ_BIND_INSTALLED"])==16
  first=r/("initial_"+key+".metrics");final=r/("final_"+key+".metrics")
  metrics[key]={k:None if metric(first,k)is None or metric(final,k)is None else metric(final,k)-metric(first,k)for k in["generation_tokens_total","num_preemptions_total","prefix_cache_queries_total","prefix_cache_hits_total","spec_decode_num_drafts_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total","request_prefill_time_seconds_sum"]}
 for n,pkey,dkey in[(0,"P0","D1"),(1,"P1","D0"),(2,"P1","D1"),(3,"P0","D0")]:
  meta=json.loads((r/("transfer_metadata_"+str(n)+".json")).read_text());pb=json.loads((r/("prefill_"+str(n)+".body.json")).read_text());db=json.loads((r/("decode_"+str(n)+".body.json")).read_text())
  assert db["kv_transfer_params"]==meta and pb["messages"]==db["messages"]
  assert meta["do_remote_prefill"]and not meta["do_remote_decode"]and meta["remote_dcp_size"]==16 and meta["remote_pcp_size"]==1 and meta["remote_ptp_size"]==16 and meta["remote_block_ids"]and all(isinstance(v,list)and v for v in meta["remote_block_ids"])
  registered=json.loads((r/"registered_P_engine_identities.json").read_text());assert registered[pkey]["API_owner"]==owners[pkey] and registered[pkey]["num_blocks"]>=42
  from registered_kv_identity import verify_registered
  verify_registered(registered)
  assert meta["remote_engine_id"]==registered[pkey]["engine_id"]and meta["remote_request_id"]and meta["remote_multi_nodes_meta_mapping"]
  transfers[str(n)]={"from":pkey,"to":dkey,"metadata":meta,"source":"Installednative CP-aware scheduler requiresfullprompt externalKV onfreshnonemptyremote metadata and async receive beforedecode; clientmetadata roundtrip/nativecompletion proof. No directper-shardlivebytestrace.","consumer_profile":{key:[l for l in (r/("pd"+str(n)+"_after_"+dkey+".metrics")).read_text().splitlines()if"kv" in l.lower()and ("transfer"in l or"fail"in l)]for key in["native_transfer_metric_lines"]}}
 assert next(x for x in valid_rows if x["name"]=="D0_fullinput_local_fallback")["usage"]["prompt_tokens"]>81933
 cp=[]
 for f in sorted(r.glob("CPU_fullCLI*.stdout")):
  events=[json.loads(l)for l in f.read_text().splitlines()if l.startswith('{"event":')];assert len(events)==3 and all(z["returncode"]==0 for z in[events[0],events[-1]])and events[1]["invalid_glm48_contract_rejected"];cp.append(events)
 assert len(cp)==4
out={"run_id":r.name,"at":utc(),"measurement_valid":True,"functional_acceptance":state["status"]=="completed","verdict":"INCONCLUSIVE","state":state,"source_pins":len(pins),"native_logs":logs,"current_owners":current,"inference_attempts":len(rows),"completed_native_requests":len(valid_rows),"effective_public_output_tokens":sum(x["effective_public_output_credit"]for x in valid_rows),"internal_P_helper_commits":sum(x["usage"]["completion_tokens"]for x in valid_rows if x["kind"]=="internal_P_KV_helper"),"native_committed_outputs":sum(x["usage"]["completion_tokens"]for x in valid_rows),"attempts":rows,"counter_deltas":metrics,"native_idle":idle,"transfer_metadata":transfers,"actual_API_HCCL_env":envs,"signals":0,"new_models":0,"new_requests":0,"limits":["No stablecapacity/KEEP/publicgateway/fullnativeAPI/stateproof","Ifcompleted directnative11attemptPDpilot only; failedphase hasactualzero/partialledger, no extrapolation; actualsamecheckpoint/nativeoperators no qualitycomparison or isolatedgain","Per-shardactualKVbytecounts/livepadding absent unlessnativecapturecontainsdirectevidence; source+metadata+completeddecode notbyteracecertificate","SDKwrapper/processcleanup/nativeGPUpeak notcertified by config/health","Failednativework/overshoot getszeroeffectiveoutputcredit"]}
atomic_json(j/"reduction.json",out)
brief={k:v for k,v in out.items()if k not in["native_logs","attempts","transfer_metadata"]};brief["native_log_refs"]={k:{"ref":v.get("ref"),"first_errors":v.get("first_errors"),"metadata_count":len(v.get("receipts",{}).get("GLM_DP_METADATA_INSTALLED",[])),"mq_count":len(v.get("receipts",{}).get("GLM_ATOMIC_MQ_BIND_INSTALLED",[])),"KV_capacity":v.get("KV_capacity")}for k,v in logs.items()};brief["full_reduction"]=ref(j/"reduction.json");atomic_json(r/"reduction_brief.json",brief)
m=json.loads((r/"manifest.json").read_text());m.update(status=state["status"],valid=out["functional_acceptance"],verdict="INCONCLUSIVE",reduction=ref(j/"reduction.json"),results={k:out[k]for k in["inference_attempts","completed_native_requests","effective_public_output_tokens","internal_P_helper_commits","native_committed_outputs"]});atomic_json(r/"manifest.json",m)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Run58terminalaudit actual"+state["status"]+"/"+str(state["active_stage"])+", attempts"+str(len(rows))+"/complete"+str(len(valid_rows))+"/publicoutputs"+str(out["effective_public_output_tokens"])+";32pins/native9source/physicalowners checked; noKEEP/capacity","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="originalnativewire/metadata/counters/logs/owners/source",**ref(j/"reduction.json"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"native_status":state["status"],"attempts":len(rows),"public_outputs":out["effective_public_output_tokens"],"reduction":ref(j/"reduction.json")}))

