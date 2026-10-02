from pathlib import Path
import json,sys,hashlib,re,subprocess,shlex,urllib.request,math,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0097"
deadline=time.monotonic()+2200
while True:
 state=json.loads((r/"state.json").read_text())
 if state["status"]!="running":break
 assert same_process(state["owner"])and time.monotonic()<deadline;time.sleep(10)
assert state["status"]=="completed"and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for v in spec["stages"][0]["sources"]:
 b=Path(v["path"]).read_bytes();assert b==Path(v["snapshot"]).read_bytes()and hashlib.sha256(b).hexdigest()==v["sha256"]
for stage in ["prepare","fullapi"]:
 a=json.loads((r/(stage+".phase.json")).read_text());assert a["status"]=="succeeded"and a["exit_code"]==0 and not a["timed_out"]
summary=json.loads((r/"full_api_summary.json").read_text());assert summary["functional_acceptance"]and summary["Responses"]["new_effective_output_tokens"]==384 and summary["tool_cases"]==8
trace=[json.loads(x)for x in(r/"router_trace.jsonl").read_text().splitlines()]
owners=json.loads((r/"response_owners.json").read_text())["owners"];assert(r/"response_owners.json").stat().st_mode&0o777==0o600
def ref(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def wire_contract(header,body,wire):
 leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==header];assert len(leases)==1;l=leases[0]
 assert l["body_sha256"]==hashlib.sha256(body.read_bytes()).hexdigest()and l["replica"]in["D0","D1"]
 ss=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==l["lease_id"]];assert len(ss)==1;s=ss[0]
 assert s["audit_error"]is None and Path(s["wire_path"]).read_bytes()==wire.read_bytes()and s["wire_sha256"]==hashlib.sha256(wire.read_bytes()).hexdigest()and s["wire_bytes"]==wire.stat().st_size
 releases=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==l["lease_id"]];assert len(releases)==1 and releases[0]["released"]and not releases[0]["backend_failure"]
 payload=json.loads(body.read_bytes())if body.stat().st_size else None
 if l.get("path")=="/v1/responses"and isinstance(payload,dict)and type(payload.get("max_output_tokens"))is int and payload["max_output_tokens"]>0:assert l["output_budget"]==payload["max_output_tokens"]
 return dict(replica=l["replica"],lease_id=l["lease_id"],body=ref(body),wire=ref(wire),affinity=l["affinity_applied"],gateway_output_budget=l["output_budget"])
rr=[]
for a in json.loads((r/"attempts.json").read_text()):
 assert a["outcome"]=="native_http_valid"
 w=Path(a["wire"]["path"]);b=Path(a["body"]["path"]);c=wire_contract(r.name+"-"+a["id"],b,w);assert c["replica"]==a["native_owner"]
 row=dict(id=a["id"],method=a["method"],path=a["path"],status=a["http_status"],contract=c)
 if a["http_status"]==200 and a["method"]=="POST"and a["path"].split("?")[0]=="/v1/responses":
  raw=w.read_bytes()
  if raw.startswith(b"event:"):
   frames=[]
   for f in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
    data=b"\n".join(l[5:].removeprefix(b" ")for l in f.splitlines()if l.startswith(b"data:"))
    if data:frames.append(json.loads(data))
   assert frames[-1]["type"]=="response.completed";v=frames[-1]["response"]
  else:v=json.loads(raw)
  assert owners[v["id"]]["replica"]==c["replica"]
 rr.append(row)
assert set(owners[k]["replica"]for k in["resp_glm_run97_base","resp_glm_run97_other"])=={"D0","D1"}
ts=[];tooloutputs=0
for label in["gateway_first","gateway_second"]:
 d=r/("tools_"+label);a=json.loads((d/"summary.json").read_text());assert a["valid"]and len(a["requests"])==4
 for x in a["requests"]:
  c=wire_contract(r.name+"-tool-"+label+"-"+x["name"],d/(x["name"]+".body.json"),d/(x["name"]+".wire"));assert x["valid"]and x["status"]==200 and x["usage"]["completion_tokens"]>0
  if x["stream"]:
   obs=NativeSSEObserver(collect_contract=True);obs.feed((d/(x["name"]+".wire")).read_bytes());con=obs.contract();assert con["done"]and not con["unknown"]and not con["native_error"]and con["usage"]==x["usage"]
  ts.append(dict(label=label,name=x["name"],native=c["replica"],usage=x["usage"],wire=c["wire"],body=c["body"]))
 tooloutputs+=a["effective_output_tokens"]
assert tooloutputs==summary["tool_outputs"]and tooloutputs+384==summary["effective_public_output_tokens"]
for name in["auto","required","named","none"]:assert{x["native"]for x in ts if x["name"]==name}=={"D0","D1"}
events=json.loads((r/"state_events.json").read_text());live=[x for x in events if x["event"]=="native_job_outlives_zero_HTTP_leases"];assert len(live)==1 and live[0]["native"]["num_requests_running"]>0 and all(x["active_requests"]==0 for x in live[0]["placement"]["replicas"])
assert all(x["placement_policy"]=="active_count"and "prefill_input_bytes_hint"in x for x in live[0]["placement"]["replicas"])
stops=[x for x in events if x["event"]=="gateway_clean_stopped"];assert len(stops)==2 and all(x["exit_code"]in[0,-15]for x in stops)
fault=json.loads((r/"response_owners.json.fault").read_text());assert not fault["open"]and all(not x["faulted"]for x in fault["groups"].values())
assert any(x["event"]=="cross_owner_conflict_rejected_before_RPC"for x in events)
assert not any(x["event"].startswith("owned_native_background_cleanup")for x in events)
stdout=(r/"native_full_api.stdout").read_text();ack=[json.loads(x)for x in stdout.splitlines()if x.startswith('{"event":')];assert ack[0]["event"]=="task_acl_init"and ack[-1]["event"]=="task_acl_finalize"and ack[0]["returncode"]==ack[-1]["returncode"]==0
physical={};apiowners=json.loads((r/"adopted_model_identities.json").read_text());members=json.loads((r/"native_member_identities.json").read_text())
check="""import pathlib,json,sys,subprocess,re
a=json.load(sys.stdin);o=a['owner'];boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for x in a['targets']+[dict(pid=o['pid'],identity=o['identity'])]:
 p=pathlib.Path('/proc/'+str(x['pid']));b=(p/'stat').read_text();v=b[b.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity']
assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);ids={int(v)for v in re.findall(r'^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker',raw,re.M)};assert ids=={x['pid']for x in a['targets']}
policy=json.loads(pathlib.Path(a['policy_path']).read_text());assert policy==a['policy'];print(json.dumps(dict(API_same=True,NPU16_same=True,npu_pids=sorted(ids),policy=policy)))
"""
policy=dict(schema_version=1,cohort_id="GLM-COHORT-0089",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
def metric(p):
 out={}
 for l in p.read_text().splitlines():
  if not l or l.startswith("#"):continue
  name=l.split("{")[0].split()[0]
  if any(x in name for x in["generation_tokens_total","prompt_tokens_total","request_success_total","num_requests_running","num_requests_waiting","prefix_cache_hits_total","num_preemptions_total","kv_cache_usage_perc"]):out[name]=out.get(name,0)+float(l.rsplit(" ",1)[1])
 return out
counters={};http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for key,o in apiowners.items():
 policy=dict(policy,cohort_id="GLM-COHORT-0089"if o["host"]=="166"else"GLM-COHORT-0096",serial=3 if o["host"]=="166"else 1)
 targets=[v for v in members[key]["owned_targets"]if v["pid"]in members[key]["npu_worker_pids"]]
 args=["python3","-c",check]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,targets=targets,policy=policy,policy_path=str(Path(o["argv"][1]).parent/"issue_budget_policy.json"))).encode(),capture_output=True,timeout=90);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();physical[key]=json.loads(z.stdout)
 with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as res:raw=res.read();assert res.status==200
 f=j/(key+".terminal.metrics");f.write_bytes(raw);before=metric(r/("initial_0_"+key+".metrics"));after=metric(f);delta={k:after.get(k,0)-v for k,v in before.items()if k.endswith("_total")}
 assert all(after.get(k,0)==0 for k in["vllm:num_requests_running","vllm:num_requests_waiting"])and delta.get("vllm:num_preemptions_total",0)==0
 counters[key]=dict(delta=delta,idle=True,raw=ref(f))
ports=subprocess.check_output(["ss","-ltnp"],text=True);assert not re.search(r":8002\s",ports)
budgets=[x["output_budget"]for x in trace if x["event"]=="lease_acquired"and x.get("path")=="/v1/responses"];assert 32 in budgets and 8192 in budgets

PDevents=[x for x in trace if x["event"].startswith("pd_")]
assert len([x for x in PDevents if x["event"]=="pd_helper_started"])==len([x for x in PDevents if x["event"]=="pd_helper_completed"])==len([x for x in PDevents if x["event"]=="pd_decoder_dispatch"])==3
assert not any(x["event"]in["pd_helper_failed","pd_helper_cancelled"]for x in PDevents)
pd_contracts=[]
for label in ["pd_json","pd_sse","pd_chat"]:
 raw=(r/(label+".body")).read_bytes();original=json.loads(raw);h=hashlib.sha256(raw).hexdigest();start=[x for x in PDevents if x["event"]=="pd_helper_started"and x["original_body_sha256"]==h];assert len(start)==1
 op=start[0]["operation"];helper=next(x for x in PDevents if x["event"]=="pd_helper_completed"and x["operation"]==op);dispatch=next(x for x in PDevents if x["event"]=="pd_decoder_dispatch"and x["operation"]==op);kv=helper["kv_transfer_params"]
 assert start[0]["producer"]=="http://172.16.10.166:9081"and start[0]["decoder"]=="http://172.16.10.167:9900"
 assert kv["do_remote_prefill"]is True and kv["remote_host"]=="172.16.10.166"and kv["remote_port"]==28000 and kv["remote_dcp_size"]==16 and kv["remote_pcp_size"]==1 and kv["remote_engine_id"]=="GLM89-P-b3d34735cac448879967aba87d7ce5a7"
 assert helper["internal_output_tokens"]==1 and helper["usage"]["output_tokens"if label!="pd_chat"else"completion_tokens"]==1
 native=dict(original,kv_transfer_params=kv);assert dispatch["native_body_sha256"]==hashlib.sha256(json.dumps(native,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
 if label!="pd_chat":
  v=json.loads((r/(label+"_retrieve.wire")).read_text());assert v["usage"]["output_tokens"]==32 and v["usage"]["input_tokens"]==11285 and v["usage"]["input_tokens_details"]["cached_tokens"]==11285 and helper["usage"]["input_tokens"]==11285 and owners[v["id"]]["replica"]=="D1"
 else:
  wire=(r/(label+".wire")).read_bytes();obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True);obs.feed(wire);v=obs.contract();assert v["usage"]==dict(prompt_tokens=81932,completion_tokens=64,total_tokens=81996)and helper["usage"]["prompt_tokens"]==81932
  frames=[]
  for frame in wire.replace(b"\r\n",b"\n").split(b"\n\n"):
   data=b"\n".join(x[5:].removeprefix(b" ")for x in frame.splitlines()if x.startswith(b"data:"))
   if data and data!=b"[DONE]":frames.append(json.loads(data))
  assert sum(len(c.get("token_ids")or[])for x in frames for c in x.get("choices",[]))==64
  assert any(len(x.get("prompt_token_ids")or[])==81932 for x in frames)
 refs={}
 for key,obj in [("original_body",start[0]["original_body_artifact"]),("helper_body",start[0]["helper_body_artifact"]),("native_D_body",dispatch["native_body_artifact"])]:
  f=Path(obj["path"]);assert "audit_error"not in obj and ref(f)==obj and f.stat().st_mode&0o777==0o600;refs[key]=obj
 wire_event=next(x for x in PDevents if x["event"]=="pd_helper_wire"and x["operation"]==op);obj=wire_event["artifact"];f=Path(obj["path"]);assert wire_event["status"]==200 and ref(f)==obj and f.stat().st_mode&0o777==0o600;refs["helper_wire"]=obj
 original_raw=Path(refs["original_body"]["path"]).read_bytes();assert original_raw==raw
 hb=json.loads(Path(refs["helper_body"]["path"]).read_bytes());hv=json.loads(f.read_bytes());nd=json.loads(Path(refs["native_D_body"]["path"]).read_bytes())
 assert hv["kv_transfer_params"]==kv and nd==native and hv["usage"]==helper["usage"]and hb["stream"]is False and hb["ignore_eos"]is True and hb["kv_transfer_params"]["do_remote_decode"]is True
 assert hb["cache_salt"]==original["cache_salt"]and hb["model"]==original["model"]and hb.get("seed")==original.get("seed")and hb.get("temperature")==original.get("temperature")
 assert kv["remote_request_id"]and kv["remote_ptp_size"]==16
 if label!="pd_chat":
  assert hb["input"]==original["input"]and hb["max_output_tokens"]==1 and hb["store"]is False and hb["background"]is False and hb["request_id"].startswith("resp_glm_pdhelper_")
 else:
  assert hb["messages"]==original["messages"]and hb["max_tokens"]==hb["min_tokens"]==1 and hb.get("return_token_ids")is True
  assert len(hv["choices"][0]["token_ids"])==1 and len(hv["prompt_token_ids"])==81932
 pd_contracts.append(dict(label=label,helper=start[0],completed_helper=helper,D_dispatch=dispatch,original_body=ref(r/(label+".body")),native_wire=ref(r/(label+".wire")),raw_PD_contracts=refs))
native_effective=summary["effective_public_output_tokens"];assert sum(x["delta"]["vllm:generation_tokens_total"]for x in counters.values())>=native_effective+3
assert sum(x["delta"]["vllm:request_success_total"]for x in counters.values())==22
assert counters["D1"]["delta"]["vllm:external_prefix_cache_hits_total"]==81932+2*11285
assert not any(k.startswith("resp_glm_pdhelper_")for k in owners)

out=dict(native_PD_contracts=pd_contracts,internal_helper_commits=3,total_minimum_native_commits=summary["effective_public_output_tokens"]+3,uncredited_native_output_tokens=sum(x["delta"]["vllm:generation_tokens_total"]for x in counters.values())-summary["effective_public_output_tokens"]-3,native_completed_metric_requests=22,cancelled_public_request_completed_credit=0,helper_raw_wire_complete=True,placement_policy="active_count",native_Responses_live_budget_values=sorted(set(budgets)),native_GLM_semantics_prevalidated="96threeDexact+coherent96coldpilot107/89priorP6exact",at=utc(),functional_acceptance=True,verdict="INCONCLUSIVE",effective_public_output_tokens=summary["effective_public_output_tokens"],fixed_completed_public_requests=11,Responses_output_tokens=384,tool_cases=8,tool_outputs=tooloutputs,tools_each_choice_both_native_API=True,HTTP_contracts=rr,tools=ts,background_live_zero_HTTP_leases=live,clean_gateway_stops=stops,physical_same=physical,native_counters=counters,source_pins=len(spec["stages"][0]["sources"]),gateway_port8002_free=True,model_operations=0,limits=["Finite fullgateway/nativeSTORE2domains functional E2E, not fullframework completion/formalSLO/stablecapacity/KEEP","Native completedmetric22 equals19publiccompleted+3helper; cancelledrequestnative statecancelled hasno successmetricdelta/zero publiccredit; native outputresidual retainedas uncredited cost, notTPS; Cancelled nativejob may generate tokens but credits zero; GET/replay/retrieve never extra outputcredit","Native API state is not replicated; APIphysicalrestart changes owner epoch/unavailable; concurrentnative duplicate-ID acceptance not GPU tested","Prefixcaches deliberately notcold performance workload; nativecounters include cancelled/background tokens; no speculative internalTPS substitution"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run97 2nativeSTOREdomains/P89D96nativeSP/V9PDv2/fullAPI/8toolcases/8fixedcommits384outputs/nativebackgroundcancelzero/cleanproxyrestart validated; originalAPI2NPU32/idle/policy preserved; noKEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="originalwires/ownerIDs/bodySHA/leaseonce/nativeSDK0/originalAPI2NPU32/counters")],unknowns=out["limits"]+["NativeD SPonly/configclass capture48; noisolatedgain/PDcapacity/cancel-between-Pexport-Ddispatch proof"],decision_request=None,next_check_at=None));print(json.dumps(dict(outputs=out["effective_public_output_tokens"],tooloutputs=tooloutputs,HTTP=len(rr),native_counters=counters)))

