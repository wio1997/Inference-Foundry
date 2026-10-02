from pathlib import Path
import json,sys,hashlib,re,subprocess,shlex,time,math
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0090";state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
spec=json.loads((r/"controller_spec.json").read_text())
for x in spec["stages"][0]["sources"]:assert ref(Path(x["path"]))["sha256"]==x["sha256"]and Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()
for name in["prepare","pd"]:
 v=json.loads((r/(name+".phase.json")).read_text());assert v["status"]=="succeeded"and v["exit_code"]==0 and not v["timed_out"]
a=json.loads((r/"pd_summary.json").read_text());assert a["functional_acceptance"]and a["effective_public_output_tokens"]==68 and a["internal_helper_commits"]==2
rows={x["name"]:x for x in a["requests"]};assert len(rows)==8 and all(x["completed"]for x in rows.values())
parsed={}
for name,x in rows.items():
 f=Path(x["wire"]["path"]);assert ref(f)==x["wire"];raw=f.read_bytes()
 if name=="chat_public":
  obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True)
  for off in range(0,len(raw),733):obs.feed(raw[off:off+733])
  v=obs.contract();assert v==x["response"]and v["done"]and not v["native_error"]and not v["unknown"]and v["usage"]==dict(prompt_tokens=81932,total_tokens=81996,completion_tokens=64)
  frames=[]
  for block in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
   ds=b"\n".join(z[5:].removeprefix(b" ")for z in block.splitlines()if z.startswith(b"data:"))
   if ds and ds!=b"[DONE]":frames.append(json.loads(ds))
  ids=[t for v in frames for c in v.get("choices",[])for t in c.get("token_ids")or[]];prompts=[v["prompt_token_ids"]for v in frames if v.get("prompt_token_ids")is not None];assert len(ids)==64 and prompts and len(prompts[0])==81932
 else:v=json.loads(raw);assert v==x["response"]
 parsed[name]=v
 if "body"in x:assert ref(Path(x["body"]["path"]))==x["body"]
 if name in["responses_wrong_store","helper_not_stored"]:assert x["http_status"]==404
 else:assert x["http_status"]==200
P=parsed["chat_helper"];assert P["usage"]["prompt_tokens"]==81932 and P["usage"]["completion_tokens"]==1 and len(P["prompt_token_ids"])==81932 and len(P["choices"][0]["token_ids"])==1
metadata={}
for kind,pn,dn in[("chat","chat_helper","chat_public"),("responses","responses_helper","responses_public")]:
 pv=parsed[pn];kv=pv["kv_transfer_params"];inp=pv["usage"]["prompt_tokens"if kind=="chat"else"input_tokens"]
 assert kv==json.loads((r/(kind+".metadata.json")).read_text())and kv["do_remote_prefill"]is True and kv["remote_dcp_size"]==16 and kv["remote_pcp_size"]==1 and kv["remote_host"]=="172.16.10.166"and int(kv["remote_port"])==28000
 assert kv["remote_engine_id"].startswith("GLM89-P-")and kv["remote_block_size"]==128 and kv["num_prompt_blocks"]==math.ceil(inp/128)
 assert len(kv["remote_block_ids"])==1 and len(kv["remote_block_ids"][0])==math.ceil(kv["num_prompt_blocks"]/16)
 hp=json.loads(Path(rows[pn]["body"]["path"]).read_text());dp=json.loads(Path(rows[dn]["body"]["path"]).read_text());assert dp["kv_transfer_params"]==kv
 for key in["model","cache_salt","temperature","seed","messages"if kind=="chat"else"input"]:assert hp[key]==dp[key]
 if kind=="responses":assert not hp["store"]and dp["store"]and hp["request_id"]!=dp["request_id"]and hp["max_output_tokens"]==1 and dp["max_output_tokens"]==32
 else:assert hp["max_tokens"]==1 and hp["min_tokens"]==1 and dp["max_tokens"]==64
 metadata[kind]=kv
def text(v):return "".join(y.get("text","")for x in v["output"]for y in x.get("content",[])if y.get("type")=="output_text")
for name in["responses_public","responses_previous"]:
 v=parsed[name];assert v["status"]=="completed"and text(v).strip()=="42"and v["usage"]["output_tokens"]==2 and v["usage"]["total_tokens"]==v["usage"]["input_tokens"]+2
assert parsed["responses_retrieve"]==parsed["responses_public"]
assert parsed["responses_helper"]["usage"]["input_tokens"]==parsed["responses_public"]["usage"]["input_tokens"]==11283 and parsed["responses_helper"]["usage"]["output_tokens"]==1
assert parsed["responses_public"]["usage"]["input_tokens_details"]["cached_tokens"]==11283
previous=json.loads(Path(rows["responses_previous"]["body"]["path"]).read_text());assert previous["previous_response_id"]==parsed["responses_public"]["id"]and not previous.get("kv_transfer_params")
def metrics(f):
 out={}
 for line in f.read_text().splitlines():
  if not line or line.startswith("#"):continue
  name=line.split("{")[0].split()[0]
  if any(s in name for s in["prompt_tokens_total","generation_tokens_total","request_success_total","prefix_cache_queries_total","prefix_cache_hits_total","num_preemptions_total","num_requests_running","num_requests_waiting","kv_cache_usage_perc"]):out[name]=out.get(name,0)+float(line.rsplit(" ",1)[1])
 return out
counters={}
for key in["D0","D1"]:
 baseline=p/"runs/GLM-RUN-0089"/(key+"_after_"+key+".metrics");b=metrics(baseline);e=metrics(r/("final_"+key+".metrics"));d={n:e.get(n,0)-v for n,v in b.items()if n.endswith("_total")}
 assert e["vllm:num_requests_running"]==e["vllm:num_requests_waiting"]==0 and d.get("vllm:num_preemptions_total",0)==0
 if key=="D0":assert d["vllm:prompt_tokens_total"]==93215 and d["vllm:generation_tokens_total"]==2 and d["vllm:request_success_total"]==2
 else:assert d["vllm:prompt_tokens_total"]==93215+parsed["responses_previous"]["usage"]["input_tokens"]and d["vllm:generation_tokens_total"]==68 and d["vllm:request_success_total"]==3 and d.get("vllm:external_prefix_cache_hits_total",0)>0
 counters[key]=dict(delta=d,prior89_terminal_baseline=b,baseline_source=ref(baseline),final=e,initial90_overwritten_by_terminal_epoch_check=True)
 for label in ["chat_helper_done","chat_done","responses_helper_done","responses_done"]:
  step=metrics(r/(label+"_"+key+".metrics"));assert step["vllm:request_success_total"]<=e["vllm:request_success_total"] and step["vllm:generation_tokens_total"]<=e["vllm:generation_tokens_total"]
 assert metrics(r/("initial_"+key+".metrics"))==e
owners=json.loads((r/"adopted_model_identities.json").read_text());members=json.loads((r/"native_member_identities.json").read_text());old=json.loads((p/"runs/GLM-RUN-0089/adopted_model_identities.json").read_text());assert owners==old
check="""import pathlib,json,sys,subprocess,re
a=json.load(sys.stdin);o=a['owner'];boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for x in a['targets']+[dict(pid=o['pid'],identity=o['identity'])]:
 f=pathlib.Path('/proc/'+str(x['pid']));s=(f/'stat').read_text();v=s[s.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity']
assert [x.decode()for x in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if x]==o['argv']
raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);ids={int(x)for x in re.findall(r'^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker',raw,re.M)};assert ids==set(x['pid']for x in a['targets']);print(json.dumps(dict(API_same=True,NPU16_same=True)))
"""
physical={};logs={}
for key,o in owners.items():
 args=["python3","-c",check]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 targets=[x for x in members[key]["owned_targets"]if x["pid"]in members[key]["npu_worker_pids"]];z=subprocess.run(args,input=json.dumps(dict(owner=o,targets=targets)).encode(),capture_output=True,timeout=90);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();physical[key]=json.loads(z.stdout)
 path="/data/tiankuan/wio/glm52-pd/deploy/logs/"+("P_run89_0.log"if key=="D0"else"D_run88_1.log");args=["cat",path]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=30);z.check_returncode();f=j/(key+".native.log");f.write_bytes(z.stdout);ls=z.stdout.decode(errors="replace").splitlines();assert not any("RuntimeError:"in l or"OutOfMemoryError"in l or"encountered exception"in l for l in ls)
 logs[key]=dict(raw=ref(f),PD_lines=[x[:1800]for x in ls if any(t in x.lower()for t in["kv send","kv recv","kv receiv","recv kv","transfer complete","receive kv","received kv","sending kv","remote kv"] )][-45:])
out=dict(at=utc(),verdict="INCONCLUSIVE",functional_acceptance=True,effective_public_output_tokens=68,internal_P_helper_commits=2,total_native_commits=70,source_pins=len(spec["stages"][0]["sources"]),true_PD_Chat_cold81932_64=True,true_PD_native_Responses_11283_exact42=True,Responses_get_equal_previous42_wrongdomain404_helperstoreFalse404=True,physical_same=physical,counters=counters,metadata=metadata,native_logs=logs,requests=[{k:x[k]for k in x if k!="response"}for x in a["requests"]],summary_source=ref(r/"pd_summary.json"),limits=["90initialmetrics overwritten by finalepoch_check.verify(initial); countersuseprior89terminalsameepoch baseline and90intermediate snapshots, notoriginal90initial; DirectnativeP89/D88 functionaldiagnostic only; notpublicgateway/fulltypedSSE/background/APIduplicateID/state replication/SLA/stablecapacity/KEEP","Cold P81932helper58.239s +D3.435s / finite61.805sE2E; no improvement vs89Dlocal60.366 isolatedTTFT/62.732wallassumed; workloadpublicPD vslocal differs","Responses11,283Phelper1→Dnativecached11,283/exact42/output2; originalinputbody/nativekvfield/counters/geometry actual; previousstateDlocalcontinuation notP reconstruction","P/D model/operator/native source unchanged; pendingKVfree/sendcompletion/nativepressurebandwidth bounds requireselectedevidence, logs/CPUhint notGPUtimebound"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="90 truePD nativeChat81932/64+Responses11283/exact42/store/retrieve/previous validated;68effective+2internal=70native; API2NPU32sameidle/nocapacityKEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="audit",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualraw/tokenIDs/nativeKV/counters/input/geometry/state/epochs")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(valid=True,effective=68,helper=2,bytes=len(b))))
