from pathlib import Path
import json,sys,hashlib,subprocess,shlex,re,time,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0065"
for _ in range(500):
 s=json.loads((r/"state.json").read_text())
 if s["status"]!="running":break
 time.sleep(3)
assert s["status"]!="running"and not same_process(s["owner"])
pins=json.loads((r/"controller_spec.json").read_text())["stages"][0]["sources"]
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]
planned=json.loads((r/"planned_launch.json").read_text());owners=json.loads((r/"startup_model_identities.json").read_text())if(r/"startup_model_identities.json").exists()else{};native={};members={}
def cmd(node,args,input=None):
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 return subprocess.run(args,input=input,capture_output=True,timeout=90)
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
for x in planned:
 node=x["node"];key="D"+str(x["rank"]);z=cmd(node,["cat",x["log"]]);f=j/(key+".native.log");f.write_bytes(z.stdout);z.check_returncode();ls=z.stdout.decode(errors="replace").splitlines()
 native[key]=dict(raw=ref(f),errors=[l for l in ls if any(v in l for v in["ERROR","OutOfMemoryError","Traceback"])][:30],SFAzero_receipts=len([l for l in ls if"GLM_UNUSED_SFA_WORKSPACE_SKIPPED "in l]),metadata_receipts=len([l for l in ls if"GLM_DP_METADATA_INSTALLED "in l]),graph=[l for l in ls if"Graph capturing finished"in l],KVpool=[l for l in ls if"GPU KV cache size"in l])
 o=owners.get(key)
 if not o:continue
 code="""import pathlib,json,subprocess,re,sys
o=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def proc(pid):
 try:
  p=pathlib.Path('/proc/'+str(pid));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();return dict(identity=dict(boot_id=boot,start_ticks=v[19]),state=v[0])
 except FileNotFoundError:return None
p=proc(o['pid']);alive=bool(p and p['identity']==o['identity']and p['state']not in['Z','X'])
if not alive:print(json.dumps(dict(API_alive=False)));raise SystemExit(0)
assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
t=subprocess.check_output(['docker','top','glm52-single','-eo','pid,ppid,stat,comm,args'],text=True);rows={}
for l in t.splitlines()[1:]:
 x=l.split(None,4)
 if len(x)==5:rows[int(x[0])]=(int(x[1]),x[2],x[3],x[4])
ids={o['pid']}
while True:
 more=ids|{pid for pid,x in rows.items()if x[0]in ids}
 if more==ids:break
 ids=more
targets=[dict(pid=pid,identity=proc(pid)['identity'],state=proc(pid)['state'],comm=rows[pid][2])for pid in sorted(ids)if proc(pid)and proc(pid)['state']not in['Z','X']]
unknown=[pid for pid,x in rows.items()if x[1][0]not in['Z','X']and('/bin/vllm serve 'in x[3]or'native_acl_lifecycle.py cli serve 'in x[3]or x[2].startswith('VLLM')or'bishengir-compile 'in x[3])and pid not in ids]
assert not unknown,unknown
raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);npus={int(v)for v in re.findall(r'^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker',raw,re.M)};assert len(npus)==16 and npus<=ids
e=dict(v.decode().split('=',1)for v in pathlib.Path('/proc/'+str(o['pid'])+'/environ').read_bytes().split(bytes([0]))if b'='in v)
print(json.dumps(dict(API_alive=True,API_owner=o,owned_targets=targets,npu_worker_pids=sorted(npus),npu_smi=raw,env={k:e.get(k)for k in ['HCCL_BUFFSIZE','HCCL_NPU_SOCKET_PORT_RANGE','HCCL_HOST_SOCKET_PORT_RANGE','VLLM_ENABLE_RESPONSES_API_STORE']})))
"""
 z=cmd(node,["python3","-c",code],json.dumps(o).encode());(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();members[key]=json.loads(z.stdout)
 if members[key]["API_alive"]:
  assert members[key]["env"]["HCCL_BUFFSIZE"]=="4096"and members[key]["env"]["VLLM_ENABLE_RESPONSES_API_STORE"]=="1"
ledger=json.loads((r/"attempts.json").read_text())if(r/"attempts.json").exists()else[];rows=[];http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def metrics(b):
 out={}
 for l in b.decode().splitlines():
  if not l or l.startswith("#"):continue
  name=l.split("{")[0].split()[0]
  if any(n in name for n in["prefix_cache_queries_total","prefix_cache_hits_total","prompt_tokens_total","generation_tokens_total","request_success_total","num_requests_running","num_requests_waiting","kv_cache_usage_perc","num_preemptions_total"]):
   try:out[name]=out.get(name,0)+float(l.rsplit(" ",1)[1])
   except ValueError:pass
 return out
for row in ledger:
 if not row["completed"]:continue
 key=row["owner"];o=owners[key];b=(r/(row["name"]+".wire")).read_bytes();obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True);obs.feed(b);c=obs.contract()
 assert c["done"]and not c["unknown"]and not c["native_error"]and c["finish_reasons"]=={"0":"length"}and c["usage"]==dict(prompt_tokens=81932,completion_tokens=64,total_tokens=81996)
 frames=[]
 for part in b.replace(b"\r\n",b"\n").split(b"\n\n"):
  d=b"\n".join(l[5:].removeprefix(b" ")for l in part.splitlines()if l.startswith(b"data:"))
  if d and d!=b"[DONE]":frames.append(json.loads(d))
 ps=[v["prompt_token_ids"]for v in frames if isinstance(v.get("prompt_token_ids"),list)];ids=[t for v in frames for ch in v.get("choices",[])for t in ch.get("token_ids")or[]];assert len(ps)==1 and len(ps[0])==81932 and len(ids)==64
 before=metrics((r/(key+"_before_"+key+".metrics")).read_bytes());delta={}
 for _ in range(15):
  with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as z:mb=z.read()
  after=metrics(mb);delta={k:after.get(k,0)-v for k,v in before.items()if k.endswith("_total")}
  if delta.get("vllm:prompt_tokens_total",0)>=81932 and delta.get("vllm:generation_tokens_total",0)>=64:break
  time.sleep(3)
 (j/(key+".terminal.metrics")).write_bytes(mb);assert all(after.get(k,0)==0 for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"])
 cold=delta.get("vllm:prefix_cache_queries_total",0)>=81932 and delta.get("vllm:prefix_cache_hits_total",0)==0 and delta.get("vllm:external_prefix_cache_hits_total",0)==0
 rows.append(dict(owner=key,name=row["name"],TTFT_s=row["ttft_s"],wall_s=row["wall_s"],usage=c["usage"],prompt_id_count=81932,committed_token_ids=ids,wire=row["wire"],cold_confirmed=cold,counter_deltas=delta,native_idle=True))
cleanup={}
for node in["166","167"]:
 f=r/("cleanup_"+node+".stdout");a=[json.loads(l)for l in f.read_text().splitlines()if l.startswith("{")];cleanup[node]=dict(raw=ref(f),sent=[v for v in a if v.get("event")=="signal_sent"],original_D56_inactive=a[-1].get("all_original_D_inactive"),signals_to_unknown=a[-1].get("signals_to_unknown"))
ok=s["status"]=="completed"and len(rows)==2 and len(members)==2 and all(v["API_alive"]for v in members.values());limits=["Finite nativeD65 diagnostic only; no publicgateway/PD/mixedload/stablecapacity/KEEP","NEW D65 APIs/physicalmembers; oldD56all647 was stopped. Frozenpilot copied D56-retained descriptive wording is stale; actualdeployment/33pins/physicalnewcohort authoritative","Batch/seq/KV/HCCL/Graph resourceallocation changedtogether, no isolatedbatch/kernel gain","Source/checkpoint/nativeoperators unchanged; no modelqualitycomparison; exacttokens/nativeusage actual"]
out=dict(at=utc(),run_status=s["status"],source_pins=len(pins),native=native,actual_D65_members=members,cleanup=cleanup,requests=rows,actual_attempts=len(ledger),measurement_valid=ok,functional_acceptance=ok,verdict="INCONCLUSIVE",effective_public_output_tokens=sum(v["usage"]["completion_tokens"]for v in rows),limits=limits)
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();brief=dict(out);brief["actual_D65_members"]={k:{f:v.get(f)for f in["API_alive","API_owner","npu_worker_pids","env"]}for k,v in members.items()};brief["full_reduction"]=ref(j/"reduction.json");atomic_json(r/"reduction_brief.json",brief)
atomic_json(r/"native_member_identities.json",members);m=json.loads((r/"manifest.json").read_text());m.update(status=s["status"],valid=ok,verdict="INCONCLUSIVE",results=brief,reduction=str(j/"reduction.json"));atomic_json(r/"manifest.json",m);(r/"summary.md").write_text("# "+r.name+"\n\n"+json.dumps(dict(state=s["status"],cases=len(rows),outputs=brief["effective_public_output_tokens"],TTFT=[v["TTFT_s"]for v in rows],cold=[v["cold_confirmed"]for v in rows],new_D65=True))+"\n\n"+". ".join(limits)+"\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run65 actualnativeD65/newphysicalmembership/33pins/cleanup/Graph/tokenIDs/cold audit; "+str(len(rows))+" completed cases",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="nativeD65 actualsource/owners/logs/wire/counters",**ref(j/"reduction.json"))],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(brief))

