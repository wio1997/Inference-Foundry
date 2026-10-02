from pathlib import Path
import json,sys,hashlib,subprocess,shlex,re,time,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0064"
for _ in range(160):
 s=json.loads((r/"state.json").read_text())
 if s["status"]!="running":break
 time.sleep(3)
assert s["status"]!="running"and not same_process(s["owner"])
pins=json.loads((r/"controller_spec.json").read_text())["stages"][0]["sources"]
for p in pins:assert Path(p["path"]).read_bytes()==Path(p["snapshot"]).read_bytes()and hashlib.sha256(Path(p["path"]).read_bytes()).hexdigest()==p["sha256"]
owners=json.loads((r/"adopted_model_identities.json").read_text());physical={}
for node in["166","167"]:
 a=json.loads((r/("preflight_"+node+".json")).read_text())[0];assert a["signals"]==0 and not a["P_API_active"]and a["npu_P"]==0 and a["npu_D"]==16
 targets=a["retained_D_targets"];o=owners["D"+str(0 if node=="166"else 1)]
 code="""import pathlib,json,sys
a=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for x in a['targets']:
 p=pathlib.Path('/proc/'+str(x['pid']));v=(p/'stat').read_text();f=v[v.rfind(')')+2:].split();assert f[0]not in['Z','X'] and dict(boot_id=boot,start_ticks=f[19])==x['identity']
o=a['owner'];argv=[v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v];assert argv==o['argv']
env={v.decode().split('=',1)[0]:v.decode().split('=',1)[1]for v in pathlib.Path('/proc/'+str(o['pid'])+'/environ').read_bytes().split(bytes([0]))if v and b'=' in v};print(json.dumps(dict(members=len(a['targets']),env={k:env.get(k)for k in ['HCCL_BUFFSIZE','VLLM_ENABLE_RESPONSES_API_STORE']})))
"""
 args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(targets=targets,owner=o)).encode(),capture_output=True,timeout=30);(j/(node+".owner.stdout")).write_bytes(z.stdout);(j/(node+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();physical[node]=json.loads(z.stdout)
assert sum(x["members"]for x in physical.values())==647
ledger=json.loads((r/"attempts.json").read_text());rows=[];http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
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
 name=row["name"];key=row["owner"];o=owners[key];b=(r/(name+".wire")).read_bytes();obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True)
 for n in range(0,len(b),4093):obs.feed(b[n:n+4093])
 c=obs.contract();assert c["done"]and not c["unknown"]and not c["native_error"]and c["finish_reasons"]=={"0":"length"}and c["usage"]==dict(prompt_tokens=81932,completion_tokens=64,total_tokens=81996)
 frames=[]
 for part in b.replace(b"\r\n",b"\n").split(b"\n\n"):
  d=b"\n".join(l[5:].removeprefix(b" ")for l in part.splitlines()if l.startswith(b"data:"))
  if d and d!=b"[DONE]":frames.append(json.loads(d))
 ps=[x["prompt_token_ids"]for x in frames if isinstance(x.get("prompt_token_ids"),list)];ids=[t for x in frames for ch in x.get("choices",[])for t in ch.get("token_ids")or[]]
 assert len(ps)==1 and len(ps[0])==81932 and len(ids)==64 and all(type(v)is int for v in ps[0]+ids)
 body=json.loads((r/(name+".body.json")).read_text());assert body["cache_salt"].startswith("GLM-RUN-0064-")and not body.get("kv_transfer_params")
 before=metrics((r/(key+"_before_"+key+".metrics")).read_bytes());after=None;delta={}
 for _ in range(15):
  with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as response:mb=response.read()
  after=metrics(mb);delta={k:after.get(k,0)-v for k,v in before.items()if k.endswith("_total")}
  if delta.get("vllm:prompt_tokens_total",0)>=81932 and delta.get("vllm:generation_tokens_total",0)>=64:break
  time.sleep(3)
 (j/(key+".terminal.metrics")).write_bytes(mb)
 cold=delta.get("vllm:prefix_cache_queries_total",0)>=81932 and delta.get("vllm:prefix_cache_hits_total",0)==0 and delta.get("vllm:external_prefix_cache_hits_total",0)==0
 assert all(after.get(k,0)==0 for k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"])
 rows.append(dict(name=name,owner=key,TTFT_s=row["ttft_s"],wall_s=row["wall_s"],usage=c["usage"],prompt_ids_count=len(ps[0]),output_ids_count=len(ids),native_committed_token_ids=ids,wire=row["wire"],cache_salt=body["cache_salt"],native_cold_counters_confirmed=cold,counter_deltas=delta,native_idle=True))
ok=s["status"]=="completed"and len(rows)==2;out=dict(at=utc(),run_status=s["status"],source_pins=len(pins),physical_D=physical,new_models=0,signals=0,requests=rows,measurement_valid=ok,functional_acceptance=ok,verdict="INCONCLUSIVE",effective_public_output_tokens=sum(x["usage"]["completion_tokens"]for x in rows),limits=["Two finite directnativeD-only longprefill cold-salt cases, no publicgateway/PD/capacity/KEEP","P61stop/D56retained changesmemory/co-resident workload vs61; no isolatedHCCL/chunk/MTPeffect","ReportedTTFT includes fullprompt-ID frame overhead; sameobserver workload required for comparison","Nativecoldstatsexport canlag; coldconfirmedflag explicit, nofalsecacheclaim"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out)
m=json.loads((r/"manifest.json").read_text());m.update(status=s["status"],valid=ok,verdict="INCONCLUSIVE",results=out,reduction=str(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\n"+json.dumps(dict(state=s["status"],requests=len(rows),outputs=out["effective_public_output_tokens"],TTFT_s=[x["TTFT_s"]for x in rows],wall_s=[x["wall_s"]for x in rows],cold=[x["native_cold_counters_confirmed"]for x in rows],D56_members=647,new_models=0,signals=0))+"\n\nFinite nativeD-only functionality, no publicgateway/PD/stablecapacity/KEEP.\n")
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run64 D-only raw nativeSSE/tokenIDs/coldcounters/physicalD647 audit; actual "+str(len(rows))+" completed cases",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualnativewire/counters/647Didentities")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(out))

