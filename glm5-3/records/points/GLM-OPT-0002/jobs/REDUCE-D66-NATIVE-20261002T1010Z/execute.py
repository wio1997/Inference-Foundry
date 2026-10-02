from pathlib import Path
import json,sys,hashlib,subprocess,shlex,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0066";s=json.loads((r/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
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
owners=json.loads((r/"adopted_model_identities.json").read_text());old=json.loads((r.parent/"GLM-RUN-0065/native_member_identities.json").read_text());physical={}
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
out=dict(at=utc(),verdict="INCONCLUSIVE",measurement_valid=True,functional_acceptance=True,source_pins=len(spec["stages"][0]["sources"]),actual_new_requests=5,effective_public_output_tokens=768,elapsed_s=a["elapsed_s"],requests=rows,native_counter_deltas=deltas,physical=physical,new_models=0,signals=0,conclusion="Finite mixednative workload: D0firstshortTTFT~.489 thenmax8.31s nativecommit-chunk gap whileD1longprefill; laterD0shortTTFT~13s,D1short~37s. Correlatedtiming/cohort evidence supports further scheduling-budget experiment, no uniquecausal attribution/kernel-limit/capacity/KEEP.",limits=a["limits"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=out,reduction=str(j/"reduction.json"));atomic_json(r/"manifest.json",m);(r/"summary.md").write_text("# "+r.name+"\n\n"+json.dumps(dict(requests=5,outputs=768,TTFT=[v["ttft_s"]for v in rows],max_chunks=[v["max_native_chunk_gap_s"]for v in rows],resident_D65=True))+"\n\n"+out["conclusion"]+"\n")
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run66VALIDfinite5mixednative requests/768commits/source/32physical/idle/counters; schedulinginterference evidence, noKEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="nativewires/chunk-timing/arrival/counters/32NPUepochs")],unknowns=a["limits"],decision_request=None,next_check_at=None));print(json.dumps(out))

