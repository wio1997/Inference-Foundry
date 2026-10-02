from pathlib import Path
import json,hashlib,re,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import utc,atomic_json
j=Path(__file__).parent;p=j.parents[1];rows=[]
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def metrics(f):
 a={}
 for l in f.read_text().splitlines():
  if not l or l.startswith("#"):continue
  name=l.split("{")[0].split()[0]
  if name.startswith("vllm:spec_decode_num_")and name.endswith("_total")or name in["vllm:iteration_tokens_total_count","vllm:iteration_tokens_total_sum","vllm:inter_token_latency_seconds_count","vllm:inter_token_latency_seconds_sum","vllm:generation_tokens_total","vllm:request_success_total"]:
   key=name
   if "per_pos" in name:key+=":"+re.search(r'position="([0-9]+)"',l).group(1)
   a[key]=a.get(key,0)+float(l.rsplit(" ",1)[1])
 return a
for n,key,depth in[(118,"D1",3),(120,"D1",5)]:
 r=p/"runs"/("GLM-RUN-%04d"%n);before=r/("before_0_"+key+".metrics");files=list(r.glob("after_*_"+key+".metrics"));after=max(files,key=lambda f:int(f.name.split("_")[1]));a=metrics(before);b=metrics(after);d={k:b[k]-v for k,v in a.items()};drafts=d["vllm:spec_decode_num_drafts_total"];drafttokens=d["vllm:spec_decode_num_draft_tokens_total"];accepted=d["vllm:spec_decode_num_accepted_tokens_total"]
 assert drafts>0 and drafttokens==depth*drafts and accepted<=drafttokens and d["vllm:generation_tokens_total"]==7552 and d["vllm:request_success_total"]==7
 positions=[d["vllm:spec_decode_num_accepted_tokens_per_pos_total:"+str(k)]for k in range(depth)];assert sum(positions)==accepted and all(positions[k]<=positions[k-1]for k in range(1,depth))
 summary=json.loads((r/"dynamic_summary.json").read_text());reqs=[x for x in summary["requests"]if x["input_kind"]=="short"]
 rows.append(dict(run=r.name,key=key,depth=depth,metrics_before=ref(before),metrics_after=ref(after),delta=d,accepted_per_request_draft=accepted/drafts,draft_acceptance=accepted/drafttokens,accepted_positions_per_request_draft=[x/drafts for x in positions],accepted_plus_bonus_proxy=1+accepted/drafts,public_output_per_request_draft=7552/drafts,native_inter_output_event_mean_s=d["vllm:inter_token_latency_seconds_sum"]/d["vllm:inter_token_latency_seconds_count"],observed_mean_output_s=[x["mean_after_first_output_s"]for x in reqs],GPU_engine_iterations_unknown=True))
sequences=[];p118=p/"runs/GLM-RUN-0118";p120=p/"runs/GLM-RUN-0120"
old=json.loads((p118/"dynamic_summary.json").read_text());new=json.loads((p120/"dynamic_summary.json").read_text())
def tokens(row):
 raw=Path(row["wire"]["path"]).read_bytes();ids=[]
 for part in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
  data=b"\n".join(v[5:].removeprefix(b" ")for v in part.splitlines()if v.startswith(b"data:"))
  if not data or data==b"[DONE]":continue
  obj=json.loads(data)
  for ch in obj.get("choices",[]):ids.extend(ch.get("token_ids")or[])
 assert len(ids)==row["usage"]["completion_tokens"]and all(type(x)is int for x in ids)
 return ids
for left,right in zip(old["requests"],new["requests"]):
 assert left["id"]==right["id"]and left["usage"]==right["usage"]
 a=tokens(left);b=tokens(right);common=next((i for i,(x,y)in enumerate(zip(a,b))if x!=y),len(a))
 sequences.append(dict(id=left["id"],outputs=len(a),identical_native_token_ids=a==b,common_prefix_tokens=common,old_token_sha256=hashlib.sha256(json.dumps(a,separators=(",",":")).encode()).hexdigest(),new_token_sha256=hashlib.sha256(json.dumps(b,separators=(",",":")).encode()).hexdigest(),old_wall_s=left["wall_s"],new_wall_s=right["wall_s"],wall_ratio=right["wall_s"]/left["wall_s"],old_mean_output_s=left["mean_after_first_output_s"],new_mean_output_s=right["mean_after_first_output_s"]))
out=dict(at=utc(),valid=True,rows=rows,sequences=sequences,inference_requests=0,models_started=0,model_signals=0,limits=["Native num_drafts counts request draft observations, not physical worker rounds or engine batches; iteration_tokens_count includes scheduled input/spec work, not accepted publictoken count","Inter-token latency measures native output events including MTP chunks, not unbatched per-token time; accepted_plus_bonus_proxy not exact public count due first/last/cancel accounting","118 and120 samePP2TP8DCP8/AllGather andsevenrequests exceptcoldsalts; K3Graph32 versusK5Graph48/newDnativeepoch jointlychanged, no stablecapacity/hardwarebound","K1/K5 costs cannot be predicted by acceptance alone; frozen105 heavyprofile Graph/MTPparent evidence only conditional, no hardwareupperbound"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Frozen118D3/120D5 native MTP counters and request-local acceptance units reduced; noGPUrequests/no physicalround substitution",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="frozenmetrics/sourceunits/7552outputs/7completed/requestdraftcounts")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(rows=[{k:x[k]for k in["run","depth","delta","accepted_per_request_draft","draft_acceptance","accepted_positions_per_request_draft","native_inter_output_event_mean_s"]}for x in rows])))

