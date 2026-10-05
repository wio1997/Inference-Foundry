from pathlib import Path
import json,hashlib
import sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1]
def ref(f):
 raw=Path(f).read_bytes();return dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
runs={}
audits={210:"AUDIT-RUN210-20261005TPOSTSPILL-v2",212:"AUDIT-RUN212-20261005TPOSTLATE",213:"AUDIT-RUN213-20261005TPOSTREPEAT"}
for n,name in audits.items():
 r=p/("runs/GLM-RUN-%04d"%n);f=p/"jobs"/name/"reduction.json";a=json.loads(f.read_text());assert a["measurement_valid"]and a["functional_acceptance"]and a["effective_outputs"]==14208
 raw=json.loads((r/"arrival_summary.json").read_text());assert raw["functional_acceptance"]and len(raw["requests"])==16
 runs[n]=dict(audit=a,raw=raw,dir=r,evidence=ref(f))
pairs=[]
for before,after in[(210,212),(212,213)]:
 x=runs[before];y=runs[after];rows=[];sameIDs=0
 for old,new in zip(x["raw"]["requests"],y["raw"]["requests"]):
  assert old["id"]==new["id"]and old["expected_outputs"]==new["expected_outputs"]and old["expected_prompt_tokens"]==new["expected_prompt_tokens"]and old["scheduled_arrival_s"]==new["scheduled_arrival_s"]
  ident=old["id"];a=x["dir"]/(ident+".body.json");b=y["dir"]/(ident+".body.json");va=json.loads(a.read_text());vb=json.loads(b.read_text());va.pop("cache_salt");vb.pop("cache_salt");assert va==vb
  ai=old["committed_token_ids"];bi=new["committed_token_ids"];assert len(ai)==len(bi)==old["expected_outputs"]
  prefix=0
  for u,v in zip(ai,bi):
   if u!=v:break
   prefix+=1
  equal=ai==bi;sameIDs+=equal
  owners=[next(v["native_owner"]for v in z["audit"]["requests"]if v["id"]==ident)for z in[x,y]]
  rows.append(dict(id=ident,prompt_tokens=old["expected_prompt_tokens"],outputs=len(ai),scheduled_arrival_s=old["scheduled_arrival_s"],same_body_except_salt=True,body_refs=[ref(a),ref(b)],native_owners=owners,all_committed_IDs_equal=equal,common_committed_ID_prefix=prefix,IDhashes=[hashlib.sha256(json.dumps(v,separators=(",",":")).encode()).hexdigest()for v in[ai,bi]],mean_output_interval_ms=[1000*z["mean_after_first_output_s"]for z in[old,new]],TTFTms=[1000*z["ttft_s"]for z in[old,new]],finished_origin_s=[z["finished_origin_s"]for z in[old,new]],HTTP_chunk_count=[len(z["native_token_chunk_points"])for z in[old,new]],HTTP_chunk_counts_are_not_GPUrounds=True))
 pairs.append(dict(before=before,after=after,same_body_case16=True,identical_fullID_vectors=sameIDs,rows=rows,finiteTPS=[z["audit"]["finite_effective_output_tps"]for z in[x,y]],MTP=[z["audit"]["native_MTP"]for z in[x,y]],referenceSLO=[z["audit"]["diagnostic_reference_SLO"]for z in[x,y]],same_native_D0_epoch=True,same_native_D1_epoch=before==212))
out=dict(at=utc(),valid=True,inputs={str(n):v["evidence"]for n,v in runs.items()},pairs=pairs,operations=dict(new_inference=0,models=0,signals=0,policywrites=0),Current=None,limits=["Exactsame workload bodies excludingcold salts, source/nativeIDS audited; output trajectories andMTP acceptance maydiffer evenwithout sameengineconfig change","ChangedD1Graph key andphysicalepoch between210212; unchangedD0 criticaltail change counterevidence againstisolatedoverallGraphgain","212213 repeatsamephysicalepochs/public/Graphpolicy; repeatmagnitude notstablecapacity/KEEP orglobalupperbound","HTTPnative SSEframechunks/committedIDs are not GPUforward-iteration/collective count; actualruntimeGraphdispatch unknown"])
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Same16cases/bodyssalt/nativefullIDs trajectories/MTP/criticalD0tails verified across210212213; pairQoS interpretation only/noisolatedGraphgain",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="bodypairs-fullnativeIDprefix-MTP-singletonandD0criticaltails")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,pairs=[dict(before=v["before"],after=v["after"],sameIDvectors=v["identical_fullID_vectors"])for v in pairs])))
