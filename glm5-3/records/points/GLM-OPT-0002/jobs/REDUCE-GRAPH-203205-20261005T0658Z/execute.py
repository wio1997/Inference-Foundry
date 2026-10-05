from pathlib import Path
import json,hashlib,statistics
from datetime import datetime,timezone
j=Path(__file__).parent;p=j.parents[1]
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
runs={}
for n,sha in[(203,"5751c86478b5f2df7b1b3041d25307476749ab628fa0c9adbefdb66a0d94dfc2"),(205,"9a9fea3b7a3faab4507848513a111be9a03f9346348e8b99f39274f92bd62615")]:
 r=p/("runs/GLM-RUN-%04d"%n);f=p/("jobs/AUDIT-RUN"+str(n)+"-20261005TPOSTMIXED/reduction.json");assert ref(f)["sha256"]==sha
 a=json.loads(f.read_text());v=json.loads((r/"arrival_summary.json").read_text());plan=json.loads((r/"arrival_plan.json").read_text());assert a["measurement_valid"]and a["functional_acceptance"]and a["effective_outputs"]==10112
 runs[n]=dict(root=r,audit=a,audit_ref=ref(f),rows={x["id"]:x for x in v["requests"]},cases={x["id"]:x for x in plan["cases"]})
pairs=[]
for key,x in runs[203]["rows"].items():
 y=runs[205]["rows"][key];bodies=[]
 for n,row in[(203,x),(205,y)]:
  r=runs[n]["root"];body=json.loads((r/runs[n]["cases"][key]["body"]).read_bytes());assert body.pop("cache_salt")==r.name+"-"+key;bodies.append(body)
  assert ref(Path(row["wire"]["path"]))==row["wire"]and len(row["committed_token_ids"])==row["effective_public_output_credit"]
 assert bodies[0]==bodies[1]and x["usage"]==y["usage"]
 a=x["committed_token_ids"];b=y["committed_token_ids"];common=next((i for i,(m,n)in enumerate(zip(a,b))if m!=n),min(len(a),len(b)))
 owner=next(v["native_owner"]for v in runs[205]["audit"]["requests"]if v["id"]==key);assert owner==next(v["native_owner"]for v in runs[203]["audit"]["requests"]if v["id"]==key)
 evidence=[]
 for row in[x,y]:
  pts=row["native_token_chunk_points"];gaps=[b["elapsed_s"]-a["elapsed_s"]for a,b in zip(pts,pts[1:])]
  evidence.append(dict(native_committed_output_ID_sha256=hashlib.sha256(json.dumps(row["committed_token_ids"],separators=(",",":")).encode()).hexdigest(),events=len(pts),native_HTTP_gap_mean_s=statistics.mean(gaps)if gaps else None,TTFT_s=row["ttft_s"],wall_s=row["wall_s"],mean_after_first_output_ms=row["mean_after_first_output_s"]*1000,wire=row["wire"]))
 pairs.append(dict(id=key,owner=owner,body_equal_except_cold_salt=True,usage_equal=True,output_ID_vectors_identical=a==b,output_ID_common_prefix=common,total_output_IDs=len(a),evidence203_205=evidence))
summary={}
for n,v in runs.items():
 d=v["audit"];summary[str(n)]=dict(evidence=v["audit_ref"],elapsed_s=d["elapsed_s"],finite_TPS=d["finite_effective_output_tps"],short_latency=d["latency"]["short"],native_MTP=d["native_MTP"],all_SLO=d["diagnostic_reference_SLO"])
out=dict(at=datetime.now(timezone.utc).isoformat(),valid=True,kind="readonly_original_203205_trajectory_reduction",complete_pairs=12,paired_original_bodies_equal_except_cold_salt=True,pairs=pairs,summary=summary,unchanged_D1_native_geometry_source_policy_held=True,Current=None,new_model_operations=0,new_inference=0,limits=["Only immutable original windows; no GPU/profile/replay or capacityKEEP","D1 physicalepoch/config did not change, but native MTP acceptance .5937->.8615 and trajectories vary: uncontrolled counterexample to attributing windowTPS gain solely to D0Graph","Native HTTP token events/gaps and draftsequence counters are not physical GPU rounds; no causal hardware normalization","D0 capturelist andphysicalepoch differ; samplinginput/control same except newsalt but async PP-MTP/order/numericaltrajectory differ; repeat206 exactresident held is independent next discriminator"])
f=j/"reduction.json";f.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
result=dict(schema_version=1,job_id=j.name,status="completed",summary="Readonly203205 12 body-usage pairs/nativeIDs-MTP-HTTP event reduction VALID; D1 unchangedconfig trajectoryconfounder; noGraphcausalgain/KEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(f),locator="original12 bodies-nativeIDs-usage/MTP/events-clock")],unknowns=out["limits"],decision_request=None,next_check_at=None)
(j/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n");print(json.dumps(dict(valid=True,equal_ID_vectors=sum(x["output_ID_vectors_identical"]for x in pairs),reduction=ref(f))))
