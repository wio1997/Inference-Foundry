#!/usr/bin/env python3
"""Historical selected scope: temporal geometry of scatter and one selected matmul, not a DAG."""
import json, hashlib, statistics
from collections import Counter
from decimal import Decimal as D
from pathlib import Path
R=Path(__file__).resolve().parents[2]; P=R/'agents/fusion'
src=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
tr= json.loads((P/'f08-actual-triples.json').read_text()); var=json.loads((P/'f08-scatter-variation.json').read_text())
sha=lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(src)==tr['source_sha256']==var['source_sha256']
want={(x['ep'],x['scope_id']):x for x in tr['records'] if x['rows']==8180}
meta={(x['ep'],x['scope']):x for x in var['all_samples_sorted']}
assert len(want)==len(meta)==640 and set(want)==set(meta)
rows=[]
for ln in src.open():
 x=json.loads(ln)
 if x.get('kind')!='rank': continue
 ep=x['data']['ep']
 for s in x['data']['layers']:
  k=(ep,s['layer_scope_id'])
  if k not in want: continue
  assert s['trace_scope_count']==1 and s['sidecar_thread_matches_trace']
  tasks=[(g['operator_name'],z) for g in s['operator_groups'] for z in g['tasks']]
  sc=[z for n,z in tasks if z['flow_id']==want[k]['flow_ids'][2]]; assert len(sc)==1
  s0=D(sc[0]['device_start_us']); sd=D(sc[0]['trace_duration_us']); s1=s0+sd
  m=meta[k]; assert sd==D(str(m['duration_us'])) and sc[0]['csv']['Task ID']==m['task']
  mats=[]
  for n,z in tasks:
   if 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3' not in n: continue
   a=D(z['device_start_us']); b=a+D(z['trace_duration_us']); ov=max(D(0),min(s1,b)-max(s0,a)); gap=max(D(0),a-s1,s0-b)
   mats.append((gap,-ov,a,b,z))
  assert mats
  mats.sort(key=lambda v:(v[0],v[1],v[2],v[3],str(v[4]['flow_id'])))
  gap,negov,a,b,z=mats[0]; overlap=-negov; hull=max(s1,b)-min(s0,a); md=b-a
  # This equals pair-hull reduction only if all endpoints stay fixed after removing scatter.
  outside=hull-md
  cls='short' if sd<50 else 'long' if sd>200 else 'middle'
  rows.append({'ep':ep,'digest':m['digest'],'class':cls,'mat_count':len(mats),'tied_distance_n':sum(v[0]==gap for v in mats),'tied_gap_overlap_n':sum(v[:2]==(gap,negov) for v in mats),'scatter_us':float(sd),'matmul_us':float(md),'overlap_us':float(overlap),'pair_hull_us':float(hull),'pair_hull_minus_matmul_us':float(outside),'outside_fraction':float(outside/sd),'gap_us':float(gap),'position':('overlap' if overlap>0 else 'mat_after' if a>=s1 else 'mat_before')})
assert len(rows)==640

def ag(xs):
 def stat(k):
  v=[r[k] for r in xs];return [min(v),statistics.median(v),max(v)] if v else None
 return {'n':len(xs),'position':dict(Counter(r['position'] for r in xs)),'ties_on_distance_n':sum(r['tied_distance_n']>1 for r in xs),'ties_after_overlap_n':sum(r['tied_gap_overlap_n']>1 for r in xs),'scatter_us_min_med_max':stat('scatter_us'),'matmul_us_min_med_max':stat('matmul_us'),'overlap_us_min_med_max':stat('overlap_us'),'pair_hull_us_min_med_max':stat('pair_hull_us'),'pair_hull_minus_matmul_us_min_med_max':stat('pair_hull_minus_matmul_us'),'outside_fraction_min_med_max':stat('outside_fraction'),'gap_us_min_med_max':stat('gap_us')}
out={'source_sha256':sha(src),'triples_sha256':sha(P/'f08-actual-triples.json'),'variation_sha256':sha(P/'f08-scatter-variation.json'),'selection':'same-scope quant matmul nearest by temporal distance; ties choose maximum overlap, then earlier timestamp; selection only, no data dependency claim','limitation':'hull is pair record span, not critical path, module duration, or removable end-to-end time; gap can inflate hull outside scatter; endpoint invariance counterfactual not demonstrated','all':{c:ag([r for r in rows if r['class']==c]) for c in ('short','long','middle')},'ep3_db852':{c:ag([r for r in rows if r['ep']==3 and r['digest'].startswith('db852') and r['class']==c]) for c in ('short','long','middle')}}
(P/'f08-pair-interval-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
