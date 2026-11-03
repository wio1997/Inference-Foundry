#!/usr/bin/env python3
"""Recheck F08 selected same-scope triples; CPU, old data, no cross-device times."""
import collections,hashlib,json,statistics
from decimal import Decimal as D
from pathlib import Path
R=Path(__file__).resolve().parents[2]
src=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';tri=R/'agents/fusion/f08-actual-triples.json'
exp={src:'0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4',tri:'e6c18bd23809d47652cc58faf8d5ac81f7879be0e483fc72d70264cd3a3940d5'}
for p,h in exp.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==h
T=json.loads(tri.read_text())['records'];want={(x['ep'],x['scope_id']):x for x in T if x['rows']==4760};assert len(want)==320
fields=('Type','Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats','Accelerator Core','Block Num','Mix Block Num')
rows=[];seen=set();sigs=set()
for l in src.open():
 j=json.loads(l)
 if j.get('kind')!='rank':continue
 for s in j['data']['layers']:
  key=s['ep'],s['layer_scope_id'];r=want.get(key)
  if r is None:continue
  assert key not in seen;seen.add(key);assert s['trace_scope_count']==1 and s['sidecar_thread_matches_trace']
  tasks=[(g['operator_name'],t) for g in s['operator_groups'] for t in g['tasks']];assert len({t['flow_id'] for _,t in tasks})==len(tasks)
  selected=[]
  for fid in r['flow_ids']:
   z=[t for _,t in tasks if t['flow_id']==fid];assert len(z)==1;selected.append(z[0])
  assert [x['csv']['Type'] for x in selected]==['RmsNorm','InplacePartialRotaryMul','ScatterNdUpdateSk']
  assert len({x['csv']['Stream ID'] for x in selected})==1
  ids=[int(x['csv']['Task ID']) for x in selected];assert ids==list(range(ids[0],ids[0]+3))
  sig=tuple(tuple(x['csv'][k] for k in fields) for x in selected);sigs.add(sig)
  a=selected[-1];assert all(t['device_track_pid']==a['device_track_pid'] for _,t in tasks)
  start=D(a['device_start_us']);end=start+D(a['trace_duration_us']);names=collections.Counter();matmul_gap=[]
  for name,t in tasks:
   if t['flow_id']==a['flow_id'] or t['device_track_tid']==a['device_track_tid']:continue
   x=D(t['device_start_us']);y=x+D(t['trace_duration_us'])
   if x<end and y>start:names[name]+=1
   if 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3' in name:matmul_gap.append(max(D(0),x-end,start-y))
  dur=end-start;assert dur<50 or dur>100
  rows.append({'ep':r['ep'],'scope_id':r['scope_id'],'layer':r['layer'],'digest':r['digest'],'flow_ids':r['flow_ids'],'tasks':ids,'duration_us':str(dur),'class':'short' if dur<50 else 'long','other_track_overlap_names':dict(names),'quant_matmul_count':len(matmul_gap),'nearest_quant_matmul_gap_us':str(min(matmul_gap)) if matmul_gap else None})
assert seen==set(want) and len(sigs)==1
assert {x['digest'] for x in rows}=={'7ddf7ae98a0060f65177698675c618f472c5697afff6bff80211376b4cabe1b9'}
for x in rows:
 if x['class']=='short':assert not x['other_track_overlap_names']
 else:assert x['other_track_overlap_names'].get('aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3',0)>0
by={(x['ep'],x['layer']):x for x in rows};assert len(by)==320 and all((e,l) in by for e in range(8) for l in range(40))
assert all(by[3,l]['class']=='short' and by[0,l]['class']=='long' for l in range(40))
def summ(z):
 ds=[D(x['duration_us']) for x in z];return {'count':len(z),'min_median_max_us':[str(min(ds)),str(statistics.median(ds)),str(max(ds))],'eps':dict(collections.Counter(str(x['ep']) for x in z))}
out={'input_sha256':{str(p.relative_to(R)):h for p,h in exp.items()},'scope':'320 selected T=4760 scope only; within-device timeline; 40 cross-rank layer pairs are matching names/shapes and same digest, not independently validated identical workload',
 'same_full_exported_three_task_signature_count':len(sigs),'signature_fields':fields,'signature':list(sigs)[0],
 'overall':{c:summ([x for x in rows if x['class']==c]) for c in ('short','long')},
 'ep3_vs_ep0_same_layer':{'ep3':summ([by[3,l] for l in range(40)]),'ep0':summ([by[0,l] for l in range(40)])},'records':rows,
 'limits':['Same layer and digest are not proof of tensor identity, matching effective slots/workload, or identical resource state.', 'Flow-linked selected tasks only, not all device work; overlap not proof of contention or causality.', 'Profiled samples; no cross-device timestamp subtraction, independent replication, end-to-end savings or utilization claim.']}
p=R/'agents/fusion/f08-4760-rank-overlap.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'overall':out['overall'],'ep3_vs_ep0_same_layer':out['ep3_vs_ep0_same_layer']},indent=2));print('output_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
