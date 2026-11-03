#!/usr/bin/env python3
"""Offline same-rank/same-digest device interval coverage without cross-rank shape joins."""
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from statistics import median
import json
root=Path(__file__).resolve().parents[2]
p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert sha256(p.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
def union(intervals):
 intervals=sorted(intervals);start=end=None;total=Decimal(0)
 for s,e in intervals:
  assert e>s
  if start is None:start,end=s,e
  elif s>end:total+=end-start;start,end=s,e
  else:end=max(end,e)
 return total+((end-start) if start is not None else 0)
def interval(rows):return [(r[0],r[1]) for r in rows]
results=[];rank_scope_coverage=[];eps=set()
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('kind')!='rank':continue
  r=x['data'];ep=r['ep'];assert ep not in eps and r['status']=='completed';eps.add(ep)
  rank_scope_coverage.append({'ep':ep,'counts':r['counts'],'selected_operator_coverage':{k:r['operator_coverage'].get(k) for k in ('DispatchFFNCombine','SparseFlashMla','EVENT WAIT','EVENT RECORD')}})
  by_digest=defaultdict(list)
  for l in r['layers']:by_digest[l['scheduler_digest']].append(l)
  assert len(by_digest)==2
  for digest,layers in by_digest.items():
   assert len(layers)==40 and sorted(l['layer'] for l in layers)==list(range(40))
   groups=defaultdict(list);uids=set()
   for l in layers:
    assert l['matched_task_count']==sum(len(g['tasks']) for g in l['operator_groups'])
    for g in l['operator_groups']:
     for t in g['tasks']:
      row=t['csv'];key=(row['Stream ID'],row['Task ID'],t['device_start_us']);assert key not in uids;uids.add(key)
      s=Decimal(t['device_start_us']);dur=Decimal(t['trace_duration_us']);assert dur>0
      assert Decimal(t['duration_abs_difference_us'])<=Decimal('0.01')
      groups[(g['operator_name'],row['Type'])].append((s,s+dur))
   allrows=[row for rows in groups.values() for row in rows];den=union(allrows)
   assert len(allrows)==len(uids) and den>0
   group_data=[]
   for (name,typ),rows in groups.items():
    group_data.append({'name':name,'type':typ,'count':len(rows),'union_us':str(union(rows))})
   group_data.sort(key=lambda y:(-Decimal(y['union_us']),y['name']))
   mc2key=('DispatchFFNCombine','DispatchFFNCombine');assert mc2key in groups and len(groups[mc2key])==40
   mc=union(groups[mc2key]);other=union([z for k,v in groups.items() if k!=mc2key for z in v]);overlap=mc+other-den
   results.append({'ep':ep,'digest':digest,'scope_count':len(layers),'selected_tasks':len(allrows),'selected_union_us':str(den),'mc2_union_us':str(mc),'mc2_fraction':str(mc/den),'mc2_other_overlap_us':str(overlap),'mc2_exclusive_against_selected_us':str(mc-overlap),'largest_type':group_data[0],'types':group_data})
assert eps==set(range(16)) and len(results)==32
summaries=[]
for digest in sorted({x['digest'] for x in results}):
 a=[x for x in results if x['digest']==digest];assert len(a)==8
 order=sorted(a,key=lambda x:Decimal(x['mc2_fraction']))
 summaries.append({'digest':digest,'rank_count':len(a),'mc2_lowest_fraction':order[0]['mc2_fraction'],'mc2_lowest_ep':order[0]['ep'],'mc2_median_fraction':str(median(Decimal(x['mc2_fraction']) for x in a)),'mc2_highest_fraction':order[-1]['mc2_fraction'],'mc2_highest_ep':order[-1]['ep'],'mc2_largest_type_count':sum(x['largest_type']['name']=='DispatchFFNCombine' for x in a)})
print(json.dumps({'source_sha256':sha256(p.read_bytes()).hexdigest(),'analysis':'each row uses only one EP device timeline and its 40 selected target scopes; no cross-rank physical matching','results':results,'by_digest':summaries,'rank_80_scope_coverage':rank_scope_coverage,'limits':['task intervals are selected, exported, uniquely joined rows, not whole-device time or a request critical path','per-type unions overlap; cannot sum percentages or infer saved time','per-digest eight ranks are descriptive samples from same capture, not independent repetitions or proof of matching request or equal work','missing CSV joins include EVENT WAIT/RECORD and identity-missing device events','captured profiler overhead and task semantic boundaries remain unquantified']},ensure_ascii=False,indent=2))
