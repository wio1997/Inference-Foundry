#!/usr/bin/env python3
"""Quantify same-device trace intervals between consecutive selected MC2 tasks."""
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json
from statistics import median
root=Path(__file__).resolve().parents[2]
p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
expected='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert sha256(p.read_bytes()).hexdigest()==expected
D=Decimal

def union(intervals):
 a=sorted(intervals); total=D(0)
 if not a:return total
 s,e=a[0]
 for x,y in a[1:]:
  if x>e:total+=e-s;s,e=x,y
  else:e=max(e,y)
 return total+e-s

def covered(gaps, intervals):
 return sum((union([(max(s,a),min(e,b)) for a,b in intervals if a<e and b>s]) for s,e in gaps), D(0))
results=[];eps=set()
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('kind')!='rank':continue
  r=x['data'];ep=r['ep'];assert ep not in eps and r['status']=='completed';eps.add(ep)
  by_digest=defaultdict(list)
  for l in r['layers']:by_digest[l['scheduler_digest']].append(l)
  assert len(by_digest)==2
  for digest,layers in by_digest.items():
   assert len(layers)==40 and sorted(l['layer'] for l in layers)==list(range(40))
   mc=[];other=[];seen=set()
   for l in layers:
    for g in l['operator_groups']:
     for t in g['tasks']:
      row=t['csv'];start=D(t['device_start_us']);end=start+D(t['trace_duration_us']);stream=row['Stream ID'];key=(t['device_track_pid'],stream,row['Task ID'],start)
      assert key not in seen and end>start;seen.add(key)
      if g['operator_name']=='DispatchFFNCombine': mc.append((l['layer'],start,end,stream,t['device_track_pid']))
      else:other.append((start,end,stream,t['device_track_pid']))
   assert len(mc)==40 and len(set(x[3] for x in mc))==len(set(x[4] for x in mc))==1
   assert len(set(x[3] for x in other)|set(x[4] for x in mc))==1
   mc.sort();assert [x[0] for x in mc]==list(range(40))
   assert all(a[2]<=b[1] for a,b in zip(mc,mc[1:]))
   gaps=[(a[2],b[1]) for a,b in zip(mc,mc[1:])]
   elapsed=mc[-1][2]-mc[0][1];active=sum((x[2]-x[1] for x in mc),D(0));gap=sum((e-s for s,e in gaps),D(0));assert elapsed==active+gap and gap>0
   same=[(s,e) for s,e,stream,pid in other if stream==mc[0][3]]
   all_other=[(s,e) for s,e,stream,pid in other]
   c_same=covered(gaps,same);c_any=covered(gaps,all_other);assert D(0)<=c_same<=c_any<=gap
   results.append({'ep':ep,'digest':digest,'mc2_us':str(active),'first_to_last_us':str(elapsed),'gap_us':str(gap),'same_stream_selected_us':str(c_same),'any_stream_selected_us':str(c_any),'same_stream_fraction':str(c_same/gap),'any_stream_fraction':str(c_any/gap),'gap_fraction_of_span':str(gap/elapsed)})
assert eps==set(range(16)) and len(results)==32
stats={}
for k in ['gap_fraction_of_span','same_stream_fraction','any_stream_fraction']:
 xs=sorted(D(r[k]) for r in results);stats[k]={'min':str(xs[0]),'median':str(median(xs)),'max':str(xs[-1])}
print(json.dumps({'source_sha256':expected,'method':'For each individual EP and digest, 40 MC2 records on same track/stream, layer-ordered and nonoverlapping; 39 gaps between consecutive records, intersected with unions of selected non-MC2 tasks.','results':results,'descriptive_across_32_scopes':stats,'limits':['same EP exported clock only; no cross-rank clock comparison','selected named task intervals only, not complete stream coverage or event WAIT/RECORD; uncovered is not device idle','task interval overlap does not prove active resource execution','snapshot under profiler, no claim for online timing or end-to-end request critical path','interval coverage is not avoidable waiting or optimization benefit']},indent=2,ensure_ascii=False))
