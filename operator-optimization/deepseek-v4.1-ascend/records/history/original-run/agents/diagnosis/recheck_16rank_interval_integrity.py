#!/usr/bin/env python3
"""Cross-scope identity, device tracks and digest overlap for frozen 16-rank selections."""
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2];p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert sha256(p.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
def union(a):
 a=sorted(a);s=e=None;v=Decimal(0)
 for x,y in a:
  if s is None:s,e=x,y
  elif x>e:v+=e-s;s,e=x,y
  else:e=max(e,y)
 return v+(e-s if s is not None else 0)
out=[]
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('kind')!='rank':continue
  r=x['data'];by=defaultdict(list);mc=defaultdict(list);tracks=set();ids=defaultdict(list)
  for l in r['layers']:
   for g in l['operator_groups']:
    for t in g['tasks']:
     row=t['csv'];start=Decimal(t['device_start_us']);dur=Decimal(t['trace_duration_us']);d=l['scheduler_digest'];uid=(t['device_track_pid'],row['Stream ID'],row['Task ID'],t['device_start_us']);ids[uid].append((d,l['layer']));tracks.add(t['device_track_pid']);by[d].append((start,start+dur))
     if g['operator_name']=='DispatchFFNCombine':mc[d].append((dur,l['layer']))
  assert len(by)==2 and len(tracks)==1
  names=sorted(by);a,b=map(lambda k:union(by[k]),names);inter=a+b-union(by[names[0]]+by[names[1]])
  minmax=[]
  for d in names:
   m=mc[d];assert len(m)==40
   largest=max(m);total=sum(x[0] for x in m)
   minmax.append({'digest':d,'shortest_task_us':str(min(m)[0]),'longest_task_us':str(largest[0]),'longest_layer':largest[1],'longest_fraction_mc2_duration_sum':str(largest[0]/total),'mc2_task_duration_sum_us':str(total)})
  out.append({'ep':r['ep'],'distinct_device_track_pids':len(tracks),'selected_rows':sum(map(len,by.values())),'unique_device_stream_task_start_identities':len(ids),'identities_seen_multiple_scopes':sum(len(v)>1 for v in ids.values()),'digest_pair_interval_intersection_us':str(inter),'digests':minmax})
print(json.dumps({'source_sha256':sha256(p.read_bytes()).hexdigest(),'rows':out,'limits':['same track pid and capture-supported time coordinate, not an independent cross-clock calibration','interval overlap only among selected tasks, not scheduler/request equivalence or complete device capture','identity uniqueness at this key does not rule out different identities referring to the same physical work','longest fraction uses sum of 40 MC2 task durations, not request time or online throughput']},ensure_ascii=False,indent=2))
