#!/usr/bin/env python3
"""Pair repeated Hc tasks in each exported scoped group; offline only."""
import collections, hashlib, json, statistics
from pathlib import Path
p=Path('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl')
source_sha=hashlib.sha256(p.read_bytes()).hexdigest()
assert source_sha=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
keys=['Name','Type','Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats','Accelerator Core','Block Num','Mix Block Num','Stream ID']
out=collections.defaultdict(list); excluded=collections.Counter(); seen_scopes=set()
for line in p.open():
 x=json.loads(line)
 if x.get('kind')!='rank': continue
 for layer in x['data']['layers']:
  for group in layer['operator_groups']:
   name=group['operator_name']
   if name not in ('HcPre','HcPost'):continue
   scope=(x['data']['ep'],layer['scheduler_digest'],layer['layer'],name)
   assert scope not in seen_scopes
   seen_scopes.add(scope)
   t=group['tasks']
   if len(t)!=2: excluded[f'{name}:count']+=1;continue
   a,b=sorted(t,key=lambda a:float(a['device_start_us']))
   ca,cb=a['csv'],b['csv']
   if any(ca.get(k)!=cb.get(k) or ca.get(k) is None for k in keys):excluded[f'{name}:signature']+=1;continue
   d1,d2=float(ca['Duration(us)']),float(cb['Duration(us)'])
   if d1<=0 or d2<=0: excluded[f'{name}:duration']+=1;continue
   delta=float(b['device_start_us'])-float(a['device_start_us'])
   overlap=delta < float(a['trace_duration_us'])
   ident=(layer['scheduler_digest'],name,ca['Input Shapes'])
   out[ident].append({'first_us':d1,'second_us':d2,'ratio':d2/d1,'overlap':overlap})
res={'source_sha256':source_sha,'signature_keys':keys,'excluded':dict(excluded),'groups':[],'limits':['Captured scopes only; identical signature does not show same tensor, data values, work or event dependency.','Different digests and shape bins are not controlled workload comparison.','Device start timestamps and durations are export records; no proof of exclusive resource usage, root cause or recoverable latency.','Ratios sorted independently, cannot join cross-operator or rank samples.']}
for (digest,name,shape),v in sorted(out.items()):
 rows=sorted(v,key=lambda t:t['ratio']);r=[x['ratio'] for x in rows]
 res['groups'].append({'digest':digest,'name':name,'input_shapes':shape,'count':len(v),'overlap_count':sum(x['overlap'] for x in rows),'first_median_us':statistics.median(x['first_us'] for x in rows),'second_median_us':statistics.median(x['second_us'] for x in rows),'ratio_median':statistics.median(r),'ratio_range':[min(r),max(r)],'pairs':rows})
assert len(seen_scopes)==2560 and len(res['groups'])==8 and not excluded
assert all(g['count']==320 and g['overlap_count']==0 for g in res['groups'])
path=Path('agents/diagnosis/hc-repeated-scope-anonymous-pairs.json');path.write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
for g in res['groups']:print(g['digest'][:8],g['name'],g['count'],g['overlap_count'],round(g['first_median_us'],3),round(g['second_median_us'],3),round(g['ratio_median'],3),*[round(x,3) for x in g['ratio_range']])
print('excluded',res['excluded'])
