#!/usr/bin/env python3
"""EP3 same-capture selected target scopes, device-interval unions; CPU/read-only."""
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from statistics import median
import json
root=Path(__file__).resolve().parents[2]
p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert sha256(p.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
with p.open() as f:
 ranks=[x['data'] for line in f if (x:=json.loads(line)).get('kind')=='rank' and x['data'].get('ep')==3]
assert len(ranks)==1
r=ranks[0];layers=[l for l in r['layers'] if l['scheduler_digest'].startswith('db852bff')]
assert len(layers)==40 and sorted(l['layer'] for l in layers)==list(range(40)) and len({l['scheduler_digest'] for l in layers})==1
groups=defaultdict(list);uids=set()
for layer in layers:
 assert layer['matched_task_count']==sum(len(g['tasks']) for g in layer['operator_groups'])
 for group in layer['operator_groups']:
  for t in group['tasks']:
   row=t['csv'];key=(row['Stream ID'],row['Task ID'],t['device_start_us'])
   assert key not in uids;uids.add(key)
   s=Decimal(t['device_start_us']);d=Decimal(t['trace_duration_us']);assert d>0
   assert Decimal(t['duration_abs_difference_us'])<=Decimal('0.01')
   groups[(group['operator_name'],row['Type'])].append({'layer':layer['layer'],'start':s,'end':s+d,'duration':d,'row':row})
def union(intervals):
 intervals=sorted(intervals);start=end=None;total=Decimal(0)
 for s,e in intervals:
  if start is None:start,end=s,e
  elif s>end:total+=end-start;start,end=s,e
  else:end=max(end,e)
 return total+((end-start) if start is not None else 0)
def iv(rows):return [(x['start'],x['end']) for x in rows]
allrows=[x for rows in groups.values() for x in rows];allunion=union(iv(allrows));out=[]
for (name,typ),rows in groups.items():
 rest=[x for key,rr in groups.items() if key!=(name,typ) for x in rr]
 u=union(iv(rows));other=union(iv(rest));intersection=u+other-allunion
 out.append({'name':name,'type':typ,'count':len(rows),'interval_sum_us':str(sum((x['duration'] for x in rows),Decimal(0))),'interval_union_us':str(u),'selected_union_fraction':str(u/allunion),'overlap_union_with_other_selected_types_us':str(intersection),'exclusive_vs_other_selected_types_us':str(u-intersection),'duration_median_us':str(median(x['duration'] for x in rows))})
out.sort(key=lambda x:(-Decimal(x['interval_union_us']),x['name']))
mc2=groups[('DispatchFFNCombine','DispatchFFNCombine')];assert len(mc2)==40
mcrows=[{'layer':x['layer'],'duration_us':str(x['duration']),'aic_mac_ratio_csv':x['row']['aic_mac_ratio'],'aic_total_cycles_csv':x['row']['aic_total_cycles'],'aiv_vec_ratio_csv':x['row']['aiv_vec_ratio'],'aiv_total_cycles_csv':x['row']['aiv_total_cycles']} for x in sorted(mc2,key=lambda x:-x['duration'])]
coverage={name:r['operator_coverage'][name] for name in ('DispatchFFNCombine','SparseFlashMla','VllmAscendQuantLightningIndexerV2','EVENT WAIT','EVENT RECORD')}
result={'source_sha256':sha256(p.read_bytes()).hexdigest(),'ep':3,'digest':layers[0]['scheduler_digest'],'scopes':len(layers),'selected_task_count':len(allrows),'unique_selected_task_identities':len(uids),'selected_task_duration_sum_us':str(sum((x['duration'] for x in allrows),Decimal(0))),'selected_task_interval_union_us':str(allunion),'type_groups':out,'mc2_layer_rows':mcrows,'rank_80_scopes_counts':r['counts'],'rank_80_scopes_selected_operator_coverage':coverage,'limits':['Intervals are selected exported records in 40 target layer scopes, not the complete device activity or request windows.','Per-type interval unions may overlap; fractions are not additive or critical-path shares.','One rank and one digest, not a cross-rank matched group or an independent online benchmark.','Device trace intervals include profiler and task-type semantics; they are not necessarily pure compute or communication.','Rank coverage counters refer to all 80 target scopes across two digests, not just the selected 40.']}
print(json.dumps(result,ensure_ascii=False,indent=2))
