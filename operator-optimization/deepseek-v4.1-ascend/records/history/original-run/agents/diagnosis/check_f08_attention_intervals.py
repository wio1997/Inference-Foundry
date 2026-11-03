#!/usr/bin/env python3
"""Read-only comparison of trace intervals paired only inside matched device/scope.

Task identities use existing host-flow linked task records, never name/time as identity;
Decimal preserves sub-microsecond precision of epoch-sized trace timestamps.
"""
import collections, hashlib, json
from decimal import Decimal
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parents[2]
src=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
tri=root/'agents/fusion/f08-actual-triples.json'
expected={src:'0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4',tri:'e6c18bd23809d47652cc58faf8d5ac81f7879be0e483fc72d70264cd3a3940d5'}
for p,h in expected.items():
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,p
records=json.loads(tri.read_text())['records']
ix={(x['ep'],x['scope_id']):x for x in records};assert len(ix)==len(records)==1280
counts=collections.defaultdict(collections.Counter); paired=collections.defaultdict(list);seen=set();dup=0; devices=collections.Counter(); triples=[]
for line in src.open():
 data=json.loads(line)
 if data.get('kind')!='rank':continue
 for scope in data['data']['layers']:
  key=scope['ep'],scope['layer_scope_id']
  if key not in ix:continue
  assert key not in seen;seen.add(key)
  tasks=[t for g in scope['operator_groups'] for t in g['tasks']]
  # Duplicate flow IDs within each selected scope are an ambiguity, not independent durations.
  flows=[t['flow_id'] for t in tasks];dup+=len(flows)-len(set(flows))
  target=ix[key]
  ts=[]
  for fid in target['flow_ids']:
   found=[t for t in tasks if t['flow_id']==fid];assert len(found)==1
   ts.append(found[0])
  norm,rope,scatter=ts
  assert [t['csv']['Type'] for t in ts]==['RmsNorm','InplacePartialRotaryMul','ScatterNdUpdateSk']
  for name in ('SparseFlashMla','VllmAscendQuantLightningIndexerV2'):
   found=[t for t in tasks if t['csv']['Type']==name]
   counts[name][len(found)]+=1
   if len(found)!=1:continue
   other=found[0]
   assert other['flow_id'] not in target['flow_ids']
   assert scatter['device_track_pid']==other['device_track_pid'],(key,scatter['device_track_pid'],other['device_track_pid'])
   def interval(t):
    a=Decimal(t['device_start_us']);return a,a+Decimal(t['trace_duration_us'])
   a,b=interval(scatter);c,d=interval(other)
   overlap=max(Decimal(0),min(b,d)-max(a,c))
   # positive gap = other starts after scatter end; negative means starts earlier.
   same_stream=scatter['csv']['Stream ID']==other['csv']['Stream ID']
   x={'ep':target['ep'],'layer':target['layer'],'scope_id':target['scope_id'],'rows':target['rows'],
      'scatter_flow':scatter['flow_id'],'other_flow':other['flow_id'],
      'scatter_task':scatter['csv']['Task ID'],'other_task':other['csv']['Task ID'],
      'scatter_stream':scatter['csv']['Stream ID'],'other_stream':other['csv']['Stream ID'],
      'scatter_start_us':str(a),'scatter_end_us':str(b),'other_start_us':str(c),'other_end_us':str(d),
      'overlap_us':str(overlap),'scatter_end_to_other_start_us':str(c-b),
      'order':'scatter_before' if b<=c else ('other_before' if d<=a else 'overlap'),
      'same_stream':same_stream}
   paired[name].append(x)
  selected={name:[t for t in tasks if t['csv']['Type']==name] for name in ('SparseFlashMla','VllmAscendQuantLightningIndexerV2')}
  if all(len(v)==1 for v in selected.values()):
   s=selected['SparseFlashMla'][0];i=selected['VllmAscendQuantLightningIndexerV2'][0]
   assert s['device_track_pid']==i['device_track_pid']==scatter['device_track_pid']
   ia=Decimal(i['device_start_us']);ib=ia+Decimal(i['trace_duration_us']);sa=Decimal(s['device_start_us']);sb=sa+Decimal(s['trace_duration_us'])
   triples.append({'ep':target['ep'],'layer':target['layer'],'scope_id':target['scope_id'],
    'indexer_flow':i['flow_id'],'attention_flow':s['flow_id'],
    'indexer_end_to_attention_start_us':str(sa-ib),'indexer_attention_overlap_us':str(max(Decimal(0),min(ib,sb)-max(ia,sa))),
    'same_stream':i['csv']['Stream ID']==s['csv']['Stream ID']})
assert len(seen)==1280 and seen==set(ix) and dup==0
summary={}
for name,rs in paired.items():
 gaps=[Decimal(x['scatter_end_to_other_start_us']) for x in rs]
 summary[name]={'count':len(rs),'order_counts':dict(collections.Counter(x['order'] for x in rs)),
 'same_stream':sum(x['same_stream'] for x in rs),
 'overlap_us_total':str(sum(Decimal(x['overlap_us']) for x in rs)),
 'scatter_end_to_other_start_us_min_median_max':[str(min(gaps)),str(median(gaps)),str(max(gaps))]}
indexer_keys={(x['ep'],x['scope_id']) for x in paired['VllmAscendQuantLightningIndexerV2']}
summary['attention_gap_by_indexer_presence']={}
for label, group in [('one_matched_indexer',[x for x in paired['SparseFlashMla'] if (x['ep'],x['scope_id']) in indexer_keys]),('no_matched_indexer',[x for x in paired['SparseFlashMla'] if (x['ep'],x['scope_id']) not in indexer_keys])]:
 gaps=[Decimal(x['scatter_end_to_other_start_us']) for x in group]
 summary['attention_gap_by_indexer_presence'][label]={'count':len(group),'scatter_end_to_attention_start_us_min_median_max':[str(min(gaps)),str(median(gaps)),str(max(gaps))]}
ig=[Decimal(x['indexer_end_to_attention_start_us']) for x in triples]
summary['indexer_vs_attention']={'count':len(triples),'same_stream':sum(x['same_stream'] for x in triples),
 'indexer_before_or_touch_attention':sum(x>=0 for x in ig), 'overlap':sum(Decimal(x['indexer_attention_overlap_us'])>0 for x in triples),
 'indexer_end_to_attention_start_us_min_median_max':[str(min(ig)),str(median(ig)),str(max(ig))]}
out={'input_sha256':{str(p.relative_to(root)):h for p,h in expected.items()},'scope_count':len(seen),'duplicate_flow_ids_within_scope':dup,
 'matching_count_distribution':{k:dict(v) for k,v in counts.items()},'summary':summary,'pairs':dict(paired),'indexer_attention_pairs':triples,
 'limitations':['Only matched exported host-flow-linked tasks in 1280 selected profiled scopes; not every task or stream captured.','Same device track and scope do not establish tensor identity, dependency, exposed critical path, or independent measurements.','Intervals are trace timestamps; gaps and no interval overlap do not prove device idle or internal waits; can have unobserved or other-stream work.','Tasks and scopes may overlap across layers or repeat request work; totals are pairwise descriptions only, not elapsed time, baseline attribution or achievable speedup.']}
p=root/'agents/diagnosis/f08-attention-intervals.json';p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'counts':out['matching_count_distribution'],'summary':summary},indent=2));print('output sha256',hashlib.sha256(p.read_bytes()).hexdigest())
