#!/usr/bin/env python3
import json,hashlib,statistics
from pathlib import Path
from collections import Counter
from decimal import Decimal
root=Path(__file__).resolve().parents[2];src=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';want='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
h=hashlib.sha256(src.read_bytes()).hexdigest()
if h!=want:raise ValueError('source SHA mismatch')
names=('VllmAscendQuantLightningIndexerV2','_prepare_indexer_indices_kernel','SparseFlashMla');target='db852bff987d2b98d3dc0734e3731ec5528e37dcf6e874860d1ebb4fd12eefc8';layers={2,8,14,20,24,28,32,36};scope=[];rankfoot=[]
with src.open() as f:
 for line in f:
  j=json.loads(line)
  if j.get('kind')=='rank':
   rankfoot.append((j['data']['ep'],j['data']['status']))
   for s in j['data']['layers']:
    if s['scheduler_digest']==target and s['layer'] in layers:scope.append(s)
assert len(scope)==64 and Counter(s['ep'] for s in scope)=={i:8 for i in range(8)} and all(status=='completed' for ep,status in rankfoot)
def selected(s):
 out={}
 for name in names:
  groups=[g for g in s['operator_groups'] if g['operator_name']==name];n=sum(len(g['tasks']) for g in groups);out[name]={'count':n}
  if n==1:
   g=next(g for g in groups if g['tasks']);t=g['tasks'][0];out[name].update(signature=g['physical_shape_dtype_format'],duration_us=t['trace_duration_us'],start_us=t['device_start_us'],stream=t['device_track_tid'],track_pid=t['device_track_pid'],flow=t['flow_id'],task=t['csv']['Task ID'])
 return out
ref=[s for s in scope if s['ep']==3 and s['layer']==36];assert len(ref)==1;ref=selected(ref[0]);assert all(ref[n]['count']==1 for n in names)
records=[]
for s in scope:
 v=selected(s);r={'ep':s['ep'],'layer':s['layer'],'scope_id':s['layer_scope_id'],'operator_counts':{n:v[n]['count'] for n in names}}
 if all(v[n]['count']==1 for n in names):
  r['signature_equal_ref']={n:v[n]['signature']==ref[n]['signature'] for n in names};r['all_signatures_equal_ref']=all(r['signature_equal_ref'].values());r['same_track']=len({(v[n]['track_pid'],v[n]['stream']) for n in names})==1
  r['durations_us']={n:v[n]['duration_us'] for n in names};r['flows']={n:v[n]['flow'] for n in names};r['task_ids']={n:v[n]['task'] for n in names}
  r['ordered_notoverlap']=Decimal(v[names[0]]['start_us'])+Decimal(v[names[0]]['duration_us'])<=Decimal(v[names[1]]['start_us']) and Decimal(v[names[1]]['start_us'])+Decimal(v[names[1]]['duration_us'])<=Decimal(v[names[2]]['start_us'])
 records.append(r)
def summarize(v):
 out={'n':len(v)}
 for n in names:
  z=[Decimal(r['durations_us'][n]) for r in v];out[n]={'min_us':str(min(z)) if z else None,'median_us':str(statistics.median(z)) if z else None,'max_us':str(max(z)) if z else None}
 return out
unique=[r for r in records if all(r['operator_counts'][n]==1 for n in names)];fixed=[r for r in unique if r['all_signatures_equal_ref'] and r['same_track'] and r['ordered_notoverlap']]
out={'source_sha256':h,'digest':target,'scope':'EP0-7 same saved scheduler digest, 8 selected index-source layers; one profile, not identical requests or work','selected_scopes':len(records),'unique_all_three':len(unique),'fixed_triple_signature_same_track_ordered':len(fixed),'reference':{'ep':3,'layer':36,'scope_id':'2128-42','durations_us':{n:ref[n]['duration_us'] for n in names},'triple_signature':{n:ref[n]['signature'] for n in names}},'all_unique':summarize(unique),'exact_signature':summarize(fixed),'exact_signature_by_ep':{str(ep):summarize([r for r in fixed if r['ep']==ep]) for ep in range(8)},'exact_signature_by_layer':{str(l):summarize([r for r in fixed if r['layer']==l]) for l in sorted(layers)},'records':records,'limits':['Physical signature and scheduler digest are not dynamic workload, values, storage, binary, request identity or no-disturbance evidence.','All durations from trace event intervals under one intrusive profiler; not independent online samples or steady-state latency.','One matched task per name is limited by extraction coverage; no-match or exclusions do not mean absence of execution.','Same track and sequence do not prove tensor edges or wait causality; durations are not additive end-to-end or achievable speedup.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
