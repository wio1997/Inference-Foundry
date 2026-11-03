#!/usr/bin/env python3
"""CPU read-only 16-rank captured-target indexer/attention analysis; JSON stdout."""
import argparse,bisect,csv,hashlib,json,statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
task=Path(a.task_root);traces=Path(a.trace_root)
windows_path=task/'captured-layer-attribution/module-call-metrics.json'
windows=json.loads(windows_path.read_text());by=defaultdict(list)
for x in windows:
 if x['run_in_model'] in (1,2,3):by[(x['rank'],str(x['model_id']),str(x['main_stream_id']))].append(x)
for k in by:
 by[k].sort(key=lambda x:x['window_start_ns'])
 assert all(u['window_end_ns']<=v['window_start_ns'] for u,v in zip(by[k],by[k][1:]))
assert len({k[0] for k in by})==16 and sum(len(v) for v in by.values())==1920
events=defaultdict(lambda:defaultdict(list));hashes={}
for rp in sorted(traces.glob('*_ascend_pt')):
 rank=rp.name;p=rp/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 if rank not in {k[0] for k in by}:continue
 hashes[rank]=hashlib.sha256(p.read_bytes()).hexdigest()
 with p.open() as f:
  for z in csv.DictReader(f):
   typ=z['Type']
   if typ not in ('VllmAscendQuantLightningIndexerV2','SparseFlashMla'):continue
   key=(rank,z['Model ID'],z['Stream ID']);spans=by.get(key,[])
   if not spans:continue
   start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'])*1000)
   idx=bisect.bisect_right([x['window_start_ns'] for x in spans],start)-1
   if idx<0 or not (spans[idx]['window_start_ns']<=start and end<=spans[idx]['window_end_ns']):continue
   w=spans[idx];unit=(rank,w['run_in_model'],w['layer']);cycles=Decimal(z['aic_total_cycles'] or '0')
   if cycles>0 and not z['aic_mac_ratio']:raise ValueError('MAC field missing for positive AIC cycles')
   events[unit][typ].append({'start_ns':start,'end_ns':end,'duration_us':float(z['Duration(us)']),
    'input_shapes':z['Input Shapes'],'cycles':int(cycles),'mac_product':float(cycles*Decimal(z['aic_mac_ratio'] or '0'))})
assert len(hashes)==16
pair=[];attention_count=0
for (rank,run,layer),e in sorted(events.items()):
 att=e['SparseFlashMla'];ind=e['VllmAscendQuantLightningIndexerV2']
 assert len(att)==1 and len(ind)<=1;attention_count+=1
 if ind:
  i=ind[0];s=att[0]
  pair.append({'rank':rank,'run':run,'layer':layer,'indexer':i,'attention':s,'indexer_end_to_attention_start_us':(s['start_ns']-i['end_ns'])/1000})
assert attention_count==1920 and len(pair)==384
# Compare only independently selected rank/layer/run records; hashes are bound to stage082.
prior_bytes=PRIOR_JSON_TEXT.encode()
if hashlib.sha256(prior_bytes).hexdigest()!='0f728cbf37a5fc32bdaa924a8c206482240c7a23241549ef54efcf60344e01e3': raise RuntimeError('prior summary hash mismatch')
prior=json.loads(prior_bytes)
if hashlib.sha256(windows_path.read_bytes()).hexdigest()!=prior['window_sha256'] or hashes!=prior['raw_kernel_sha256_by_rank']: raise RuntimeError('raw source mismatch')
pairs={}
for p in pair:
 key=(p['rank'],p['layer'],p['run'])
 if key in pairs: raise RuntimeError('duplicate unit')
 pairs[key]=p
ranks=sorted({p['rank'] for p in pair});layers=(2,8,14,20,24,28,32,36)
if len(ranks)!=16 or len(pairs)!=384: raise RuntimeError('incomplete identities')
rows=[]
for rank in ranks:
 for layer in layers:
  g=[pairs[(rank,layer,run)] for run in (1,2,3)]
  if any(x['indexer_end_to_attention_start_us']<0 for x in g): raise RuntimeError('unexpected stream order')
  for left,right in ((1,2),(1,3),(2,3)):
   a,b=g[left-1],g[right-1]
   record={'rank':rank,'layer':layer,'from_run':left,'to_run':right,
      'indexer_same_shape':a['indexer']['input_shapes']==b['indexer']['input_shapes'],
      'indexer_end_before_attention_start':[a['indexer_end_to_attention_start_us'],b['indexer_end_to_attention_start_us']]}
   for kind in ('indexer','attention'):
    x,y=a[kind],b[kind]
    if x['duration_us']<=0 or y['duration_us']<=0:raise RuntimeError('invalid duration')
    record[kind]={'from_start_ns':x['start_ns'],'from_end_ns':x['end_ns'],'to_start_ns':y['start_ns'],'to_end_ns':y['end_ns'],
      'from_duration_us':x['duration_us'],'to_duration_us':y['duration_us'],'ratio':y['duration_us']/x['duration_us'],
      'direction':'up' if y['duration_us']>x['duration_us'] else ('down' if y['duration_us']<x['duration_us'] else 'equal')}
   rows.append(record)
summary=[]
for layer in layers:
 for left,right in ((1,2),(1,3),(2,3)):
  subset=[r for r in rows if (r['layer'],r['from_run'],r['to_run'])==(layer,left,right)]
  if len(subset)!=16:raise RuntimeError('group incomplete')
  entry={'layer':layer,'from_run':left,'to_run':right,'samples':len(subset),'indexer_same_shape':sum(r['indexer_same_shape'] for r in subset)}
  for kind in ('indexer','attention'):
   ratios=[r[kind]['ratio'] for r in subset];entry[kind]={'up':sum(r[kind]['direction']=='up' for r in subset),'down':sum(r[kind]['direction']=='down' for r in subset),'equal':sum(r[kind]['direction']=='equal' for r in subset),'ratio_min':min(ratios),'ratio_median':statistics.median(ratios),'ratio_max':max(ratios)}
  entry['both_up']=sum(r['indexer']['direction']=='up' and r['attention']['direction']=='up' for r in subset)
  summary.append(entry)
print(json.dumps({'scope':'16 fixed ranks × 8 indexer layers × runs 1/2/3; same-rank per-layer durations',
 'inputs':{'prior_sha256':hashlib.sha256(prior_bytes).hexdigest(),'windows_sha256':prior['window_sha256'],'csv_sha256_by_rank':hashes},
 'samples':rows,'summary':summary,'limits':['same rank/layer cross-run is not same request, equal inputs or controlled intervention',
 'model/stream/window attribution and identical shape do not bind runtime tensor identity',
 'task duration is not critical-path time; correlation is not causal',
 'no comparison of absolute timestamps across rank'], 'no_device_work':True},ensure_ascii=False,indent=2))
