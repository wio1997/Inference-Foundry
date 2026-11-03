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
groups=defaultdict(list)
for p in pair:groups[(p['rank'],p['layer'],p['indexer']['input_shapes'])].append(p)
rows=[]
for (rank,layer,shape),v in sorted(groups.items()):
 cy=sum(x['indexer']['cycles'] for x in v)
 rows.append({'rank':rank,'layer':layer,'shape':shape,'calls':len(v),
  'indexer_mean_us':statistics.mean(x['indexer']['duration_us'] for x in v),
  'attention_mean_us':statistics.mean(x['attention']['duration_us'] for x in v),
  'indexer_mac_cycle_proxy':sum(x['indexer']['mac_product'] for x in v)/cy if cy else None,
  'indexer_before_attention_count':sum(x['indexer_end_to_attention_start_us']>=0 for x in v),
  'gap_mean_us':statistics.mean(x['indexer_end_to_attention_start_us'] for x in v)})
layers=defaultdict(list)
for p in pair:layers[p['layer']].append(p)
layer_summary=[{'layer':k,'calls':len(v),'ranks':len({x['rank'] for x in v}),
 'indexer_median_us':statistics.median(x['indexer']['duration_us'] for x in v),
 'attention_median_us':statistics.median(x['attention']['duration_us'] for x in v),
 'same_stream_indexer_end_before_attention_start':sum(x['indexer_end_to_attention_start_us']>=0 for x in v)} for k,v in sorted(layers.items())]
out={'scope':'already attributed target interior runs1/2/3, 16 ranks; same model/stream windows','window_sha256':hashlib.sha256(windows_path.read_bytes()).hexdigest(),'raw_kernel_sha256_by_rank':hashes,
 'paired_calls':len(pair),'attention_calls':attention_count,'layer_summary':layer_summary,'rank_layer_shape_groups':rows,
 'limitations':['Same-stream time order supports temporal ordering only; it does not establish data identity, critical path or exposed time.',
 'Frozen source supports conditional index selection consumed by attention; exact runtime tensor mapping and branches require separate evidence.',
 'Rank/shape groups are samples, not causal comparison or end-to-end improvement; kernel outputs and stream waits may be unobserved.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
