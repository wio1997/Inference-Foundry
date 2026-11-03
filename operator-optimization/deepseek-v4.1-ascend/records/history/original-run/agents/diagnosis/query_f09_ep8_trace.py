#!/usr/bin/env python3
"""Read EP8 saved CSV to compare named F09 events with captured windows."""
import argparse,csv,hashlib,json
from collections import Counter
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
t=Path(a.task_root);wp=t/'captured-layer-attribution/module-call-metrics.json';manifest=t/'evidence/TRACE-EXPORT-SHA256.json'
w=[x for x in json.loads(wp.read_text()) if '_ep8_' in x['rank'] and x['run_in_model'] in (1,2,3) and x['layer'] in (2,8,14,20,24,28,32,36)]
assert len(w)==24 and len({x['rank'] for x in w})==1
rank=w[0]['rank'];p=Path(a.trace_root)/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
sh=hashlib.sha256(p.read_bytes()).hexdigest(); expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(manifest.read_text())['exports']}
assert sh==expected[rank]
events=[]
with p.open() as f:
 for z in csv.DictReader(f):
  if z['Type'] not in ('_quantize_indexer_query_kernel','VllmAscendQuantLightningIndexerV2'):continue
  start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
  events.append({'type':z['Type'],'start_ns':start,'end_ns':end,'stream':z['Stream ID'],'model':z['Model ID'],
     'input_shapes':z['Input Shapes'],'input_dtypes':z['Input Data Types'],'output_shapes':z['Output Shapes'],'output_dtypes':z['Output Data Types']})
f09=[x for x in events if x['type']=='_quantize_indexer_query_kernel'];assert len(f09)<=100
details=[]
for x in w:
 own=[z for z in events if z['type']=='VllmAscendQuantLightningIndexerV2' and z['stream']==str(x['main_stream_id']) and z['model']==str(x['model_id']) and x['window_start_ns']<=z['start_ns'] and z['end_ns']<=x['window_end_ns']]
 assert len(own)==1
 overlap=[z for z in f09 if z['start_ns']<x['window_end_ns'] and z['end_ns']>x['window_start_ns']]
 assert len(overlap)<=10
 details.append({'run':x['run_in_model'],'layer':x['layer'],'window_ns':[x['window_start_ns'],x['window_end_ns']],
                 'qli':own[0],'overlapping_f09_events':overlap,
                 'f09_end_before_qli_start':sum(z['end_ns']<=own[0]['start_ns'] for z in overlap)})
groups=Counter((z['input_shapes'],z['input_dtypes'],z['output_shapes'],z['output_dtypes'],z['model'],z['stream']) for z in f09)
out={'scope':'EP8 existing saved CSV; 24 target index-source layer windows, plus all same-name F09 CSV events on this rank',
 'rank':rank,'window_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'kernel_sha256':sh,'f09_event_count_all_trace':len(f09),
 'f09_shape_dtype_model_stream_groups':[{'input_shapes':k[0],'input_dtypes':k[1],'output_shapes':k[2],'output_dtypes':k[3],'model':k[4],'stream':k[5],'count':v} for k,v in sorted(groups.items())],
 'windows':sorted(details,key=lambda z:(z['run'],z['layer'])),
 'limitations':['Window overlap does not bind F09 to a QLI or request or prove tensor identity.',
                'CSV physical shapes may be missing or describe storage, not effective token/head counts or input data values.',
                'Task names alone do not prove original source, actual precision or dynamic branch.',
                'No NPU execution and no data-dependent tensor values are read.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
