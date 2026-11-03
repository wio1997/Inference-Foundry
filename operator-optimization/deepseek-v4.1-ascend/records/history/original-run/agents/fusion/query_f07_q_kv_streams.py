#!/usr/bin/env python3
"""Read one saved rank's kernel CSV for nine bounded target windows; no device work."""
import argparse,csv,hashlib,json
from pathlib import Path
from decimal import Decimal
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
t=Path(a.task_root);wp=t/'captured-layer-attribution/module-call-metrics.json';mp=t/'evidence/TRACE-EXPORT-SHA256.json'
windows=[x for x in json.loads(wp.read_text()) if '_ep8_' in x['rank'] and x['run_in_model'] in (1,2,3) and x['layer'] in (0,2,24)]
assert len(windows)==9 and len({x['rank'] for x in windows})==1
windows.sort(key=lambda x:x['window_start_ns']);assert all(x['window_end_ns']<=y['window_start_ns'] for x,y in zip(windows,windows[1:]))
rank=windows[0]['rank'];p=Path(a.trace_root)/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';assert p.stat().st_size<=50*1024*1024
sha=hashlib.sha256(p.read_bytes()).hexdigest();expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(mp.read_text())['exports']};assert sha==expected[rank]
want={'QuantBatchMatmulV3','MatMulV2','RmsNorm','InplacePartialRotaryMul','ScatterNdUpdateSk','SparseFlashMla','DynamicQuant'}
matched={(w['run_in_model'],w['layer']):[] for w in windows}
with p.open(newline='') as f:
 for z in csv.DictReader(f):
  if z['Type'] not in want:continue
  start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
  if end<=start:continue
  for w in windows:
   if start>=w['window_start_ns'] and end<=w['window_end_ns']:
    ev={'type':z['Type'],'name':z.get('Name',''),'model':z['Model ID'],'stream':z['Stream ID'],'task_id':z['Task ID'],
        'start_rel_ns':start-w['window_start_ns'],'end_rel_ns':end-w['window_start_ns'],
        'input_shapes':z['Input Shapes'],'input_dtypes':z['Input Data Types'],
        'output_shapes':z['Output Shapes'],'output_dtypes':z['Output Data Types']}
    matched[(w['run_in_model'],w['layer'])].append(ev)
rows=[]
for w in windows:
 ev=sorted(matched[(w['run_in_model'],w['layer'])],key=lambda e:e['start_rel_ns'])
 if len(ev)>100:raise RuntimeError('per-window output bound exceeded')
 rows.append({'run':w['run_in_model'],'layer':w['layer'],'main_stream':w['main_stream_id'],
   'model':w['model_id'],'events':ev})
print(json.dumps({'scope':'Existing EP8 CSV: three saved interior runs, layers 0,2,24; rank local window-contained selected operator types',
 'rank':rank,'window_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
 'csv_sha256':sha,'samples':rows,
 'limitations':['Same window and shape do not prove producer/consumer tensor identity or source branch.',
 'All selected streams shown, but events with missing model/stream IDs remain unknown; timing is not a happens-before event.',
 'Only selected operator types and nine windows are included; CSV event/wait identifiers are not independently bound.',
 'No cross-rank times, runtime/NPU operation, input values, or output text.']},ensure_ascii=False,indent=2))
