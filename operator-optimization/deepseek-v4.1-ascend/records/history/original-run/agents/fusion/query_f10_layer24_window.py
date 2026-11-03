#!/usr/bin/env python3
"""Read fixed historical EP8 layer24 run2 indexer weight projection neighborhood."""
import argparse,csv,hashlib,json
from pathlib import Path
from decimal import Decimal
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
t=Path(a.task_root); wp=t/'captured-layer-attribution/module-call-metrics.json'; mp=t/'evidence/TRACE-EXPORT-SHA256.json'
w=[x for x in json.loads(wp.read_text()) if '_ep8_' in x['rank'] and x['run_in_model']==2 and x['layer']==24]
if len(w)!=1:raise RuntimeError('window ambiguous')
w=w[0]; assert (w['window_start_ns'],w['window_end_ns'])==(1793152960208017846,1793152960209362219)
p=Path(a.trace_root)/w['rank']/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';assert p.stat().st_size<=50*1024*1024
sha=hashlib.sha256(p.read_bytes()).hexdigest(); expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(mp.read_text())['exports']};assert sha==expected[w['rank']]
want={'MatMulV2','Cast','Mul','Muls','InplacePartialRotaryMul','_quantize_indexer_query_kernel','VllmAscendQuantLightningIndexerV2'};events=[]
with p.open(newline='') as f:
 for z in csv.DictReader(f):
  if z['Type'] not in want:continue
  s=int(Decimal(z['Start Time(us)'].strip())*1000);e=s+int(Decimal(z['Duration(us)'].strip())*1000)
  if s>=w['window_start_ns'] and e<=w['window_end_ns'] and e>s:
   events.append({'type':z['Type'],'name':z.get('Name',''),'model':z['Model ID'],'stream':z['Stream ID'],'task_id':z['Task ID'],'start_rel_ns':s-w['window_start_ns'],'end_rel_ns':e-w['window_start_ns'],'input_shapes':z['Input Shapes'],'input_dtypes':z['Input Data Types'],'output_shapes':z['Output Shapes'],'output_dtypes':z['Output Data Types']})
if len(events)>100:raise RuntimeError('event cap exceeded')
events.sort(key=lambda x:x['start_rel_ns'])
print(json.dumps({'scope':'fixed EP8 layer24 run2 selected op names only; no storage/callchain identification','rank':w['rank'],'window_ns':[w['window_start_ns'],w['window_end_ns']],'events':events,'csv_sha256':sha,'window_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'limits':['Names, shapes and ordering cannot bind tensors or Python source calls.','Selected operators do not include all events or necessarily all compiler-generated variants.','Physical shape does not imply valid token count, request or replay identity.','No profiling execution or NPU access.']},ensure_ascii=False,indent=2))
