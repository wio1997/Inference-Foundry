#!/usr/bin/env python3
"""Inspect only preselected EP8 layer24 run2 historical CSV; no device work."""
import argparse,csv,hashlib,json
from pathlib import Path
from decimal import Decimal
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
t=Path(a.task_root); wp=t/'captured-layer-attribution/module-call-metrics.json'; mp=t/'evidence/TRACE-EXPORT-SHA256.json'
w=[x for x in json.loads(wp.read_text()) if '_ep8_' in x['rank'] and x['run_in_model']==2 and x['layer']==24]
if len(w)!=1: raise RuntimeError('expected exactly one EP8 layer24 run2 window')
w=w[0]; assert (w['window_start_ns'],w['window_end_ns'])==(1793152960208017846,1793152960209362219)
p=Path(a.trace_root)/w['rank']/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'; assert p.stat().st_size<=50*1024*1024
sha=hashlib.sha256(p.read_bytes()).hexdigest(); expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(mp.read_text())['exports']}; assert sha==expected[w['rank']]
rows=[]
with p.open(newline='') as f:
 for z in csv.DictReader(f):
  s=int(Decimal(z['Start Time(us)'].strip())*1000); e=s+int(Decimal(z['Duration(us)'].strip())*1000)
  if s>=w['window_start_ns'] and e<=w['window_end_ns'] and e>s:
   rows.append({'type':z['Type'],'name':z.get('Name',''),'model':z['Model ID'],'stream':z['Stream ID'],'task_id':z['Task ID'],'start_ns':s,'end_ns':e,'input_shapes':z['Input Shapes'],'input_dtypes':z['Input Data Types'],'output_shapes':z['Output Shapes'],'output_dtypes':z['Output Data Types']})
if len(rows)>500: raise RuntimeError('window event cap exceeded')
main=[r for r in rows if r['model']==str(w['model_id']) and r['stream']==str(w['main_stream_id'])]
qli=[r for r in main if r['type']=='VllmAscendQuantLightningIndexerV2'];smla=[r for r in main if r['type']=='SparseFlashMla']
if len(qli)!=1 or len(smla)!=1 or qli[0]['end_ns']>smla[0]['start_ns']:raise RuntimeError('anchors absent, ambiguous, or reversed')
lo=qli[0]['end_ns'];hi=smla[0]['start_ns'];between=[r for r in rows if r['start_ns']>=lo and r['end_ns']<=hi]
for r in (qli+smla+between):r['start_rel_ns']=r['start_ns']-w['window_start_ns'];r['end_rel_ns']=r['end_ns']-w['window_start_ns']
print(json.dumps({'scope':'Single fixed source-attributed EP8 layer24 run2: QLI, SMLA, and complete recorded kernel events between their boundaries','rank':w['rank'],'window_ns':[w['window_start_ns'],w['window_end_ns']],'main_stream':w['main_stream_id'],'model':w['model_id'],'qli':qli[0],'smla':smla[0],'gap_events':sorted(between,key=lambda x:x['start_ns']),'input_sha256':sha,'window_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'limits':['Same window, shape, model, stream and time order do not prove tensor identity, full call stack, source branch, runtime values or critical path.','Events overlapping gap edges are omitted from the contained list; this is not a complete whole-window event listing.','Only one historical sample; graph task IDs can recur; no request identity, new profiling or device operation.']},ensure_ascii=False,indent=2))
