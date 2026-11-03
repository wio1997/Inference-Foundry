#!/usr/bin/env python3
"""Read EP8 exported rows after the selected target-layer39 marker, not device work."""
import argparse,csv,hashlib,json
from pathlib import Path
from decimal import Decimal
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
t=Path(a.task_root); wp=t/'captured-layer-attribution/module-call-metrics.json';mp=t/'evidence/TRACE-EXPORT-SHA256.json'
w=[x for x in json.loads(wp.read_text()) if '_ep8_' in x['rank'] and x['run_in_model']==2 and x['layer']==39]
if len(w)!=1:raise RuntimeError('window ambiguous')
w=w[0];assert (w['window_start_ns'],w['window_end_ns'])==(1793152960222703473,1793152960223546576)
p=Path(a.trace_root)/w['rank']/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';assert p.stat().st_size<=50*1024*1024
sha=hashlib.sha256(p.read_bytes()).hexdigest();expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(mp.read_text())['exports']};assert sha==expected[w['rank']]
want={'Mul','ReduceSum','RmsNorm','Cast','hcom_allGather_','MatMulV2','HcPost','Copy','InplaceCopy'};events=[]
start=w['window_end_ns'];end=start+500_000
with p.open(newline='') as f:
 for z in csv.DictReader(f):
  if z['Type'] not in want:continue
  s=int(Decimal(z['Start Time(us)'].strip())*1000);e=s+int(Decimal(z['Duration(us)'].strip())*1000)
  if s>=start and e<=end and e>s:
   events.append({'type':z['Type'],'name':z.get('Name',''),'model':z['Model ID'],'stream':z['Stream ID'],'task_id':z['Task ID'],'start_rel_ns':s-start,'end_rel_ns':e-start,'input_shapes':z['Input Shapes'],'input_dtypes':z['Input Data Types'],'output_shapes':z['Output Shapes'],'output_dtypes':z['Output Data Types']})
if len(events)>100:raise RuntimeError('event cap exceeded')
events.sort(key=lambda x:x['start_rel_ns'])
print(json.dumps({'scope':'EP8 layer39 run2 final-marker end to +500us, selected operator names only; not request or target-tail attribution','rank':w['rank'],'anchor_end_ns':start,'events':events,'csv_sha256':sha,'window_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'limits':['500us adjacency alone is not same request, graph, source call, producer-consumer relation or target/draft identity.','Missing type, aliases or compiler-fused implementation do not prove absence.','Event records may duplicate or reflect different streams; no task-count or time addition implied.','No device or payload work.']},ensure_ascii=False,indent=2))
