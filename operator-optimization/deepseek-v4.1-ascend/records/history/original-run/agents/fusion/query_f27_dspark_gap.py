#!/usr/bin/env python3
"""Inspect only previously exported EP8 kernel shapes in one fixed inter-forward gap."""
import argparse,csv,hashlib,json,re
from pathlib import Path
from decimal import Decimal
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
t=Path(a.task_root);wp=t/'captured-layer-attribution/module-call-metrics.json';mp=t/'evidence/TRACE-EXPORT-SHA256.json'
w=[x for x in json.loads(wp.read_text()) if '_ep8_' in x['rank'] and x['run_in_model']==2 and x['layer']==39]
if len(w)!=1:raise RuntimeError('ambiguous layer window')
w=w[0];assert w['window_end_ns']==1793152960223546576
start=w['window_end_ns'];end=1793152960261768012  # fixed boundary from saved EP8 rank_gaps.csv
p=Path(a.trace_root)/w['rank']/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';assert p.stat().st_size<=50*1024*1024
sha=hashlib.sha256(p.read_bytes()).hexdigest();expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(mp.read_text())['exports']};assert sha==expected[w['rank']]
events=[]
with p.open(newline='') as f:
 for z in csv.DictReader(f):
  s=int(Decimal(z['Start Time(us)'].strip())*1000);e=s+int(Decimal(z['Duration(us)'].strip())*1000)
  if not(start<=s<end and e>s):continue
  events.append({'type':z['Type'],'name':z.get('Name',''),'model':z['Model ID'],'stream':z['Stream ID'],'task_id':z['Task ID'],'start_rel_ns':s-start,'end_rel_ns':e-start,'input_shapes':z['Input Shapes'],'input_dtypes':z['Input Data Types'],'output_shapes':z['Output Shapes'],'output_dtypes':z['Output Data Types']})
pat=re.compile(r'(?<!\d)15360(?!\d)');matches=[e for e in events if pat.search(e['input_shapes']+' '+e['output_shapes'])]
if len(matches)>50:raise RuntimeError('candidate cap exceeded')
neighbor_types={'RmsNorm','hcom_allGather_','Cast','MatMulV2','MatMulV3','ReduceSum','Mul'}
near=[e for e in events if e not in matches and e['type'] in neighbor_types and any(e['start_rel_ns'] < m['end_rel_ns']+30000 and e['end_rel_ns']>m['start_rel_ns']-30000 for m in matches)]
if len(near)>200:raise RuntimeError('neighbor cap exceeded')
print(json.dumps({'scope':'fixed EP8 layer39 run2 final-marker end to next target layer0 start; shape text containing 15360 and selected types within 30us','rank':w['rank'],'anchor_end_ns':start,'gap_end_ns':end,'matches':sorted(matches,key=lambda x:x['start_rel_ns']),'neighbor_events':sorted(near,key=lambda x:x['start_rel_ns']),'csv_sha256':sha,'window_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'limits':['Same shape, time or physical stream do not establish tensor identity, source call, request identity, or event causality.','No hit cannot rule out unrecorded shapes, non-exported tasks, another rank, packed or fused implementation.','Task IDs may be reused; nearby records may duplicate. No summation is assumed.','No device execution or payload reads.']},ensure_ascii=False,indent=2))
