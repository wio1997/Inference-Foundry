#!/usr/bin/env python3
"""Bounded CPU-only query: same-rank/model/stream MC2 in existing draft brackets."""
import argparse,csv,hashlib,json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--trace-root',required=True); args=ap.parse_args()
x=json.loads(FROZEN_INPUT); by=defaultdict(list)
for w in x['windows']:by[w['rank']].append(w)
if len(x['windows'])!=288 or len(by)!=16:raise RuntimeError('unexpected scope')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
records=[]
for rank,ws in sorted(by.items()):
 p=Path(args.trace_root)/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 if sha(p)!=x['csv_sha256_by_rank'][rank]:raise RuntimeError('CSV hash changed: '+rank)
 for key in {(w['model'],w['stream']) for w in ws}:
  a=sorted((w for w in ws if (w['model'],w['stream'])==key),key=lambda w:w['start'])
  if any(u['end']>v['start'] for u,v in zip(a,a[1:])):raise RuntimeError('overlap '+rank)
 hits=[[] for w in ws]
 with p.open(newline='') as f:
  rd=csv.DictReader(f); required={'Type','Model ID','Stream ID','Task ID','Start Time(us)','Duration(us)'}
  if not required.issubset(rd.fieldnames or []):raise RuntimeError('missing CSV fields')
  for z in rd:
   if z['Type']!='DispatchFFNCombine':continue
   start=Decimal(z['Start Time(us)'].strip())*1000;duration=Decimal(z['Duration(us)'].strip())*1000;end=start+duration
   if duration<=0:raise RuntimeError('nonpositive MC2')
   for i,w in enumerate(ws):
    if z['Model ID']==w['model'] and z['Stream ID']==w['stream'] and z['Task ID']==w['mc2_task_id'] and start>=w['start'] and end<=w['end']:
     hits[i].append((start,end,duration))
 for w,h in zip(ws,hits):
  if len(h)!=1:raise RuntimeError('MC2 not unique in window '+rank+' '+str(w['mc2_task_id'])+' '+str(len(h)))
  start,end,duration=h[0]; window=Decimal(w['end']-w['start'])
  if window<=0:raise RuntimeError('bad window')
  records.append({'rank':rank,'group':w['group'],'window_start_ns':w['start'],'window_end_ns':w['end'],'model':w['model'],'stream':w['stream'],'task_id':w['mc2_task_id'],'mc2_start_ns':str(start),'mc2_end_ns':str(end),'mc2_duration_ns':str(duration),'window_duration_ns':str(window)})
print(json.dumps({'records':records,'inputs':x['inputs'],'csv_sha256_by_rank':x['csv_sha256_by_rank'],'limits':['Same rank/model/stream and CSV task; two groups differ and are not paired workloads','Window-local MC2 duration is not end-to-end critical-path contribution, device busy time, communication wait or speedup','Other streams and requests may overlap; rank clocks cannot be compared without synchronization proof','Zero exported attention does not establish draft identity or absence of attention'], 'no_device_work':True}))
