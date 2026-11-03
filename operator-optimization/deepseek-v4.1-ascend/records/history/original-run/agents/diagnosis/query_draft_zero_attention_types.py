#!/usr/bin/env python3
"""Reproduce existing zero-attention windows and decompose main-stream AIC proxy by exported type. CPU-only."""
import argparse,csv,hashlib,json
from collections import defaultdict
from decimal import Decimal,InvalidOperation
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--trace-root',required=True);a=ap.parse_args()
x=json.loads(FROZEN_INPUT); by=defaultdict(list)
for w in x['windows']: by[w['rank']].append(w)
if len(by)!=16 or len(x['windows'])!=126:raise RuntimeError('scope mismatch')
def decimal(v):
 try:
  d=Decimal(v);return d if d.is_finite() else None
 except (InvalidOperation,TypeError):return None
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
operator=defaultdict(lambda:{'rows':0,'den':Decimal(0),'num':Decimal(0),'skipped':0}); windows=[]
for rank,ws in sorted(by.items()):
 p=Path(a.trace_root)/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 if sha(p)!=x['csv_sha256_by_rank'][rank]:raise RuntimeError('CSV changed: '+rank)
 for key in {(w['model'],w['stream']) for w in ws}:
  group=sorted((w for w in ws if (w['model'],w['stream'])==key),key=lambda w:w['start'])
  if any(a['end']>b['start'] for a,b in zip(group,group[1:])):raise RuntimeError('overlapping windows')
 work=[{'den':Decimal(0),'num':Decimal(0),'skipped':0,'positive':0,'mc2':0} for _ in ws]
 with p.open(newline='') as f:
  reader=csv.DictReader(f)
  required={'Type','Model ID','Stream ID','Task ID','Start Time(us)','Duration(us)','aic_total_cycles','aic_mac_ratio'}
  if not required.issubset(reader.fieldnames or ()):raise RuntimeError('missing columns')
  for z in reader:
   if 'metadata' in z['Type'].lower():continue
   start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
   for i,w in enumerate(ws):
    if z['Model ID']!=w['model'] or z['Stream ID']!=w['stream'] or start<w['start'] or end>w['end']:continue
    name=z['Type'];q=work[i]
    if len(name)>128:raise RuntimeError('unexpected type')
    g=operator[name];c=decimal(z['aic_total_cycles']);r=decimal(z['aic_mac_ratio'])
    if c is None or c<=0:q['skipped']+=1;g['skipped']+=1;continue
    if r is None or not Decimal(0)<=r<=Decimal(1):raise RuntimeError('invalid ratio')
    q['num']+=c*r;q['den']+=c;q['positive']+=1;g['num']+=c*r;g['den']+=c;g['rows']+=1
    if name=='DispatchFFNCombine':
     if z['Task ID']!=w['mc2_task_id']:raise RuntimeError('wrong MC2 id '+rank)
     q['mc2']+=1
 for i,w in enumerate(ws):
  q=work[i]
  if q['mc2']!=1:raise RuntimeError('non-unique positive MC2 '+rank)
  if q['num']!=Decimal(w['expected_num']) or q['den']!=Decimal(w['expected_den']) or q['skipped']!=w['expected_skipped']:raise RuntimeError('window total mismatch '+rank)
  windows.append({'rank':rank,'start':w['start'],'end':w['end'],'positive':q['positive'],'skipped':q['skipped'],'den':str(q['den']),'num':str(q['num'])})
result=[]
for name,v in sorted(operator.items()):
 result.append({'type':name,'rows':v['rows'],'skipped_no_positive_cycles':v['skipped'],'den':str(v['den']),'num':str(v['num']),'ratio':str(v['num']/v['den']) if v['den'] else None})
print(json.dumps({'scope':'126 zero-exported-attention candidate brackets, 16 ranks, v5 main-stream rules','inputs':x['inputs'],'csv_sha256_by_rank':x['csv_sha256_by_rank'],'windows':windows,'operators':result,'limits':['Missing SparseFlashMla does not prove draft identity or no attention','Positive-cycle weighted exported counter proxy, not time or NPU utilization','Counts duplicated or overlapping CSV rows as original metric; other streams excluded','Task types are export labels; arithmetic contribution is not a cause'], 'no_device_work':True},ensure_ascii=False))
