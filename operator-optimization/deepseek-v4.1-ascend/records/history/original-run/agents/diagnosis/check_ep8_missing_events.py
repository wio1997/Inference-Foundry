#!/usr/bin/env python3
"""Read saved EP8 trace and kernel CSV only; inspect three fixed L14 flow ids."""
from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal
import csv, hashlib, json, ijson

root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT')
trace=root/'trace_view.json'; table=root/'kernel_details.csv'
expected={'trace':'53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23','csv':'20cfb97abe03befda3fbf2c1194e48531aac44ce17d4f5fc130ea8ffb5261730'}
for name,p in [('trace',trace),('csv',table)]:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 if h.hexdigest()!=expected[name]:raise ValueError(name+' hash mismatch')
ids={'1793241329133278593','1793241329133776863','1793241329133776903'}
start=defaultdict(list); end=defaultdict(list);scope=[]
with trace.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict):continue
  if e.get('ph')=='X' and e.get('name')=='wio_diag:DeepseekV41DecoderLayer.forward:737-20' and str(e.get('pid'))=='737' and str(e.get('tid'))=='737':scope.append({'start':str(e['ts']),'duration':str(e['dur'])})
  if e.get('name')=='torch_to_npu' and e.get('cat')=='async_npu' and str(e.get('id')) in ids:
   if e.get('ph') in ('s','f'):
    (start if e['ph']=='s' else end)[str(e['id'])].append({'pid':str(e['pid']),'tid':str(e['tid']),'ts':str(e['ts'])})
if len(scope)!=1 or any(len(start[k])!=1 or len(end[k])!=1 for k in ids):raise ValueError('scope or flow not unique')
lo=Decimal(scope[0]['start']);hi=lo+Decimal(scope[0]['duration'])
if any(start[k][0]['pid']!='737' or start[k][0]['tid']!='737' or not lo<=Decimal(start[k][0]['ts'])<hi for k in ids):raise ValueError('flow start outside scope')
keys={(v[0]['pid'],v[0]['tid'],Decimal(v[0]['ts'])):k for k,v in end.items()}
device=defaultdict(list)
with trace.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict) or e.get('ph')!='X' or e.get('ts') is None:continue
  key=(str(e.get('pid')),str(e.get('tid')),Decimal(str(e['ts'])))
  if key in keys:
   a=e.get('args') or {};device[keys[key]].append({'name':e.get('name'),'ts':str(e['ts']),'duration':str(e.get('dur')),'model':str(a.get('Model Id')),'task':str(a.get('Task Id')),'stream':str(a.get('Physic Stream Id'))})
if any(len(device[k])!=1 for k in ids):raise ValueError('device X not unique')
exact=Counter();identity=defaultdict(list);names=Counter();rows=0
with table.open(newline='',encoding='utf8') as f:
 for row in csv.DictReader(f):
  rows+=1;names[row.get('Name')]+=1
  for k in ids:
   e=device[k][0]
   if (str(row.get('Model ID')),str(row.get('Task ID')),str(row.get('Stream ID')))!=(e['model'],e['task'],e['stream']):continue
   identity[k].append({'name':row.get('Name'),'start':row.get('Start Time(us)','').strip()})
   try:
    if Decimal(row.get('Start Time(us)','').strip())==Decimal(e['ts']):exact[k]+=1
   except Exception:pass
print(json.dumps({'scope':'737-20','scope_interval_us':scope[0],'source_sha256':expected,'csv_rows':rows,'csv_name_counts':{n:names[n] for n in ('EVENT RECORD','EVENT WAIT','RmsNorm')},'cases':{k:{'start':start[k][0],'finish':end[k][0],'device_x':device[k][0],'csv_exact_key_count':exact[k],'csv_same_model_task_stream_rows':identity[k]} for k in sorted(ids)}},indent=2))
