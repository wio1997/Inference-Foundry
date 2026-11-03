#!/usr/bin/env python3
"""Read only: exact EP8 host layer scope -> flow -> device X, bounded output."""
from pathlib import Path
from decimal import Decimal
from collections import defaultdict,Counter
import hashlib,json,ijson

rank=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT')
trace=rank/'trace_view.json'
expected='53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23'
h=hashlib.sha256()
with trace.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
if h.hexdigest()!=expected:raise ValueError('trace hash mismatch')
targets={14:('737-20',Decimal('1793241329140513.188'),Decimal('160365.95575'),'44217'),39:('737-45',Decimal('1793241329850316.287'),Decimal('8290.41425'),'45363')}
scopes=defaultdict(list);starts=defaultdict(list);ends=defaultdict(list)
with trace.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict):continue
  name=e.get('name');phase=e.get('ph')
  if phase=='X' and str(e.get('pid'))=='737' and str(e.get('tid'))=='737':
   for layer,(sid,_,_,_) in targets.items():
    if name=='wio_diag:DeepseekV41DecoderLayer.forward:'+sid:
     a=Decimal(str(e['ts']));scopes[layer].append((a,a+Decimal(str(e['dur']))))
  if name=='torch_to_npu' and e.get('cat')=='async_npu' and phase in ('s','f'):
   try:x=(str(e.get('pid')),str(e.get('tid')),Decimal(str(e['ts'])))
   except (KeyError,ValueError):continue
   (starts if phase=='s' else ends)[str(e.get('id'))].append(x)
for layer in targets:
 if len(scopes[layer])!=1:raise ValueError('scope missing or duplicated')
keys=defaultdict(list);chosen=defaultdict(list);skipped=Counter()
for flow,ss in starts.items():
 if len(ss)!=1 or len(ends[flow])!=1:
  skipped['ambiguous_flow']+=1;continue
 start=ss[0]
 if start[:2]!=('737','737'):continue
 for layer in targets:
  lo,hi=scopes[layer][0]
  if lo<=start[2]<=hi:
   key=ends[flow][0];keys[key].append((layer,flow));chosen[layer].append(flow)
device=defaultdict(list)
with trace.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict) or e.get('ph')!='X':continue
  try:key=(str(e.get('pid')),str(e.get('tid')),Decimal(str(e['ts'])))
  except (KeyError,ValueError):continue
  if key not in keys:continue
  a=e.get('args') if isinstance(e.get('args'),dict) else {}
  try:d=Decimal(str(e['dur']))
  except (KeyError,ValueError):continue
  device[key].append({'name':e.get('name'),'cat':e.get('cat'),'duration':d,'model':str(a.get('Model Id')),'task':str(a.get('Task Id')),'stream':str(a.get('Physic Stream Id'))})
result={}
for layer,(_,base,dur,taskid) in targets.items():
 rows=[];unmatched=ambiguous=0
 for key,refs in keys.items():
  local=[flow for li,flow in refs if li==layer]
  if not local:continue
  if len(refs)!=1 or len(device[key])!=1:
   if len(device[key])==0:unmatched+=len(local)
   else:ambiguous+=len(local)
   continue
  ev=device[key][0];start=key[2];end=start+ev['duration'];clip=max(Decimal(0),min(base+dur,end)-max(base,start))
  rows.append({'flow_id':local[0],'track_pid':key[0],'track_tid':key[1],'name':ev['name'],'cat':ev['cat'],'model':ev['model'],'task':ev['task'],'stream':ev['stream'],'relative_start_us':str(start-base),'duration_us':str(ev['duration']),'overlap_us':str(clip)})
 anchor=[r for r in rows if r['name']=='DispatchFFNCombine' and r['task']==taskid and r['relative_start_us']=='0.000']
 if len(anchor)!=1:raise ValueError('target MC2 anchor not uniquely reproduced')
 result[str(layer)]={'scope':str(targets[layer][0]),'mc2_duration_us':str(dur),'same_thread_flow_count':len(chosen[layer]),'matched_unique':len(rows),'unmatched':unmatched,'ambiguous':ambiguous,'rows':rows}
print(json.dumps({'trace_sha256':expected,'scope':'EP8 only; same layer host thread flow->unique finish->unique X; same scope is not tensor or causal dependency','layers':result,'globally_skipped_ambiguous_flow_count':skipped['ambiguous_flow']},indent=2))
