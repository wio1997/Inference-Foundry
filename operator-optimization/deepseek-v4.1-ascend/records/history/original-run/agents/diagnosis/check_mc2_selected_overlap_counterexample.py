#!/usr/bin/env python3
"""Offline selected-task counterexamples; no causality or device access."""
import hashlib, json
from decimal import Decimal
from pathlib import Path
root=Path(__file__).resolve().parents[2]
src=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
over=root/'agents/diagnosis/cross-layer-aicpu-attribution.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(src)=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert sha(over)=='61c8639fd775a2907ebdf31ce5103f974831df61da9ba8031216f346002b7f3f'
select={'7ddf7ae9':{'zero':[0,1],'full':[4,5,6,7]},'db852bff':{'zero':[0,1,2],'full':[4,5,6,7]}}
pairs=json.loads(over.read_text())['pairs']
look={(p['digest'][:8],p['ep']):p for p in pairs}
assert len(look)==len(pairs)==32
fields=['Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats','Accelerator Core','Block Num','Mix Block Num']
tasks={}
for line in src.open():
 r=json.loads(line)
 if r.get('kind')!='rank': continue
 rank=r['data']
 for layer in rank['layers']:
  prefix=layer['scheduler_digest'][:8]; ep=rank['ep']
  if prefix not in select or ep not in select[prefix]['zero']+select[prefix]['full'] or layer['layer'] not in (14,39): continue
  for group in layer['operator_groups']:
   if group['operator_name']!='DispatchFFNCombine': continue
   assert len(group['tasks'])==1
   t=group['tasks'][0]; c=t['csv']; key=(prefix,ep,layer['layer'])
   assert key not in tasks
   tasks[key]={'duration':Decimal(c['Duration(us)']),'signature':tuple(c[k] for k in fields), 'task':str(c['Task ID']), 'stream':str(c['Stream ID']),'start':str(t['device_start_us']),'line':t['csv_line']}
result=[]
for digest,groups in select.items():
 baseline=None
 for name,eps in groups.items():
  for ep in eps:
   a=tasks[digest,ep,14]; b=tasks[digest,ep,39]; p=look[digest,ep]
   assert a['duration']==Decimal(p['layer14_mc2']['duration_us']) and a['task']==p['layer14_mc2']['task'] and a['stream']==p['layer14_mc2']['stream'] and a['start']==p['layer14_mc2']['device_start_us'] and a['line']==p['layer14_mc2']['csv_line']
   frac=Decimal(p['overlap_fraction_of_mc2']); assert frac==({'zero':Decimal(0),'full':Decimal(1)}[name])
   assert a['signature']==b['signature']
   if baseline is None: baseline=a['signature']
   assert a['signature']==baseline
   result.append({'digest':digest,'ep':ep,'overlap':name,'l14_us':str(a['duration']),'l39_us':str(b['duration']),'l14_l39':str(a['duration']/b['duration']),'l14_task':a['task'],'l39_task':b['task']})
 assert baseline
print(json.dumps({'sources':{'all_tasks_sha256':sha(src),'overlap_sha256':sha(over)},'signature_fields':fields,'selected_group_ep':select,'rows':result,'limits':['observational same-digest cross-rank selection; no controlled intervention','overlap only with selected L23 AICPU RS record in exported coordinates','same signature is not equal effective workload, rank state, weights, executable, or routing','does not exclude other streams, remote-rank effects or resource contention','profiled device task duration not end-to-end latency']},ensure_ascii=False,indent=2))
