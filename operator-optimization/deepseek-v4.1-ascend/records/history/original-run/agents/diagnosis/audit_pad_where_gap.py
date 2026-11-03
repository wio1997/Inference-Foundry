#!/usr/bin/env python3
"""Summarize selected same-scope tasks between pad and gate; CPU-only."""
import hashlib,json,statistics
from decimal import Decimal
from pathlib import Path
p=Path('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl');a=Path('agents/diagnosis/int64-pad-gating-scope.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(a.read_bytes()).hexdigest()=='49e4028342df962d0724c7f95aac4537b4804f7227049ffa42bf22040829c218'
idx={(r['ep'],r['scope']):r for r in json.loads(a.read_text())['rows']};assert len(idx)==320
names=('aclnnEqScalar_EqualAiCore_Equal','aclnnInplaceFillScalar_FillAiCore_Fill','aclnnSWhere_SelectV2AiCore_SelectV2')
out=[]
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('kind')!='rank' or x['data']['ep'] not in range(8):continue
  ep=x['data']['ep']
  for l in x['data']['layers']:
   if (ep,l['layer_scope_id']) not in idx:continue
   r=idx[(ep,l['layer_scope_id'])];lo=Decimal(r['pad']['start_us'])+Decimal(r['pad']['duration_us']);hi=Decimal(r['gate']['start_us']);tasks=[]
   for g in l['operator_groups']:
    for t in g['tasks']:
     s=Decimal(t['device_start_us']);e=s+Decimal(t['trace_duration_us'])
     if t['csv']['Stream ID']==r['pad']['stream'] and s>=lo and e<=hi:
      tasks.append((s,e,g['operator_name'],t['csv']['Task ID']))
   tasks.sort();assert tuple(t[2] for t in tasks)==names and all(tasks[i][1]<=tasks[i+1][0] for i in range(2))
   total=sum((e-s for s,e,_,_ in tasks),Decimal(0));gap=hi-lo;assert gap>0
   out.append({'ep':ep,'layer':r['layer'],'scope':r['scope'],'gap_us':str(gap),'three_task_union_us':str(total),'fraction':str(total/gap),'tasks':[{'name':n,'start_offset_us':str(s-lo),'duration_us':str(e-s),'task':tid} for s,e,n,tid in tasks]})
assert len(out)==320
v=[Decimal(r['three_task_union_us']) for r in out];f=[Decimal(r['fraction']) for r in out]
result={'scope':'same saved digest and selected scope; only uniquely matched same-stream tasks','n':len(out),'three_task_union_us':{'min':str(min(v)),'median':str(statistics.median(v)),'max':str(max(v))},'union_fraction_of_gap':{'min':str(min(f)),'median':str(statistics.median(f)),'max':str(max(f))},'rows':out}
Path('agents/diagnosis/pad-where-gap.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
