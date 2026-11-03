#!/usr/bin/env python3
"""Read existing matched task aggregation; compare target L14/L39 gate and MC2."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,statistics
R=Path(__file__).resolve().parents[2];p=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
assert h.hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
name={'gate':'MoeGatingTopKHash','mc2':'DispatchFFNCombine'};items={}
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x['kind']!='rank' or x['data']['ep'] not in range(8,16):continue
  ep=x['data']['ep']
  for l in x['data']['layers']:
   if l['layer'] not in (14,39) or not l['scheduler_digest'].startswith('1b826883b782'):continue
   group={}
   for alias,pattern in name.items():
    gs=[g for g in l['operator_groups'] if pattern in g['operator_name']];assert len(gs)==1 and len(gs[0]['tasks'])==1
    g=gs[0];t=g['tasks'][0];group[alias]={'operator':g['operator_name'],'signature':g['physical_shape_dtype_format'],'flow_id':t['flow_id'],'stream':t['csv']['Stream ID'],'start_us':t['device_start_us'],'duration_us':t['trace_duration_us'],'task_id':t['csv']['Task ID']}
   assert group['gate']['stream']==group['mc2']['stream'];gate,mc2=group['gate'],group['mc2'];gap=Decimal(mc2['start_us'])-Decimal(gate['start_us'])-Decimal(gate['duration_us']);assert gap>=0
   items[ep,l['layer']]={'scope':l['layer_scope_id'],'digest':l['scheduler_digest'],'tasks':group,'gap_us':str(gap)}
pairs=[]
for ep in range(8,16):
 a,b=items[ep,14],items[ep,39]
 for k in name:assert a['tasks'][k]['operator']==b['tasks'][k]['operator'] and a['tasks'][k]['signature']==b['tasks'][k]['signature']
 pairs.append({'ep':ep,'l14':a,'l39':b})
def stats(alias,layer):
 arr=[Decimal(p['l'+str(layer)]['tasks'][alias]['duration_us']) for p in pairs];return {'min':str(min(arr)),'median':str(statistics.median(arr)),'max':str(max(arr))}
longer=sum(Decimal(p['l14']['tasks']['gate']['duration_us'])>Decimal(p['l39']['tasks']['gate']['duration_us']) for p in pairs);shorter=sum(Decimal(p['l14']['tasks']['gate']['duration_us'])<Decimal(p['l39']['tasks']['gate']['duration_us']) for p in pairs)
out={'input_sha256':h.hexdigest(),'pairs':pairs,'gate_l14_longer':longer,'gate_l14_shorter':shorter,'gate_l14_stats':stats('gate',14),'gate_l39_stats':stats('gate',39),'mc2_l14_stats':stats('mc2',14),'mc2_l39_stats':stats('mc2',39),'limits':['All samples from same intrusive profiler capture; comparisons are same-rank, not independent requests.','Operator records and shape compatible within each layer scope do not establish tensor identity or producer-consumer edge.','Same physical signature does not prove same routing values, workloads, binary or internal dependencies.','Task gaps are not proof of idle or complete timeline.']}
q=R/'agents/diagnosis/mc2-l14-l39-gating-comparison.json';q.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'n':len(pairs),'gate_l14_longer':longer,'gate_l14_shorter':shorter,'gate14':out['gate_l14_stats'],'gate39':out['gate_l39_stats'],'output_sha256':hashlib.sha256(q.read_bytes()).hexdigest()}))
