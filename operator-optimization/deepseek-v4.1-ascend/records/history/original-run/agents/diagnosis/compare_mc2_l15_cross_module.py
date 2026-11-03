#!/usr/bin/env python3
"""Reproduce selected same-scope L15 rank comparisons from frozen JSONL; CPU only."""
import hashlib, json, statistics
from decimal import Decimal
from pathlib import Path
p=Path('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl')
assert hashlib.sha256(p.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
names=('DispatchFFNCombine','SparseFlashMla','MoeGatingTopKHash')
fields=('Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats','Accelerator Core','Block Num','Mix Block Num')
rows={}
with p.open() as f:
 for line in f:
  obj=json.loads(line)
  if obj.get('kind')!='rank' or obj['data']['ep'] not in range(8): continue
  ep=obj['data']['ep']
  ls=[l for l in obj['data']['layers'] if l['layer']==15 and l['scheduler_digest'].startswith('db852bff987d')]
  assert len(ls)==1
  groups=ls[0]['operator_groups']; selected={}
  for name in names:
   gs=[g for g in groups if g['operator_name']==name]
   assert len(gs)==1 and len(gs[0]['tasks'])==1
   t=gs[0]['tasks'][0]; c=t['csv']
   selected[name]={'duration_us':t['trace_duration_us'],'stream':c['Stream ID'],'task_id':c['Task ID'],'flow_id':t['flow_id'],'signature':{k:c[k] for k in fields}}
  assert len({v['stream'] for v in selected.values()})==1
  rows[ep]={'scope':ls[0]['layer_scope_id'],'tasks':selected}
assert sorted(rows)==list(range(8))
for ep in range(7):
 for name in names:
  assert rows[ep]['tasks'][name]['signature']==rows[3]['tasks'][name]['signature']
assert rows[7]['tasks']['SparseFlashMla']['signature']!=rows[3]['tasks']['SparseFlashMla']['signature']
ratios={}
for name in names:
 vals=[Decimal(rows[ep]['tasks'][name]['duration_us'])/Decimal(rows[3]['tasks'][name]['duration_us']) for ep in (0,1,2,4,5,6)]
 ratios[name]={'min':str(min(vals)),'median':str(statistics.median(vals)),'max':str(max(vals))}
out={'input_sha256':'0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4','selected_ep':list(range(7)),'reference_ep':3,'excluded_ep7':'SparseFlashMla signature differs','ratios':ratios,'rows':rows}
Path('agents/diagnosis/mc2-l15-cross-module.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(ratios,indent=2))
