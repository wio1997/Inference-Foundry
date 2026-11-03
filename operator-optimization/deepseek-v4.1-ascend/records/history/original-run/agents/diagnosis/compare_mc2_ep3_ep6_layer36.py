#!/usr/bin/env python3
"""Compare five exact-signature task pairs in existing EP3/EP6 L36; CPU only."""
from pathlib import Path
from decimal import Decimal
import hashlib,json
R=Path(__file__).resolve().parents[2];p=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
assert h.hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
names=('DispatchFFNCombine','VllmAscendQuantLightningIndexerV2','_prepare_indexer_indices_kernel','SparseFlashMla','DequantSwigluQuant');items={}
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x['kind']!='rank' or x['data']['ep'] not in (3,6):continue
  ep=x['data']['ep']
  for l in x['data']['layers']:
   if l['layer']!=36 or not l['scheduler_digest'].startswith('db852bff987d'):continue
   for name in names:
    groups=[g for g in l['operator_groups'] if g['operator_name']==name];assert len(groups)==1 and len(groups[0]['tasks'])==1
    g=groups[0];t=g['tasks'][0];csv=t['csv'];assert (ep,name) not in items;items[ep,name]={'scope':l['layer_scope_id'],'digest':l['scheduler_digest'],'signature':g['physical_shape_dtype_format'],'core':csv['Accelerator Core'],'block':csv['Block Num'],'mixblock':csv['Mix Block Num'],'stream':csv['Stream ID'],'task_id':csv['Task ID'],'flow_id':t['flow_id'],'duration_us':t['trace_duration_us']}
pairs=[]
for name in names:
 a,b=items[3,name],items[6,name]
 for k in ('signature','core','block','mixblock','stream','digest'):assert a[k]==b[k]
 d3,d6=Decimal(a['duration_us']),Decimal(b['duration_us']);assert d3>0 and d6>0
 pairs.append({'name':name,'ep3':a,'ep6':b,'duration_ratio_ep6_over_ep3':str(d6/d3)})
out={'source_sha256':h.hexdigest(),'selection':'EP3/EP6 L36 db852... scope, one matched task per selected type, exact exported signature/core/block/stream','pairs':pairs,'limits':['One heavy-profiler capture, rank values and dynamic work not proven identical.','Same shape, task names, and launch blocks do not prove tensor, binary, or numerical workload identity.','Different task ratios reject a uniform elapsed-time multiplier for these records only, not a hardware fault or root cause.']}
q=R/'agents/diagnosis/mc2-ep3-ep6-layer36-comparison.json';q.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'n':len(pairs),'ratios':{x['name']:x['duration_ratio_ep6_over_ep3'] for x in pairs},'output_sha256':hashlib.sha256(q.read_bytes()).hexdigest()}))
