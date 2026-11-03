#!/usr/bin/env python3
"""Compare fixed 32 matched same-rank/layer pairs of existing task records; CPU only."""
import hashlib,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2];p=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';expected='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
assert h.hexdigest()==expected
name='aclnnDynamicQuantV2_DynamicQuant_DynamicQuant';digests={'7ddf7ae98a00':595,'db852bff987d':1023};wanted=(0,12,24,36); events={}
with p.open() as f:
 for line in f:
  d=json.loads(line)
  if d['kind']!='rank' or d['data']['ep'] not in range(8):continue
  for l in d['data']['layers']:
   if l['layer'] not in wanted:continue
   prefix=l['scheduler_digest'][:12]
   if prefix not in digests:continue
   rows=digests[prefix];sig={'Input Shapes':f'"{rows},5120"','Output Shapes':f'"{rows},5120;{rows}"','Input Data Types':'DT_BF16','Output Data Types':'INT8;FLOAT','Input Formats':'ND','Output Formats':'ND;ND'}
   gs=[g for g in l['operator_groups'] if g['operator_name']==name and g['physical_shape_dtype_format']==sig];assert len(gs)==1 and len(gs[0]['tasks'])==1
   t=gs[0]['tasks'][0]; c=t['csv'];assert c['Name']==name and c['Type']=='DynamicQuant' and c['Input Shapes']==sig['Input Shapes'] and c['Output Shapes']==sig['Output Shapes'] and c['Input Data Types']==sig['Input Data Types'] and c['Output Data Types']==sig['Output Data Types'] and c['Input Formats']==sig['Input Formats'] and c['Output Formats']==sig['Output Formats']
   k=(d['data']['ep'],l['layer'],rows);assert k not in events
   events[k]={'scope':l['layer_scope_id'],'digest':l['scheduler_digest'],'flow':t['flow_id'],'stream':c['Stream ID'],'task_id':c['Task ID'],'duration_us':float(t['trace_duration_us']),'signature':sig}
pairs=[]
for ep in range(8):
 for layer in wanted:
  a=events[ep,layer,595];b=events[ep,layer,1023];assert a['stream']==b['stream'];assert a['duration_us']>0 and b['duration_us']>0
  ratio=b['duration_us']/a['duration_us'];pairs.append({'ep':ep,'layer':layer,'rows_595':a,'rows_1023':b,'duration_ratio':ratio,'per_physical_row_ratio':ratio*595/1023})
def stats(k):
 a=[x[k] for x in pairs];return {'min':min(a),'median':statistics.median(a),'max':max(a)}
out={'source':str(p.relative_to(R)),'source_sha256':h.hexdigest(),'selection':'EP0–7, layers 0/12/24/36, one matched task with exact input and output shapes, dtype and format per digest; same-rank and layer, different batches','count':len(pairs),'duration_ratio':stats('duration_ratio'),'per_physical_row_ratio':stats('per_physical_row_ratio'),'pairs':pairs,'limits':['Logical physical row dimensions are not known effective rows.','Profiler samples are different batches, not controlled repeated workload or uninstrumented latency.','Tensor/storage identity, scale contents, runtime API and binary unbound.','Numerical ratios do not prove fixed overhead, cause, critical path or optimization benefit.']}
q=R/'agents/diagnosis/dynamicquant-crossshape-pairs.json';q.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'count':len(pairs),'duration_ratio':out['duration_ratio'],'per_physical_row_ratio':out['per_physical_row_ratio'],'output_sha256':hashlib.sha256(q.read_bytes()).hexdigest()}))
