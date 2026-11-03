#!/usr/bin/env python3
"""Read existing EP9 metadata and unique target layer gating/MC2 records; no tensor data."""
from decimal import Decimal as D
from hashlib import sha256
from pathlib import Path
import json
root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';m=root/'diagnostic-20261029/runtime/metadata/events-752.jsonl'
def digest(p):
 with p.open('rb') as f:return sha256(f.read()).hexdigest()
assert digest(p)=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert digest(m)=='9528466dbfcfe95b28a7dd7b98c0f8e69ab405dedcd3d50d5d9976ff8045c65a'
with p.open() as f:
 ranks=[x['data'] for line in f if (x:=json.loads(line)).get('kind')=='rank' and x['data'].get('ep')==9]
assert len(ranks)==1
l=[v for v in ranks[0]['layers'] if v['layer_scope_id']=='752-51' and v['layer']==0 and v['scheduler_digest'].startswith('8c18c293')];assert len(l)==1;l=l[0]
events=[json.loads(line) for line in m.open()]
def event(id,label):
 a=[x for x in events if x.get('call_id')==id and x.get('kind')=='call_begin' and x.get('label')==label]
 assert len(a)==1 and a[0]['pid']==752 and a[0]['tid']==752
 return a[0]
runner=event('752-50','NPUModelRunner.execute_model');layer=event('752-51','DeepseekV41DecoderLayer.forward')
assert runner['schedule']['schedule_digest_sha256']==l['scheduler_digest']
tasks={}
for name,id in [('MoeGatingTopKHash','46754'),('DispatchFFNCombine','46755')]:
 z=[t for g in l['operator_groups'] if g['operator_name']==name for t in g['tasks'] if t['csv']['Task ID']==id]
 assert len(z)==1;t=z[0];row=t['csv'];assert row['Stream ID']=='47'
 keys=['Task ID','Stream ID','Input Shapes','Input Data Types','Output Shapes','Output Data Types']
 tasks[name]={'device_start_us':t['device_start_us'],'duration_us':t['trace_duration_us'],'fields':{k:row[k] for k in keys}}
g=tasks['MoeGatingTopKHash'];c=tasks['DispatchFFNCombine'];gap=D(c['device_start_us'])-D(g['device_start_us'])-D(g['duration_us'])
print(json.dumps({'source_sha256':{'target_jsonl':digest(p),'ep9_metadata':digest(m)},'scope':'EP9 layer0 selected scope only; no tensor content','runner_tensor_metadata':runner.get('tensor_metadata'),'layer_tensor_metadata':layer.get('tensor_metadata'),'tasks':tasks,'same_stream_gating_end_to_mc2_start_us':str(gap),'limits':['metadata contains tensor shapes, dtypes and device types only; tensor inputs are compacted without positional indices, kwargs are not recorded','shape/dtype compatibility is not storage identity, data dependency, or source ABI slot proof','bool input shape is not the value of an active mask, nor its sum','gating output expert IDs or distribution, complete request type and historical branch remain unbound','same-stream numerical order does not prove causal edge or graph replay behavior']},indent=2,ensure_ascii=False))
