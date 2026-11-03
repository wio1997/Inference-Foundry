#!/usr/bin/env python3
"""Reproduce a conditional BF16 output-projection model; no NPU access."""
import hashlib,json
from decimal import Decimal
from pathlib import Path
root=Path(__file__).resolve().parents[2]
files=[('agents/diagnosis/tensormove-ep8-neighbors.json','22c3cf7e9248ca2f11da46a2aac8856ac66165dde36db2c09248b6f04d71b14b'),('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl','0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4')]
for name,expected in files:
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
trace=json.loads((root/files[0][0]).read_text())
dev=next(x for x in trace['device'] if x['args']['Task Id']==43440)
flow=next(x for x in trace['flows'] if x['end']['ts']==dev['ts'])
assert flow['s_count_global']==flow['f_count_global']==1
start=Decimal(flow['start']['ts'])
for name in ('vllm::unquantized_gemm','aten::linear','aten::matmul','aclnnMatmul'):
 matches=[h for h in trace['host'] if h['name']==name and h['pid']==flow['start']['pid'] and h['tid']==flow['start']['tid'] and Decimal(h['ts'])<=start<=Decimal(h['ts'])+Decimal(h['dur'])]
 assert len(matches)==1,name
 if name=='aten::linear':
  a=matches[0]['args'];assert a['Input Dims'].replace('\r\n','').replace(' ','')=='8184,1024;5120,1024;'
  assert a['Input type'].count('c10::BFloat16')==2
rows=[]
with (root/files[1][0]).open() as f:
 for line in f:
  j=json.loads(line)
  if j.get('kind')!='rank' or j['data']['ep']!=8:continue
  for layer in j['data']['layers']:
   if layer['layer']!=0 or not layer.get('scheduler_digest','').startswith('1b826883'):continue
   for group in layer['operator_groups']:
    for t in group['tasks']:
     if str(t.get('csv',{}).get('Task ID','')).strip()=='43440': rows.append((group,t))
assert len(rows)==1
group,t=rows[0]; physical=group['physical_shape_dtype_format'];row=t['csv']
assert group['operator_name']=='aclnnMatmul_MatMulV3Common_MatMulV3'
assert physical['Input Data Types']=='DT_BF16;DT_BF16'
assert physical['Input Shapes']=='"8184,1024;5120,1024"' and physical['Output Shapes']=='"8184,5120"'
assert Decimal(row['Start Time(us)'])==Decimal(dev['ts'])
m,n,k=8184,5120,1024
work=m*n*k
capacity=24*1800*10**6*4096
model_us=Decimal(work)*Decimal(10**6)/Decimal(capacity)
sample_us=Decimal(row['Duration(us)'])
print('scope: EP8 layer 0 selected task 43440; BF16 host linear and recorded device signature')
print('conventional dense nominal MAC:',work)
print('model assumptions: direct dense work; one 24-core pool; <=1.8 GHz; <=4096 MAC/core/cycle; no in-flight credit; no other execution resources')
print('conditional compute-only lower time, us:',round(model_us,6))
print('profiled CSV task duration, us:',sample_us)
print('conditional model time / profiled duration:',round(model_us/sample_us,6))
print('not a proven hardware capacity, instruction count, measured efficiency, achievable speedup, unprofiled latency, or service TPS ceiling')
