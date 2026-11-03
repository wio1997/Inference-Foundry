#!/usr/bin/env python3
"""Compare four selected, same-stream profiled tasks across two row shapes; CPU only."""
import hashlib,json,re,statistics
from decimal import Decimal
from pathlib import Path
R=Path(__file__).resolve().parents[2];p=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
h=hashlib.sha256(p.read_bytes()).hexdigest();assert h=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
D={}
for z in p.open():
 d=json.loads(z)
 if d['kind']!='rank' or d['data']['ep'] not in range(8):continue
 for l in d['data']['layers']:
  pref=l['scheduler_digest'][:12]
  if l['layer'] not in (0,12,24,36) or pref not in ('7ddf7ae98a00','db852bff987d'):continue
  T=595 if pref=='7ddf7ae98a00' else 1023
  q='aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3'
  patterns={'dynamic_quant':('aclnnDynamicQuantV2_DynamicQuant_DynamicQuant',f'"{T},5120"'),'projection_1':(q,f'"{T},5120;144,320,16,32;4608"'),'activation_quant':('DequantSwigluQuant',f'"{T},4608;4608;{T};;;;"'),'projection_2':(q,f'"{T},2304;160,144,16,32;5120;{T}"')}
  tasks={}
  for k,(name,shape) in patterns.items():
   gs=[g for g in l['operator_groups'] if g['operator_name']==name and g['physical_shape_dtype_format']['Input Shapes']==shape]
   assert len(gs)==1 and len(gs[0]['tasks'])==1,(d['data']['ep'],l['layer'],T,k)
   t=gs[0]['tasks'][0]; c=t['csv']; sig=gs[0]['physical_shape_dtype_format']; assert c['Name']==name and all(c[k]==v for k,v in sig.items());dur=Decimal(t['trace_duration_us']);start=Decimal(t['device_start_us']);assert dur>0
   tasks[k]={'duration_us':str(dur),'start_us':str(start),'stream':t['csv']['Stream ID'],'flow':t['flow_id'],'type':c['Type'],'signature':gs[0]['physical_shape_dtype_format']}
  order=list(patterns);assert len({tasks[k]['stream'] for k in order})==1
  assert all(Decimal(tasks[a]['start_us'])+Decimal(tasks[a]['duration_us'])<=Decimal(tasks[b]['start_us']) for a,b in zip(order,order[1:]))
  key=(d['data']['ep'],l['layer'],T);assert key not in D;D[key]=tasks

def normalized(x):return {k:re.sub(r'(?<![0-9])(?:595|1023)(?![0-9])','T',v) for k,v in x.items()}
pairs=[]
for ep in range(8):
 for layer in (0,12,24,36):
  a,b=D[ep,layer,595],D[ep,layer,1023]
  for k in a:
   assert a[k]['type']==b[k]['type'] and normalized(a[k]['signature'])==normalized(b[k]['signature']),(ep,layer,k,a[k]['signature'],b[k]['signature'])
  pairs.append({'ep':ep,'layer':layer,'595':a,'1023':b})
summary={}
for k in ('dynamic_quant','projection_1','activation_quant','projection_2'):
 ratios=[float(Decimal(x['1023'][k]['duration_us'])/Decimal(x['595'][k]['duration_us'])) for x in pairs]
 summary[k]={'min':min(ratios),'median':statistics.median(ratios),'max':max(ratios)}
out={'source_sha256':h,'selection':'8 EPs x 4 layers, two selected distinct scheduler digests; unique matched device records, normalized full signatures, same stream in each scope and non-overlapping order','scope_count':64,'paired_scopes':32,'summary':summary,'pairs':pairs,'limitations':['Temporal order and compatible shapes do not establish tensor or module identity.','Physical rows may differ from effective work; batches, inputs and profiler interference differ.','No causal attribution, per-request path or unprofiled performance inferred.']}
q=R/'agents/diagnosis/ffn-crossshape-chain-pairs.json';q.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'paired_scopes':len(pairs),'summary':summary,'output_sha256':hashlib.sha256(q.read_bytes()).hexdigest()},ensure_ascii=False))
