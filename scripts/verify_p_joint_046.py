#!/usr/bin/env python3
"""Verify P recorded shape/type event distributions, not matched workload."""
import hashlib,json,math,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'evidence/p_joint_046/distributions.json'
SHA='0fdc279b5181476aee41502ae43376440ac4b6d87810b754a86bd7628881d7a3'


def need(v,msg):
 if not v:raise ValueError(msg)


def verify():
 raw=SRC.read_bytes();need(hashlib.sha256(raw).hexdigest()==SHA,'source SHA mismatch');d=json.loads(raw)
 need(d['rank_count']==8 and d['input_groups_per_rank']==15 and len(d['group_coverage'])==120 and len(d['pairs'])==112,'coverage')
 keys=set()
 for x in d['pairs']:
  key=(x['rank'],x['kernel_name'],x['input_shape'],x['output_shape'],x['input_dtype'],x['output_dtype']);need(key not in keys and x['rank'] in range(8),'duplicate pair');keys.add(key)
  need(all(v and v!='<unknown>' for v in key[2:]),'unknown signature')
  for side in ('sg','vllm'):
   z=x[side];need(type(z['count']) is int and z['count']>0,'bad count')
   need(all(isinstance(z[k],(int,float)) and math.isfinite(z[k]) and z[k]>=0 for k in ('min_us','p50_us','p90_us','p99_us','max_us')),'bad duration')
   need(z['min_us']<=z['p50_us']<=z['p90_us']<=z['p99_us']<=z['max_us'],'quantile order')
   need(sum(x[side+'_stream_counts'].values())==z['count'],'stream coverage')
 for rank in range(8):need(sum(x['rank']==rank for x in d['pairs'])==14,'rank pair count')
 rows=sorted((x for x in d['pairs'] if x['kernel_name']=='aclnnDynamicQuantV2_DynamicQuant_DynamicQuant' and x['input_shape']=='8192,4096'),key=lambda x:x['rank']);need(len(rows)==8,'target rank coverage')
 for x in rows:
  need(x['input_dtype']=='DT_BF16' and x['output_shape']=='8192,4096;8192' and x['output_dtype']=='INT8;FLOAT' and x['sg']['count']==120 and x['vllm']['count']==399,'target signature')
 text=raw.decode();need(not re.search(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',text) and not re.search(r'(?<!\w)/(?:home|root|data|data1|task|tmp|models)(?:/|\b)',text),'private output')
 return {'ranks':8,'rank_input_groups':120,'complete_joint_pairs':112,'dynamic_quant_rank0':{'sg_count':120,'sg_p50_us':rows[0]['sg']['p50_us'],'vllm_count':399,'vllm_p50_us':rows[0]['vllm']['p50_us']},'rank_p50_medians_us':{'sg':statistics.median(x['sg']['p50_us'] for x in rows),'vllm':statistics.median(x['vllm']['p50_us'] for x in rows)},'scope':'recorded shape/dtype only; independent captures, formats/steps/data unpaired'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
