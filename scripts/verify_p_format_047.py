#!/usr/bin/env python3
"""Verify P matched recorded fields/counters; not same-workload causality."""
import gzip,hashlib,json,math,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'evidence/p_formats_047/distributions.json.gz'
SHA='9ae081169cd00b9cebaa9bf2b922232c1def10d71db44ef5562da79946e15b2f'
PRIOR=ROOT/'evidence/p_joint_046/distributions.json'
PRIOR_SHA='0fdc279b5181476aee41502ae43376440ac4b6d87810b754a86bd7628881d7a3'
FIELDS=('rank','kernel_name','input_shape','output_shape','input_dtype','output_dtype')


def need(v,m):
 if not v:raise ValueError(m)


def verify():
 raw=gzip.decompress(SRC.read_bytes());old=PRIOR.read_bytes();need(hashlib.sha256(raw).hexdigest()==SHA and hashlib.sha256(old).hexdigest()==PRIOR_SHA,'source SHA')
 d=json.loads(raw);p=json.loads(old);pairs=d['format_pairs'];need(d['rank_count']==8 and d['base_joint_pairs']==112 and len(pairs)==len(d['format_coverage'])==112,'coverage')
 prior={tuple(x[k] for k in FIELDS):x for x in p['pairs']};need(len(prior)==112,'prior duplicates')
 seen=set()
 for x in pairs:
  k=tuple(x[f] for f in FIELDS);need(k not in seen and k in prior,'pair identity');seen.add(k);need(x['input_format']!='<unknown>' and x['output_format']!='<unknown>','format unavailable')
  for side in ('sg','vllm'):
   z=x[side];need(z==prior[k][side] and sum(x[side+'_stream_counts'].values())==z['count'],'previous distribution changed')
   for metric in x[side+'_metrics'].values():
    need(metric['available']+metric['unavailable']==z['count'],'counter coverage')
    need(metric['p50'] is None if metric['available']==0 else isinstance(metric['p50'],(int,float)) and math.isfinite(metric['p50']),'counter p50')
 t=sorted((x for x in pairs if x['kernel_name']=='aclnnDynamicQuantV2_DynamicQuant_DynamicQuant' and x['input_shape']=='8192,4096'),key=lambda x:x['rank']);need(len(t)==8,'target coverage')
 for x in t:
  need(x['input_format']=='ND' and x['output_format']=='ND;ND' and x['input_dtype']=='DT_BF16' and x['output_dtype']=='INT8;FLOAT','target signature');need(x['sg']['count']==120 and x['vllm']['count']==399,'target counts')
  need(all(x[s+'_metrics'][m]['available']==x[s]['count'] for s in ('sg','vllm') for m in ('aiv_time(us)','aiv_vec_time(us)','aiv_mte2_time(us)','aiv_mte3_time(us)')),'target counter missing')
 a=t[0]['sg_metrics'];b=t[0]['vllm_metrics'];need(abs(a['aiv_mte2_time(us)']['p50']-94.1335)<1e-6 and abs(b['aiv_mte2_time(us)']['p50']-15.443)<1e-6,'reference MTE2')
 text=raw.decode();need(not re.search(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',text) and not re.search(r'(?<!\w)/(?:home|root|data|data1|task|tmp|models)(?:/|\b)',text),'private strings')
 return {'ranks':8,'pairs':len(pairs),'format':'ND → ND;ND','rank0':{'sg_events':120,'vllm_events':399,'sg_vec_p50_us':a['aiv_vec_time(us)']['p50'],'vllm_vec_p50_us':b['aiv_vec_time(us)']['p50'],'sg_mte2_p50_us':a['aiv_mte2_time(us)']['p50'],'vllm_mte2_p50_us':b['aiv_mte2_time(us)']['p50']},'scope':'independent captures; raw overlapping counters; strides/values/steps and runtime versions unmatched'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
