#!/usr/bin/env python3
"""Validate D-side shape-matched observational summary; not an A/B test."""
import hashlib,json,math,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'evidence/d_shape_040/matched_distributions.json'
EXPECTED='3e910b859e4b97176b65553450df69e8721fb375f2f0bfa00acda5894c048347'
FIELDS=('kernel_name','input_shape','output_shape','input_dtype','output_dtype')


def require(cond,msg):
 if not cond:raise ValueError(msg)


def verify():
 raw=SOURCE.read_bytes();require(hashlib.sha256(raw).hexdigest()==EXPECTED,'source SHA mismatch');d=json.loads(raw)
 require(d['rank_count']==16 and d['all_signature_counts_match'] is True and d['count_mismatch_count']==0,'rank/coverage failure')
 by={(r['engine'],r['rank']):r for r in d['ranks']};require(len(by)==len(d['ranks'])==16,'duplicate rank')
 pairs=[]
 for rank in range(8):
  a=by[('sg',rank)];b=by[('vllm',rank)];lookup={tuple(x[k] for k in FIELDS):x for x in b['signatures']};require(len(lookup)==15 and len(a['signatures'])==15,'signature count')
  for x in a['signatures']:
   y=lookup[tuple(x[k] for k in FIELDS)];require(x['count_match'] is True and y['count_match'] is True,'count mismatch')
   for item in (x,y):
    z=item['duration_us'];require(item['actual_count']==item['expected_count']==z['count'] and z['count']>0,'event count')
    require(all(isinstance(z[k],(int,float)) and not isinstance(z[k],bool) and math.isfinite(z[k]) and z[k]>=0 for k in ('sum_us','min_us','p50_us','p90_us','p99_us','max_us','mean_us')),'invalid duration')
    require(z['min_us']<=z['p50_us']<=z['p90_us']<=z['p99_us']<=z['max_us'],'quantile order')
   pairs.append((rank,x,y))
 require(len(pairs)==120,'pair count')
 n=[(x,y) for rank,x,y in pairs if rank==0 and x['kernel_name']=='MoeGatingTopK'];require(len(n)==1,'reference signature');x,y=n[0]
 require(x['actual_count']==8 and y['actual_count']==360 and abs(x['duration_us']['p50_us']-5.22)<1e-6 and abs(y['duration_us']['p50_us']-4.68)<1e-6,'reference values')
 text=raw.decode();require(not re.search(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',text),'IP-like text');require(not re.search(r'(?<!\w)/(?:home|root|data|data1|task|tmp|models)(?:/|\b)',text),'private path')
 return {'ranks':16,'matched_physical_pairs':len(pairs),'reference_rank0':{'sg_count':8,'sg_p50_us':5.22,'vllm_count':360,'vllm_p50_us':4.68},'scope':'same physical shape/type, not same request/step or causal effect'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
