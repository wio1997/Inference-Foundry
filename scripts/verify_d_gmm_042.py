#!/usr/bin/env python3
"""Check rank-local physical-buffer GMM distributions, not A/B causality."""
import hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'evidence/d_gmm_042/distributions.json'
SHA='e055132d8459602867055c2bbb8e37d3c8e661a96ae3338e1fcf0ff9e20d806b'
NORMAL='aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul'
FUSED='aclnnGroupedMatmulSwigluQuantWeightNZ_GroupedMatmulSwigluQuant_GroupedMatmulSwigluQuant'


def need(ok,msg):
 if not ok:raise ValueError(msg)


def verify():
 raw=PATH.read_bytes();need(hashlib.sha256(raw).hexdigest()==SHA,'source SHA mismatch');d=json.loads(raw)
 need(d['rank_count']==16 and d['all_signature_counts_match'] is True and d['count_mismatch_count']==0,'rank coverage')
 by={(r['engine'],r['rank']):r for r in d['ranks']};need(len(by)==len(d['ranks'])==16,'duplicate ranks')
 sg=[]
 for engine in ('sg','vllm'):
  for rank in range(8):
   r=by[(engine,rank)];need(r['count_validation']=='pass' and r['stable_during_hash_and_parse'] is True and r['csv_expected_sha256']==r['csv_actual_sha256'],'source integrity')
   need(len(r['signatures'])==2,'signature count');names={x['kernel_name'] for x in r['signatures']}
   need(names==({NORMAL} if engine=='sg' else {NORMAL,FUSED}),'kernel names')
   for x in r['signatures']:
    z=x['duration_us'];need(x['actual_count']==x['expected_count']==z['count']==360 and x['count_match'] is True,'count mismatch')
    need(all(isinstance(z[k],(int,float)) and math.isfinite(z[k]) and z[k]>=0 for k in ('min_us','p50_us','p90_us','p99_us','max_us')),'duration invalid')
    need(z['min_us']<=z['p50_us']<=z['p90_us']<=z['p99_us']<=z['max_us'],'quantile order')
    need(sum(s['count'] for s in x['stream_stats'].values())==360,'stream count mismatch')
   if engine=='sg':
    k={x['input_shape'].split(';')[0]:x for x in r['signatures']};need(set(k)=={'8192,2048','8192,4096'},'unexpected SG shapes')
    sg.append({'rank':rank,'p50_2048_us':k['8192,2048']['duration_us']['p50_us'],'p50_4096_us':k['8192,4096']['duration_us']['p50_us']})
 need(abs(sg[0]['p50_2048_us']-67.9315)<.001 and abs(sg[0]['p50_4096_us']-126.0515)<.001,'reference values')
 text=raw.decode();need(not re.search(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',text),'address-like string');need(not re.search(r'(?<!\w)/(?:home|root|data|data1|task|tmp|models)(?:/|\b)',text),'private path')
 return {'ranks':16,'signature_groups':32,'sg_by_rank':sg,'scope':'single 32K diagnostic; shapes do not establish equal token/expert load or causality'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
