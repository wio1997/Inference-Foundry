#!/usr/bin/env python3
"""Verify 32K P DynamicQuant post-hoc partitions and same-stream contexts."""
import gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'evidence/p_quant_048'
TEMP_SHA='7bf47e03562583b86d925471b392872028654cbcb52fc2743c8f01515d917aae'
NEIGHBOR_SHA='5b1996a40250ddcfed8d6414517c86da29f1381826b872227fd4b6addcae4dc7'


def need(c,m):
 if not c:raise ValueError(m)


def load(n,sha):
 b=gzip.decompress((ROOT/n).read_bytes());need(hashlib.sha256(b).hexdigest()==sha,'source hash');return json.loads(b)


def verify():
 t=load('temporal.json.gz',TEMP_SHA);n=load('neighbors.json.gz',NEIGHBOR_SHA)
 tt={(x['framework'],x['rank']):x for x in t['ranks']};nn={(x['framework'],x['rank']):x for x in n['ranks']};want={(e,r) for e in ('sglang','vllm') for r in range(8)}
 need(set(tt)==set(nn)==want and n['temporal_source_sha256']==TEMP_SHA,'rank identity')
 canonical={}
 for k in sorted(want):
  e,rank=k;x=tt[k]['temporal'];y=nn[k]['neighbors'];total=120 if e=='sglang' else 399;lo=40 if e=='sglang' else 399;hi=80 if e=='sglang' else 0
  need(x['count']==total and x['threshold_us']==100 and x['at_or_below_threshold']['count']==lo and x['above_threshold']['count']==hi,'partitions')
  need(x['slice_count']==(5 if e=='sglang' else 7) and sum(s['count'] for s in x['slices'])==total,'slice coverage')
  for label,count in [('at_or_below_threshold',lo),('above_threshold',hi)]:
   part=y['partitions'][label];need(part['count']==count and sum(c['count'] for c in part['contexts'])==count,'context counts')
   for metric in x[label]['metrics'].values():need(metric['available']+metric['unavailable']==count,'counter count')
  signature=tuple((label,tuple((c['count'],c['previous_name'],c['previous_input_shape'],c['next_name'],c['next_input_shape']) for c in y['partitions'][label]['contexts'])) for label in ('at_or_below_threshold','above_threshold'))
  if e not in canonical:canonical[e]=signature
  need(canonical[e]==signature,'context patterns vary by rank')
 s=tt[('sglang',0)]['temporal'];low=s['at_or_below_threshold'];high=s['above_threshold']
 need(abs(low['duration_us']['p50_us']-89.032)<1e-6 and abs(high['duration_us']['p50_us']-123.0025)<1e-6,'duration p50')
 need(abs(low['metrics']['aiv_mte2_time(us)']['p50']-14.573)<1e-6 and abs(high['metrics']['aiv_mte2_time(us)']['p50']-94.6795)<1e-6,'conditional MTE2')
 return {'ranks':16,'sg_per_rank_fast':40,'sg_per_rank_slow':80,'vllm_per_rank_fast':399,'vllm_per_rank_slow':0,
         'rank0_sg_fast_slow_duration_p50_us':[low['duration_us']['p50_us'],high['duration_us']['p50_us']],
         'rank0_sg_fast_slow_mte2_p50_us':[low['metrics']['aiv_mte2_time(us)']['p50'],high['metrics']['aiv_mte2_time(us)']['p50']],
         'same_stream_contexts_consistent_by_rank':True,'scope':'100us is post-hoc; local adjacent events do not prove a producer or layer; no request pairing'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
