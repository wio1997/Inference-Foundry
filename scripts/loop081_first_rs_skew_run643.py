#!/usr/bin/env python3
"""Join first Model45 ReduceScatter arrival/finish across all8 Run611 ranks."""
from __future__ import annotations
import bisect,hashlib,json
from collections import Counter
from decimal import Decimal
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry');BASE=ROOT/'evidence/20260928_loop081_bound/run611'
OUT=ROOT/'evidence/20260928_loop081_bound/run643/first_rs_arrival.json'
MANIFEST_SHA='b51eac01dcdc9f183544d51d20040f371f5ffb23a686ded7372b2cb366c84e83'
REPLAY_SHA='9518ee739ece657c8cff467ff476d92e7cc939048945abccc1e9f7e5a009336d'
def dec(v):return Decimal(str(v).strip())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise AssertionError(msg)
def main():
 mp=BASE/'offline_parse/manifest.json';rp=BASE/'native_graph_replay.json'
 need(sha(mp)==MANIFEST_SHA and sha(rp)==REPLAY_SHA,'pinned sources')
 m=json.loads(mp.read_text());r=json.loads(rp.read_text())
 need(len(m['parsed'])==8 and len(r['rows'])==24,'all8x3')
 refs={(v['rank'],v['ordinal_in_profile']):v for v in r['rows']}
 need(set(refs)=={(rank,i) for rank in range(8) for i in range(3)},'rank/ordinal bijection')
 rows=[]
 for rank,s in enumerate(m['parsed']):
  need(s['rank']==rank,'rank order')
  p=ROOT/s['trace_view_json'];need(sha(p)==s['trace_view_sha256'],'raw SHA')
  trace=json.loads(p.read_text())
  model=[x for x in trace if x.get('ph')=='X' and x.get('args',{}).get('Model Id')==45]
  need(len(model)==16236,'model events')
  key=lambda x:(x['args']['Physic Stream Id'],x['args']['Task Id'],x['args']['Batch Id'],x['args']['Subtask Id'])
  counts=Counter(key(x) for x in model)
  need(len(counts)==5412 and set(counts.values())=={3},'static triple')
  anchors=sorted(dec(x['ts']) for x in model if key(x)[:2]==(1,0));need(len(anchors)==3,'anchors')
  buckets=[[],[],[],[]]
  for x in model:buckets[bisect.bisect_right(anchors,dec(x['ts']))].append(x)
  need(not buckets[0] and [len(v) for v in buckets[1:]]==[5412]*3,'partition')
  for i,q in enumerate(buckets[1:]):
   need({key(x) for x in q}==set(counts),'static equality')
   graph=refs[(rank,i)];first=dec(graph['first_task_start_us'])
   need(abs(min(dec(x['ts']) for x in q)-first)<Decimal('1'),'graph anchor')
   h=sorted((x for x in q if 'hcom' in x.get('name','').lower()),key=lambda x:dec(x['ts']))
   need(len(h)==260,'hcom count')
   x=h[0];need(x['name'].lower().startswith('hcom_reducescatter') and x['args']['Physic Stream Id']==0,'first RS identity')
   high=[v for v in trace if v.get('name')==f'hcom_reduceScatter__503_0_{i+1}']
   need(len(high)==1,'high-level HCCL occurrence')
   high=high[0];meta=high['args']
   need(meta['count']==49152 and meta['data_type']=='BFP16' and meta['alg_type']=='MESH-RING-NHR' and meta['model id']==45,'HCCL shape/algorithm identity')
   need(abs(dec(high['ts'])-dec(x['ts']))<Decimal('1') and abs(dec(high['dur'])-dec(x['dur']))<Decimal('1'),'high/native HCCL join')
   start=dec(x['ts']);dur=dec(x['dur']);need(Decimal(0)<start-first<Decimal('100'),'first RS early')
   previous=refs[(rank,i-1)]['last_task_end_us'] if i else None
   rows.append({'rank':rank,'ordinal':i,'graph_start_us':str(first),
                'first_rs_start_us':str(start),'first_rs_duration_us':str(dur),
                'first_rs_end_us':str(start+dur),'first_rs_start_after_graph_us':str(start-first),
                'prior_graph_end_to_this_graph_start_us':str(first-dec(previous)) if previous is not None else None,
                'hcom_high_level_name':high['name'],'hcom_local_connection_id':meta['connection_id'],
                'hcom_count':meta['count'],'hcom_dtype':meta['data_type'],'hcom_algorithm':meta['alg_type'],
                'trace_view_sha256':s['trace_view_sha256']})
 occurrences=[]
 for i in range(3):
  q=sorted((x for x in rows if x['ordinal']==i),key=lambda x:x['rank'])
  starts=[dec(x['first_rs_start_us']) for x in q];ends=[dec(x['first_rs_end_us']) for x in q]
  latest=max(q,key=lambda x:dec(x['first_rs_start_us']))
  occurrences.append({'ordinal':i,'first_rs_arrival_spread_us':str(max(starts)-min(starts)),
   'first_rs_finish_spread_us':str(max(ends)-min(ends)),
   'latest_arrival_rank':latest['rank'],'latest_arrival_duration_us':latest['first_rs_duration_us'],
   'first_rs_start_minus_graph_start_range_us':[str(min(dec(x['first_rs_start_after_graph_us']) for x in q)),str(max(dec(x['first_rs_start_after_graph_us']) for x in q))]})
 result={'status':'all8_three_occurrence_first_rs_arrival_join_pass',
  'scope':'Run611 observer-perturbed same W0. Common timestamp alignment is supported by about12-21us collective finish spread but not an independently calibrated absolute cross-rank clock. Arrival skew and short latest-rank duration indicate peer-wait contamination; prior-work dependency and removable scheduling delay remain unknown. No Product TPS transfer.',
  'manifest_sha256':sha(mp),'replay_sha256':sha(rp),'rows':rows,'occurrences':occurrences,
  'framework_only_whole_product_tps_bound':None}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'occurrences':occurrences}))
if __name__=='__main__':main()
