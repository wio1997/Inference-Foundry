#!/usr/bin/env python3
"""Check P full recorded input-shape comparison, not cross-framework causality."""
import gzip,hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'evidence/p_shapes_041/fullshape_summary.json.gz'
SHA='e1b4e1faef1c460d7688c65f1ea69f5177ae81594e3938d1ef9b2f0362aa27c7'


def need(ok,msg):
 if not ok:raise ValueError(msg)


def verify():
 raw=gzip.decompress(PATH.read_bytes());need(hashlib.sha256(raw).hexdigest()==SHA,'source sha')
 doc=json.loads(raw);need(doc['source_sha256']=='3b5a99bf6882e2c8c13f1c9e5540528ccff429758e2509117ad5d9b2430b6622','source identity')
 ranks=doc['rank_pairs'];need(len(ranks)==8 and {r['rank'] for r in ranks}==set(range(8)),'rank set')
 for r in ranks:
  for name in ('gmm_exact','indexer_exact','sparse_attention_exact'):
   f=r['focus'][name];x=f['comparison'];need(x['common_recorded_shape_count']==0 and x['sg']['unrecorded_or_unsafe_events']==0 and x['vllm']['unrecorded_or_unsafe_events']==0,'recorded field comparison')
  f=r['focus']['gmm_exact'];need(f['sg']['empty_tensor_slot_events']==688 and f['vllm']['empty_tensor_slot_events']==360,'GMM empty slot scope')
  f=r['focus']['sparse_attention_exact'];need(f['vllm']['empty_tensor_slot_events']==112,'attention empty slot scope')
  f=r['focus']['sg_kda'];need(f['sg']['recorded_events']==0 and f['sg']['event_count']==272,'KDA unknown scope')
 totals=doc['rank_aggregated_event_counts_not_independent_trials'];need(totals['same_exact_kernel_names_across_rank_pairs']==3339 and totals['same_name_kernels_with_shared_recorded_shape']==336,'coverage totals')
 text=raw.decode();need(not re.search(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',text),'address-like string');need(not re.search(r'(?<!\w)/(?:home|root|data|data1|task|tmp|models)(?:/|\b)',text),'path-like string')
 return {'ranks':8,'rank_kernel_same_name':3339,'shared_recorded_shape_name_groups':336,'focus_literal_shape_intersection':0,'scope':'physical field strings only; empty slots, dtype, workload and steps not fully known'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
