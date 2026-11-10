#!/usr/bin/env python3
"""Charge a concrete plan representation against the same old SMLA parent model."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json,zlib,statistics
from analyze_attention_output_delivery import extract

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')
RATE=2405.3037563488215

def main():
 raw=OUT/'smla-execution-windows.json.zlib';ws=json.loads(zlib.decompress(raw.read_bytes()))
 old=json.loads((OUT/'smla-execution-parent-frontier.json').read_text());data=extract(BASE)
 assert len(ws)==2560 and sum(w['query_rows']>0 for w in ws)==1840 and old['counts']['independent_cost_groups_matched']==30
 by=defaultdict(list)
 for w in ws:by[w['dp'],w['carrier'],w['layer']].append(w)
 def price(K,r,h=0,factor=1,plan=True):
  def delta(w):
   if not w['query_rows'] or not w['ratio']:return 0.
   q=w['query_rows'];saved=(4-2/r)*K*512*2+2*64*512*4
   # 12 B/selected item = original-order64-bit address +32-bit transfer descriptor.
   # Original logical TopK and metadata outputs are retained; no old index-read credit.
   first=w['layer']==w['index_source'];extra=(K*(24+12*first)+4*(1+first))*factor if plan else 0
   return min(w['us'],(.5*saved-extra)*q/(RATE*1000)-h)
  carriers={m:0. for m in data['carrier_order']}
  for sec in data['arrival_sections']:
   for l in range(40):
    credits=[]
    for dp in [0,1]:
     selected=by[dp,sec['M'],l];assert len(selected)==8
     credits.append(max(delta(w) for w in selected))
    lo,hi=sec['layers'][l]['DP0_minus_DP1_us']
    gain=lambda d:max(d,0)-max(d-credits[0],-credits[1])
    carriers[sec['M']]+=max(gain(lo),gain(hi))/1000
  reqs=[]
  for req in data['requests']:
   credit=sum(carriers[m] for m in data['carrier_order'][:data['carrier_order'].index(req['first_eager_M'])+1])
   reqs.append(dict(**req,conditional_credit_ms=credit,percent=100*credit/req['TTFT_ms']))
  return dict(carriers_ms=carriers,requests=reqs,max_percent=max(r['percent'] for r in reqs),min_percent=min(r['percent'] for r in reqs))
 checks=[]
 for K in [128,256,512]:
  for r in [1.,.75]:
   v=price(K,r,plan=False);expected=old['conditional_values'][f'direct_and_first_partial/K{K}/sparse_rate{r}']['max_percent'];assert abs(v['max_percent']-expected)<1e-10
   checks.append(dict(K=K,sparse_rate=r,reproduced_max_percent=v['max_percent']))
 vals=[]
 for K in [128,256,512]:
  for r in [1.,.75,.5]:
   for h in [0,5,20]:
    for factor in [1,8]:vals.append(dict(K=K,sparse_rate=r,added_us_per_native_call=h,plan_charged_byte_multiplier=factor,**price(K,r,h,factor)))
 thresholds=[]
 for K in [128,256,512]:
  for h in [0,5,20]:
   for factor in [1,8]:
    lo,hi=.5,8.
    for _ in range(32):
     mid=(lo+hi)/2
     if price(K,mid,h,factor)['max_percent']>=1:hi=mid
     else:lo=mid
    thresholds.append(dict(K=K,added_us_per_native_call=h,plan_charged_byte_multiplier=factor,sparse_rate_for_one_percent=hi if price(K,hi,h,factor)['max_percent']>=1 else None))
 out=dict(status='OLD_MATCHED_PARENT_PRICE_WITH_EXPLICIT_PLAN_AND_COORDINATION_COSTS',previous_goal_turn_classification=dict(classification='progress',authoritative_local_commit='3e5f158f39edb7ae14b8406152f55ac5fc046f16',reason='Native workspace source/lifetime/conditional capacity evidence and mapped next SMLA actual entry were committed and public36-file readback verified.'),whole_chain_complete=False,every_replacement_bitwise=True,representation=dict(bytes_per_selected_item=12,fields=['64-bit address in ORIGINAL gather order','32-bit transfer descriptor/reserved flags'],original_TopK_and_metadata_kept=True,per_source_first_plan_store=True,per_consumer_two_plan_reads=True,valid_length_bytes=4),cohort=dict(windows=2560,native_calls=1840,prior_typed_groups=30),old_reference_reproduction=checks,scenarios=vals,one_percent_thresholds=thresholds,max_C8_K512_plan_payload_MiB=818*512*12/1024**2,scope=['Same old diagnostic C8 first-token1840native/2560windows/30typed groups, fixed original EP arrival and optimistic allTP max-credit grant. Not measured net, whole-C8 decode or ordinary workload.','Gross data movement50% exposure and old two-way charged-byte reference2405.303756GB/s retained; reference is NOT sparse Cube payload bandwidth. Sparse_rate remains decision-changing unknown.','New12-byte representation and source-only write/two per-layer reads/full exposed plan traffic charged at factor1 or8; factors are scenarios, NOT measured cacheline amplification. Original logical IDs and all original metadata output/FP16-BF16 boundaries retained.','h is added total non-overlapped microseconds per native call; sparse rate must include payload DMA issue cost. No new allocator/capture/last-reader/resource co-residency credit.','First-partial byte saving remains conditional on a first nonfinal tile; later FP32 read retained. Negative local deltas/carrier/request credits are not clipped; optimistic TP grant can still hide a loss on a noncritical rank.','Full primitive bit identity, ND2NZ layout/masking/padding/event qualification and true K/charged plan/DMA rates remain pending; these numbers do not admit an implementation.'],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [raw,OUT/'smla-execution-parent-frontier.json',BASE/'full-review/PROFILE-UNIT-COSTS.json']],new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'smla-address-plan-paid-price.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(reference_checks=len(checks),scenarios=len(vals),negative_requests=sum(r['conditional_credit_ms']<0 for s in vals for r in s['requests']),selected=[{k:s[k] for k in ['K','sparse_rate','added_us_per_native_call','plan_charged_byte_multiplier','max_percent','min_percent']} for s in vals if s['K'] in [256,512] and s['sparse_rate']==.75 and s['added_us_per_native_call']==5],thresholds=[r for r in thresholds if r['K'] in [256,512] and r['added_us_per_native_call']==5]),ensure_ascii=False))

if __name__=='__main__':main()
