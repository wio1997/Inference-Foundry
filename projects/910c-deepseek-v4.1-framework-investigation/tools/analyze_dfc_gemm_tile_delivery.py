#!/usr/bin/env python3
"""Bind routed GMM receiver geometry and whole common DFC suffix, no device work."""
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys,zlib
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104');OUT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'))
from price_prefill_expert_cuts import latest_difference


def stats(v):return dict(n=len(v),min=min(v),median=median(v),max=max(v))

def main():
 pins={}
 def read(p):
  p=Path(p);p=p if p.is_absolute() else BASE/p;raw=p.read_bytes();pins[str(p)]=sha256(raw).hexdigest();return json.loads(zlib.decompress(raw) if p.suffix=='.zlib' else raw)
 native=read('.cache/first-token-v168-protocol-bodies.json.zlib')
 popreceipt=read('full-review/runs/prefill-f21-populations.json');pop=read(popreceipt['cache']);assert pins[str(BASE/popreceipt['cache'])]==popreceipt['sha256']
 hist=read('full-review/PREFILL-F21-READINESS.json');arrival=read('full-review/PREFILL-OA-JOINT-ARRIVAL.json');units=read('full-review/PROFILE-UNIT-COSTS.json')['signatures']
 moe=json.loads(zlib.decompress((OUT/'moe-delivery-windows.json.zlib').read_bytes()));pins[str(OUT/'moe-delivery-windows.json.zlib')]=sha256((OUT/'moe-delivery-windows.json.zlib').read_bytes()).hexdigest()
 shared={(w['dp']*8+w['tp'],w['carrier'],w['layer']):w['shared_pipeline_native_us'] for w in moe['windows']}
 populations={(p['rank'],r[0],r[1]):r for p in pop for r in p['calls']}
 joins={(c['M'],v['layer']):v for c in hist['cohorts'] for v in c['layers']}
 rows=[];groups=defaultdict(list);stage=defaultdict(dict)
 for p in native['profiles']:
  rank=p['DP']*8+p['TP'];regs=p['regimes'];assert p['columns'][7]=='global_task_id'
  for body in p['bodies']:
   by={r[0]:r for r in body['rows']};m=body['F21_carrier_rows']
   for layer,task in enumerate(body['F21_TASK_ROWS']):
    r=by[task];g=regs[r[-1]];assert g[1]=='DispatchFFNCombine'
    pr=populations[rank,m,layer];counts=pr[-1];assert len(counts)==24 and pr[8]==65536
    total=sum(counts);join=joins[m,layer];assert total==join['receiver_rows_per_rank'][rank] and task==join['native_task_rows'][rank]
    assert total<=pr[8]
    w=dict(rank=rank,dp=p['DP'],tp=p['TP'],carrier=m,layer=layer,task=task,start_ns=r[1],end_ns=r[2],DFC_us=(r[2]-r[1])/1000,
      receiver_rows=total,counts=counts,first21_rows=sum(counts[:21]),last3_rows=sum(counts[21:]),premature_wait_group20_rows=counts[20],active_experts=sum(v>0 for v in counts),
      logical_weight_request_bytes=sum((v+127)//128 for v in counts)*5120*(4608+2304),
      FP16_GMM1_publish_read_bytes=total*4608*2*2,
      activation_codes_write_bytes=total*2304,
      GMM2_A_logical_requests_bytes=total*2304*20,
      GMM1_A_logical_requests_bytes=total*5120*18,
      pertoken_scale2_write_bytes=total*4,
      shared_pipeline_us=shared[rank,m,layer])
    rows.append(w);stage[m,layer][rank]=w;groups[p['DP'],m,tuple(g)].append(w['DFC_us'])
 assert len(rows)==2560
 matched=[]
 for (dp,m,g),vs in groups.items():
  u,=[x for x in units if x['phase']=='eager_first_token' and tuple(x[k] for k in ['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes'])==g]
  c,=[x for x in u['conditions'] if (x['dp'],x['carrier'])==(dp,m)];assert c['calls']==len(vs)==320
  for key,val in [('min',min(vs)),('median',median(vs)),('max',max(vs))]:assert abs(c['observed_us'][key]-val)<1e-7
  matched.append(dict(dp=dp,carrier=m,calls=len(vs),signature=list(g),DFC_us=stats(vs)))
 offsets=arrival['clock_contract']['offset_upper_ns'];assert all(offsets[i][j]<=offsets[i][k]+offsets[k][j] for i in range(16) for j in range(16) for k in range(16))
 stages=[]
 for (m,layer),rs in sorted(stage.items()):
  assert sorted(rs)==list(range(16));start={i:w['start_ns'] for i,w in rs.items()};end={i:w['end_ns'] for i,w in rs.items()}
  bound=latest_difference(end,start,offsets);assert 0<=bound[0]<=bound[1]
  stages.append(dict(carrier=m,layer=layer,common_DFC_suffix_us=[v/1000 for v in bound],
    max_local_DFC_us=max(w['DFC_us'] for w in rs.values()),
    max_shared_pipeline_us=max(w['shared_pipeline_us'] for w in rs.values()),
    max_FP16_pair_bytes=max(w['FP16_GMM1_publish_read_bytes'] for w in rs.values())))
 order=[s['M'] for s in arrival['sections']];oldreq=[dict(slot=r['slot'],old_ms=r['TTFT_ms'],last=r['first_eager_M']) for r in arrival['requests']];old=max(r['old_ms'] for r in oldreq)
 def price(f=1.,rho=1.,h=0.,endpoint=1,kind='common_suffix',rate_GBs=2405.3037563488215):
  carriers={m:0. for m in order}
  for s in stages:
   if kind=='common_suffix':credit=f*rho*s['common_DFC_suffix_us'][endpoint]-h
   elif kind=='FP16_pair':credit=rho*s['max_FP16_pair_bytes']/(rate_GBs*1000)-h
   elif kind=='shared_reserve4':credit=rho*s['max_shared_pipeline_us']-.2*f*s['common_DFC_suffix_us'][endpoint]-h
   else:raise AssertionError(kind)
   # Reserving cores across the whole plan must retain losses. Other scoped
   # hypothetical replacements grant a free per-layer fallback, optimistically.
   carriers[s['carrier']]+=(credit if kind=='shared_reserve4' else max(credit,0))/1000
  req=[dict(slot=r['slot'],old_ms=r['old_ms'],conditional_credit_ms=sum(carriers[m] for m in order[:order.index(r['last'])+1])) for r in oldreq]
  saved=old-max(r['old_ms']-r['conditional_credit_ms'] for r in req)
  return dict(kind=kind,common_critical_stage_fraction=f,removable_or_hidden_fraction=rho,added_us_per_layer=h,clock_endpoint=endpoint,reference_rate_GBs=rate_GBs if kind=='FP16_pair' else None,P90_saved_ms=saved,P90_percent=100*saved/old,requests=req)
 values=[price(f,r,h,e) for f,r,h in [(1,1,0),(.25,.5,5),(.1,.5,5),(.1,.5,20)] for e in [0,1]]
 byte_values=[price(rho=r,h=h,kind='FP16_pair',rate_GBs=b) for b in [500,1200,2405.3037563488215] for r,h in [(1,0),(.5,5)]]
 reserve=[price(f,r,5,kind='shared_reserve4') for f in [0,.25,.5,1] for r in [1,.5]]
 thresholds=[]
 for r,h in [(.5,5),(.5,20),(.75,5),(.75,20)]:
  ends=[]
  for e in [0,1]:
   lo,hi=0.,1.
   for _ in range(60):
    mid=(lo+hi)/2
    if price(mid,r,h,e)['P90_percent']>=1:hi=mid
    else:lo=mid
   ends.append(hi if price(1,r,h,e)['P90_percent']>=1 else None)
  thresholds.append(dict(removable_fraction=r,added_us_per_layer=h,required_common_critical_stage_fraction_by_clock_endpoint=ends))
 population_summary=dict(windows=2560,receiver_rows=stats([w['receiver_rows'] for w in rows]),maximum_single_expert_rows=max(max(w['counts']) for w in rows),zero_last3=sum(w['last3_rows']==0 for w in rows),last3_and_group20_nonempty=sum(w['last3_rows']>0 and w['premature_wait_group20_rows']>0 for w in rows),first21_rows=stats([w['first21_rows'] for w in rows]),last3_rows=stats([w['last3_rows'] for w in rows]))
 alias_witnesses=[]
 for w in rows:
  boundary=w['first21_rows'];tail=w['last3_rows']
  # Original equal K-stride C1/C2 regions meet at boundary*K; compactN changes
  # that boundary and can expose an earlier GMM2 writer to unread batch2 data.
  if not tail:continue
  first_unsafe_output_row=boundary*4608//5120
  start_group20=sum(w['counts'][:20])
  if first_unsafe_output_row < start_group20:
   alias_witnesses.append(dict(rank=w['rank'],carrier=w['carrier'],layer=w['layer'],first21_rows=boundary,last3_rows=tail,
    first_compact_tail_element=boundary*4608,first_GMM2_row_overlapping_compact_unread_tail=first_unsafe_output_row,group20_start_row=start_group20,
    condition='Only changing C1 to compactN stride lets an earlier GMM2 output overwrite unread batch2 C1 even before original second V2C wait. Original equalK stride keeps batch row domains separate. Element offsets shown for FP16 bank; this is an address counterexample, not an observed bit mismatch.'))
 resources=dict(L1_bytes=2*(128*512+256*512)+256*8,L0A_bytes=2*128*128,L0B_bytes=2*128*256,L0C_bytes=128*256*4,UB_base_double_stage_bytes=2*12*4608,scope='Selected source allocation formulas only, not measured dynamic occupancy or co-residency; L1 scale actual single stage while assertion conservatively counts2.',single_C1_C2_alias_bank_MiB=65536*5120*2/2**20,single_A_Activation_alias_bank_MiB=65536*5120/2**20,global_sender_bound_decode3=16*3*6,decode3_two_payload_banks_at_bound_MiB=(16*3*6)*5120*3/2**20)
 result=dict(status='BOUNDED_NATIVE_DFC_TILE_ACTIVATION_READERS_RESOURCE_CORRECTION_AND_WHOLE_VALUE',whole_chain_complete=False,every_replacement_bitwise=True,windows=rows,stages=stages,matched_cost_conditions=matched,populations=population_summary,source_resource_formula=resources,
  conditional_values=values,FP16_nominal_byte_scenarios=byte_values,shared_resource_reservation_scenarios=reserve,one_percent_stage_thresholds=thresholds,
  integer_partial_sum_bound=dict(GMM1=5120*128*128,GMM2=2304*128*128,INT32_limit=2**31-1,qualification='Exact integer-math worst-case sum bound only; does not prove new DMA/tiling/Fixpipe/FP16/vector primitive bit identity.'),
  alias_geometry=dict(original_C1_C2_same_K_stride=True,original_A_Activation_same_allocation=True,compact_C1_only_has_new_capacity_credit_bytes=0,
   compact_stride_counterexample_windows=len(alias_witnesses),first_witnesses=alias_witnesses[:4],
   wait_boundary='For original equalK bank and original row domains, group20 reads only batch1 codes/scale and writes only batch1 C2 rows. Moving second V2C wait from beforegroup20 to beforegroup21 retains two flag consumptions but still needs actual DMA/Fixpipe/primitive bit/liveness gates.',
   missing_time='min(exposed positive secondV2C wait at group20, same original group20 GMM2 time) minus new cost; populations alone give no phase time.'),
  source_pins=[dict(path=p,sha256=h) for p,h in pins.items()],
  scope='Four original diagnostic eager bodies,2560 matched source/runtime-rank tasks and exact24-expert receiver counts. Whole common suffix is lastDFCend minus lastDFCstart under archived clock contract, includes protocol/compute/return, not GMM or activation phase time. Fixed external eager/request ordering, grant uniform savings; not expected/causal request improvement or arbitrary-schedule ceiling. Byte quantities are logical GM requests not observed HBM traffic; byte rate is a separate transfer reference and cannot establish current effective DFC phase bandwidth.',
  analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[])
 (OUT/'dfc-gemm-tile-profile.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(matched=len(matched),population=population_summary,resource=resources,common_suffix_prices=[{k:v for k,v in x.items() if k!='requests'} for x in values],byte_prices=[{k:v for k,v in x.items() if k!='requests'} for x in byte_values],thresholds=thresholds,shared_reserve=[{k:v for k,v in x.items() if k!='requests'} for x in reserve]),ensure_ascii=False))

if __name__=='__main__':main()
