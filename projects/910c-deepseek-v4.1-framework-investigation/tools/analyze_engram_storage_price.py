#!/usr/bin/env python3
"""Whole-UVA erase screen before any compact-scale native development."""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict,Counter
import sys,json,re,statistics

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104');OUT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'))
from price_embedding_engram_fusion import records,bindings,load,SPANS,ENDPOINTS,INPUTS
from price_terminal_head_delivery import request_price

def digest(p):return sha256(p.read_bytes()).hexdigest()
def main():
 saved=BASE/'full-review/EMBEDDING-ENGRAM-FUSION-ESTIMATES.json';doc=load(saved)
 raw=BASE/'.cache/v168-engram-delivery-on106.json';assert digest(raw)==doc['sources'][str(raw.relative_to(BASE))]
 delivery=BASE/'full-review/V168-ENGRAM-DELIVERY.json';dv=load(delivery);assert digest(raw)==dv['input_sha256'][raw.name]
 waves,hist,sched,_,_=records();events,counts=bindings(waves,load(SPANS));ends=load(ENDPOINTS)['endpoint_ms'];refs={k:max(v) for k,v in ends.items()};assert refs==doc['C8_reference_ms']
 rows=[];groupmax=[]
 for n,rs in waves.items():
  for dp in [0,1]:
   if not rs[8*dp].get('scheduled'):continue
   for layer in ['L1','L14']:
    cut=[events[n][layer][r] for r in range(dp*8,dp*8+8)];t,={r[0] for r in cut};dur=[(r[3][2]-r[3][1])/1000 for r in cut];assert all(v>=0 for v in dur)
    rows.append(dict(wave=n,dp=dp,kind=rs[8*dp]['kind'],layer=layer,global_tokens=t,lookup_us_by_TP=dur,lookup_TASK_ROWS=[r[3][0] for r in cut]));groupmax.append(max(dur))
 assert len(rows)==906 and len(rows)*8==dv['matched_local_deliveries']==7248
 def price(f,h):
  credits={}
  for n,rs in waves.items():
   costs={}
   for dp in [0,1]:
    costs[dp]={}
    if not rs[8*dp].get('scheduled'):continue
    for layer in ['L1','L14']:
     dur=[(events[n][layer][r][3][2]-events[n][layer][r][3][1])/1000 for r in range(dp*8,dp*8+8)];net=[v*f-h for v in dur];costs[dp][layer]=[min(net),max(net)]
   stages=set().union(*(d.keys() for d in costs.values()))
   credits[n]=[sum(fn(costs[d].get(k,[0.,0.])[side] for d in [0,1]) for k in stages) for side,fn in enumerate([min,max])]
  v=request_price(credits,waves,hist,sched,ends)
  return dict(erased_fraction=f,added_us_per_lookup=h,conditional_endpoint_credit_ms=v,conditional_endpoint_percent={k:[100*x/refs[k] for x in pair] for k,pair in v.items()},negative_wave_credits=sum(any(x<0 for x in pair) for pair in credits.values()))
 scenarios=[price(f,h) for f in [1.,1/12,1/24,0.] for h in [0,1,5]]
 # Source-independent cache-line traffic scenarios, not observed host/SIO transactions.
 cache=[]
 for line in [32,64,128]:
  old=(256+line-1)//line*line+(32+line-1)//line*line;new=(256+line-1)//line*line+(8+line-1)//line*line
  cache.append(dict(assumed_line_B=line,distinct_cold_row_separate_banks_old_B=old,new_B=new,saved_B=old-new,adjacent_rows_or_crossrow_cache_reuse='Not included; may change traffic.'))
 cat=load(OUT/'engram-storage-catalog.json');capacity=[]
 for v in cat['checkpoint_headers']:
  assert v['weight']['dtype']=='I8' and v['scale']['dtype']=='F8_E8M0';n,d=v['weight']['shape'];assert d==256 and v['scale']['shape']==[n,8]
  capacity.append(dict(layer=v['layer'],rows=n,old_scale_B=n*8*4,new_scale_B=n*8,saved_requested_shared_host_B=n*8*3))
 assert sum(r['saved_requested_shared_host_B'] for r in capacity)==18432548400
 pins=[raw,delivery,saved,Path(SPANS),Path(ENDPOINTS),Path(INPUTS),OUT/'engram-storage-catalog.json']+[BASE/'tools'/n for n in ['price_embedding_engram_fusion.py','price_terminal_head_delivery.py','price_vocab_selection_fusion.py']]
 out=dict(status='MATCHED_WHOLE_LOOKUP_ERASE_SCREEN_BEFORE_COMPACT_SCALE_NATIVE_WORK',previous_goal_turn_classification=dict(classification='progress',authoritative_local_commit='2729de670e52dc2f55103b8e96793beddd1fddd1',public_commit='70b68809c1443f3372455812d3701941f010dc7a',reason='SMLA original-order source/CPU/jointprice changed nextaction,78local paths and48public paths exactreadback; no newprimitive or full-chain completion.'),whole_chain_complete=False,every_replacement_bitwise=True,cohort=dict(original_C8='32768in/1024out,8requests,TP8/DP2,originalfixed schedule',waves=len(waves),Engram_TP8_groups=len(rows),local_native_lookups=7248,counts=counts,reference_endpoint_ms=refs,source_scope='Current11source SHA pins and checkpoint header guard identify proposedI8+E8M0 route. Old native task links identify lookup duration; no percallloaded dtype/binary attestation.'),all_groups_serial_sum_of_max_TP_lookup_ms=sum(groupmax)/1000,group_lookup_records=rows,scenarios=scenarios,cache_line_scenarios=cache,capacity=dict(tables=capacity,saved_requested_shared_host_B=18432548400,saved_requested_shared_host_GiB=18432548400/1024**3,scope='Code unchanged. One physical shared host copy acrossDP perTPshard, count once; dtype changes requested/pinned shared-host scale payload, not newNPU HBM/KV pool capacity or measured freed residentpages. Pointer bank count by CHUNK_ROWS unchanged.'),decision=dict(stopped='Under same old1%fixed-latency budget, stop standalone compactscale/groupbroadcast/lookup-only kernel development if even all oldUVA lookup erased remains low. No new native bit/perf experiment solely for that small recipe.',pending='Hostcapacity/changedNUMA or larger input-owner/WKV/iteration joint representation remains separate; need a concrete capacity/readiness mechanism that changes go/no-go before native qualification. Low lookup price is not a global Engram/input upper bound.'),limits=['Conditional fixed-order original endpoint model gives best/worstTP/DP bracket, retains original request deadlines and other stages; not measured gain, universal resource-schedule ceiling or causal P90 claim.','f=1 removes entire lookup including requiredcode loads/dequant/stores at zero replacement cost; f=1/12 is a byte-proportional288->264 sensitivity only, not a bound on physical cacheline traffic or latency. h is fully exposed extra cost and negative wave credits retained.','Collective readiness waits are kept, not priced as erased lookup bytes. Changedhostsubmission/communication/owner/WKV/weights are distinct larger scopes; no adding oldcandidate credits.','No initial checkpoint/registration/coldload time credited to warm requests. No scale payload read, no actual compact allocation, no NPU/CPU Torch import and no newprofile/service/request/model work.','Sixordinary cases and any largerbatch/hostcapacity effect are not inferred from originalC8. Keep exact source output and perprimitive bitwise requirement for any replacement admitted later.'],input_pins=[dict(path=str(p),sha256=digest(p)) for p in pins],analyzer_sha256=digest(Path(__file__)),new_NPU_work=0,new_profile_collection=0,service_requests=0,model_loads=0,remote_writes=[])
 (OUT/'engram-storage-price.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(groups=len(rows),lookups=len(rows)*8,serial_max_ms=sum(groupmax)/1000,reference_ms=refs,selected=[s for s in scenarios if s['added_us_per_lookup']==0],capacity=out['capacity']['saved_requested_shared_host_GiB']),ensure_ascii=False))
if __name__=='__main__':main()
