#!/usr/bin/env python3
"""Price direct selectedKV plus optional L1 tail using matched archived windows."""
from pathlib import Path
from collections import defaultdict
from hashlib import sha256
import json,zlib,statistics
from analyze_attention_output_delivery import extract

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104');OUT=Path('audit/execution-search')
RATE=2405.3037563488215
def digest(p):return sha256(p.read_bytes()).hexdigest()

def main():
 raw=OUT/'smla-execution-windows.json.zlib';windows=json.loads(zlib.decompress(raw.read_bytes()));data=extract(BASE);by=defaultdict(list)
 for w in windows:by[w['dp'],w['carrier'],w['layer']].append(w)
 assert len(windows)==2560 and sum(w['query_rows']>0 for w in windows)==1840
 def price(K,r,h=5,factor=1,tail_fraction=0,extra_instructions_per_query=0,issue_ns=0):
  local=[]
  def delta(w):
   if not w['query_rows'] or not w['ratio']:return 0.
   q=w['query_rows'];tail=min(K,128)*tail_fraction
   # Full source gather+panel delivery old4K vs newQK+PV2K-tail; first partial still conditional.
   saved=(4*K-(2*K-tail)/r)*512*2+2*64*512*4
   first=w['layer']==w['index_source'];plan=(K*(24+12*first)+4*(1+first))*factor
   # Synthetic balanced24-AIC issue model; payload-only r explicitly excludes this separate penalty.
   issue=((q+23)//24)*max(extra_instructions_per_query,0)*issue_ns/1000
   v=min(w['us'],(.5*saved-plan)*q/(RATE*1000)-h-issue);local.append(v);return v
  carriers={m:0. for m in data['carrier_order']}
  for sec in data['arrival_sections']:
   for layer in range(40):
    credit=[]
    for dp in [0,1]:
     ws=by[dp,sec['M'],layer];assert len(ws)==8;credit.append(max(delta(w) for w in ws))
    lo,hi=sec['layers'][layer]['DP0_minus_DP1_us'];gain=lambda d:max(d,0)-max(d-credit[0],-credit[1])
    carriers[sec['M']]+=max(gain(lo),gain(hi))/1000
  req=[]
  for r0 in data['requests']:
   ms=sum(carriers[m] for m in data['carrier_order'][:data['carrier_order'].index(r0['first_eager_M'])+1]);req.append(dict(**r0,conditional_credit_ms=ms,percent=100*ms/r0['TTFT_ms']))
  return dict(carriers_ms=carriers,requests=req,max_percent=max(r['percent'] for r in req),min_percent=min(r['percent'] for r in req),local_delta_us_range=[min(local),max(local)],negative_local_windows=sum(v<0 for v in local))
 old=json.loads((OUT/'smla-address-plan-paid-price.json').read_text());checks=[]
 for s in old['scenarios']:
  now=price(s['K'],s['sparse_rate'],s['added_us_per_native_call'],s['plan_charged_byte_multiplier']);assert abs(now['max_percent']-s['max_percent'])<1e-10;checks.append(now['max_percent'])
 vals=[]
 for K in [128,256,512]:
  for r in [.5,.75,1.]:
   for tail in [0.,.5,1.]:
    for h in [5,20]:
     for factor in [1,8]:vals.append(dict(K=K,payload_only_relative_rate=r,retained_tail_eligible_fraction=tail,added_nonoverlapped_us=h,plan_charged_multiplier=factor,**price(K,r,h,factor,tail)))
 issues=[]
 for count in [20,26,62,1280,2560]:
  for issue in [0,1,5,20]:issues.append(dict(K=512,payload_only_relative_rate=.75,retained_tail_eligible_fraction=1,total_new_ND2NZ_instructions_per_query=count,extra_instructions_over_old24=max(count-24,0),issue_ns_per_extra_instruction=issue,**price(512,.75,5,1,1,max(count-24,0),issue)))
 thresholds=[]
 for tail in [0,.5,1]:
  for factor in [1,8]:
   lo,hi=.1,4.
   for _ in range(32):
    mid=(lo+hi)/2
    if price(512,mid,5,factor,tail)['max_percent']>=1:hi=mid
    else:lo=mid
   thresholds.append(dict(K=512,retained_tail_eligible_fraction=tail,plan_charged_multiplier=factor,payload_only_rate_for_one_percent=hi))
 cal=BASE/'full-review/PREFILL-FFN-ENTRY-EXPOSURE.json';cv=json.loads(cal.read_text())['same_byte_copy_calibration'];assert abs(2*statistics.median(v['one_payload_GB_s'] for v in cv.values())-RATE)<1e-10
 cp=BASE/'full-review/PREFILL-OUTPUT-PROJECTION-AND-COPIES.json';copies=json.loads(cp.read_text())['all_TensorMove_signature_costs'];refs=[v for v in copies if v['dtype']=='DT_BF16' and v['logical_payload_bytes'] in [r['logical_B'] for r in cv.values()]];assert len(refs)==5
 native=BASE/'full-review/PREFILL-SELECTED-KV-NATIVE.json';nv=json.loads(native.read_text());receipt=BASE/'full-review/runs/selected-kv-a5580309fcab-result.json';assert digest(receipt)==nv['receipt_sha256'];contract=BASE/'probes/selected_kv/RUN-CONTRACT.json'
 out=dict(status='CONDITIONAL_DIRECT_L1_PLUS_TAIL_AND_ISSUE_PRICE_NOT_MEASURED',whole_chain_complete=False,every_replacement_bitwise=True,cohort=dict(old_C8_windows=2560,native_calls=1840,typed_groups=30),old_paid_scenarios_exactly_reproduced=len(checks),reference_rate=dict(total_charged_byte_GB_s=RATE,source=str(cal),source_sha256=digest(cal),BF16_TensorMove_references=refs,meaning='Median one-payload BF16 contiguous TensorMove rate x2; same old cohort. Not sparse GM-to-L1 bandwidth, not DMA instruction throughput, not an AIC calibration.'),representation=dict(bytes_per_item=12,address_bits=64,descriptor='within512-row original destination9bits, runLength1..128 encoded8bits, positive row gap<=127 encoded7bits, remaining8bits flags;12B/item reserved worst case. Row addresses remain available to reproduce pair fallback; plan group construction charged only by h/traffic scenarios, not timed.',original_TopK_metadata_retained=True),scenarios=vals,issue_sensitivity=issues,one_percent_rate_thresholds=thresholds,existing_stock_prepacked_measurement=dict(path=str(receipt),sha256=digest(receipt),contract_sha256=digest(contract),cases=nv['cases'],decision=nv['decision'],scope='Q416,G64,D512,K512,15AB/BA rounds,warm/after256MiB touch,prepaid compactdictionary,U1024/8192,ratios1/2; same stock SMLA. Preserved old whole-output bit observations apply only those fixtures and old package receipts. Stops stock GM prepacking alone at old1% budget, not direct-to-L1 or retained-tail parent.'),limits=['Request projection is the old fixed EP-arrival favorable maxTP model,50% gross exposure, conditional trueK uniform, conditional first nonfinal partial credit. No percentile label, whole-C8 decode, ordinary workload, measured/request guarantee or additive credit.','Tail eligibility is a scenario across uniformK128/256/512. FullN128 eligibility can exclude hardware tails; unsupported schedule/data-domain guards retain original path.','Sparse r is payload-only here. Issue penalty is separately charged per extraND2NZ times ceil(queryRows/24), a synthetic balanced24-AIC worst-loaded count, not an observed task mapping or issue time. h includes coordination/plan computation and reduced prefetch not otherwise timed; avoid double counting measured effective rate later.','Retain2of3KVslots and use third as work; no extra L1 allocation or fourthslot assumed. FullKV512KiB+QP256KiB cannot be claimed to fit original448KiB allocation.','Plan/cache generations must distinguish indexsource/KVsource pair, currentpage-table/cache and request/querymask domain. Zero-output/cache-only and unsupported/tail/adjacentCMP branches require exact fallback; no native implementation or bit admission.','12nativeQLI PA scenarios24..78direct,20..62retained instructions are narrow fixtures. Random allpair512 gives1536 direct/1280retained, allsingle3072/2560; these are not empirical model distributions.','Stop old stock GM prepacking alone and literal logical-order helper; retain direct+retention+firstpartial larger candidate until actualK/address-group distribution, ND2NZ padded lanes/LoadData bitwise, new events/prefetch and effective net cost decide.'],inputs=[dict(path=str(p),sha256=digest(p)) for p in [raw,OUT/'smla-address-plan-paid-price.json',OUT/'smla-l1-layout-cpu-a1/result.json',native,receipt,contract,cp]],new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[],analyzer_sha256=digest(Path(__file__)))
 (OUT/'smla-direct-l1-price.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(reproductions=len(checks),scenarios=len(vals),issue_scenarios=len(issues),K512_r075_h5_factor1=[{k:s[k] for k in ['retained_tail_eligible_fraction','min_percent','max_percent','local_delta_us_range']} for s in vals if s['K']==512 and s['payload_only_relative_rate']==.75 and s['added_nonoverlapped_us']==5 and s['plan_charged_multiplier']==1],thresholds=thresholds,issue=[{k:s[k] for k in ['total_new_ND2NZ_instructions_per_query','issue_ns_per_extra_instruction','min_percent','max_percent']} for s in issues if s['issue_ns_per_extra_instruction'] in [1,5]]),ensure_ascii=False))

if __name__=='__main__':main()
