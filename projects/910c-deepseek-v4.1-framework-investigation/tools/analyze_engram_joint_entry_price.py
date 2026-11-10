#!/usr/bin/env python3
"""Bind entire saved WKV/rotated-gate delivery before choosing smaller recipes."""
from pathlib import Path
from hashlib import sha256
import json,zlib,itertools
from collections import Counter
from analyze_o_projection_parent_price import start,dur,union
ROOT=Path('audit/execution-search')
def main():
 rawpath=ROOT/'swa-publication-parent-calls.json.zlib';hcpath=ROOT/'hc-both-delivery-price.json';raw=json.loads(zlib.decompress(rawpath.read_bytes()));hc=json.loads(hcpath.read_text());profiles=[]
 for src in raw:
  h=next(p for p in hc['profiles'] if p['ep']==src['ep']);target={r['attention_HCPre']:r for r in h['deliveries'] if r['rows']==24};byid={r['csv_row']:r['data'] for r in src['selected']};ids=sorted(byid);recs=[]
  for i in ids:
   d=byid[i]
   if not (d['Type']=='MatMulV2' and d['Input Shapes'].strip('"').split(';')==['3,6144','25600,6144']):continue
   stop=next(j for j in ids if j>i and byid[j]['Stream ID']==d['Stream ID'] and byid[j]['Type']=='HcPre');assert stop in target
   body=[j for j in ids if i<=j<stop and byid[j]['Stream ID']==d['Stream ID']];assert len(body)==38 and all(j-i<100 for j in body)
   span=float(start(byid[stop])-start(d));gemm=float(dur(d));assert span>=gemm
   recs.append(dict(WKV_csv_row=i,next_attention_HCPre=stop,main_stream=d['Stream ID'],WKV_input_shape=d['Input Shapes'],WKV_input_dtypes=d['Input Data Types'],WKV_input_formats=d['Input Formats'],native_calls=len(body),gate_calls=len(body)-1,gate_type_histogram=dict(Counter(byid[j]['Type'] for j in body[1:])),whole_WKV_to_HC_ready_us=span,WKV_us=gemm,keep_original_WKV_other_span_us=span-gemm,native_union_us=float(union([byid[j] for j in body]))))
  assert len(recs)==24
  sums={k:sum(r[k] for r in recs) for k in ['whole_WKV_to_HC_ready_us','WKV_us','keep_original_WKV_other_span_us','native_union_us']};scenarios=[]
  for scope,f,cost in itertools.product(['whole_WKV_to_HC_ready_us','WKV_us','keep_original_WKV_other_span_us'],[1,.5,.25],[0,5,20]):
   net=f*sums[scope]-len(recs)*cost;scenarios.append(dict(scope=scope,fraction_removed=f,new_nonoverlap_us_each=cost,conditional_net_us=net,conditional_window_pct=net/h['window_us']*100))
  profiles.append(dict(ep=src['ep'],window_us=h['window_us'],source_consistent_complete_deliveries=24,totals_us=sums,allfree_window_pct={k:v/h['window_us']*100 for k,v in sums.items()},scenarios=scenarios,deliveries=recs))
 result=dict(status='ENGRAM_CURRENT_WKV_ROTATED_GATE_WHOLE_ENTRY_PAID_PRICE',profiles=profiles,
  geometry=dict(original_local_SP_M=3,original_K=6144,original_N=25600,original_weight_BF16_bytes_each=25600*6144*2,target_layers=[1,14],global_lookup_rows_before_SP=24,global_row_scope='Saved targetbucket24 only; modelproducer alreadyglobal fullfeatures before sp_shard. Not allordinary/prefill families.',proposed_N_shard_width=3200,proposed_both_model_weights_logical_bytes_per_rank=2*3200*6144*2,original_both_weights_logical_bytes_per_rank=2*25600*6144*2,original_row_panel_padded_M_reference=16,proposed_eight_original_row_panels_total_M_reference=128,illustrative_L0C_for_eight_M16_N128_FP32_panels_bytes=8*16*128*4,qualification='Logical checkpoint/ND tensor geometry and illustrative tiledschedule, not original active tiling, achieved HBM traffic, physicalcapacity or implementationbit proof.'),
  candidate_relations=['Nsharded WKV consumes globalcompleteK rows alreadypresent beforeSP and needs new row-restoring AllToAll for full originalK/V gateinput. DirectSP-owner lookup removes that globalfullrowbank; their savings cannot be combined without paying a differentexchange.','Preserve each original3row panel padded16, originalK accumulation/Fixpipe and Ntiles while reorder independentpanels to reuseB. Illustrative eightpanels retain same paddedMAC/rank as originalwholeN rather than assuming a free8xcompute reduction. ActiveoriginalK order/Ntile/key unknown.','OutputAllToAll and weightslicing/capture/layoutcosts must be paid. Gate still needs originalBF16WKV, FP32block32rotation and IEEE mask math; inactive passthrough alreadyfailed oldnative and not retried.','Only WKV/Gate independentold1%fixedC8TTFT reference children remain stopped by existing matched wholefree ceilings .615615/.489683%; this entry doesnot replace with a newuserthreshold. Largerinput/owner/HC parent separate.'],
  limits=['Two mirroredload shortwindows, not originalrequest TTFT/makespan or sixordinary prediction. Old endpointqualified prices separately referenced Engram完整交付与融合判断.md.','Allfree spans are screening envelopes, not removable arithmetic. FirstkeeporiginalWKV and priced wholeparent, no tinygate development from rawtypecounts.','No actual transport/weight processing/binary key or native primitivebit admitted. Payload metadata only; rotation header alone does not prove repeatedblock values or loadedbasis.'],
  inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [rawpath,hcpath]],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
 (ROOT/'engram-joint-entry-price.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps([dict(ep=p['ep'],chains=24,allfree_window_pct=p['allfree_window_pct'],half_h5=[s for s in p['scenarios'] if s['fraction_removed']==.5 and s['new_nonoverlap_us_each']==5]) for p in profiles],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
