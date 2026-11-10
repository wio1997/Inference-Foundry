#!/usr/bin/env python3
"""Price completed MoE->HC outputs, without counting an unrelated collective."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
from statistics import mean,median
import json,zlib
from analyze_o_projection_parent_price import start,dur,union,first_shape
R=Path('audit/execution-search')
def main():
 paths=[R/n for n in ['moe-output-parent-primary.json','moe-output-config.json','swa-publication-parent-calls.json.zlib','hc-early-delivery-price.json','qkv-joint-delivery-price.json']]
 primary,config=json.loads(paths[0].read_text()),json.loads(paths[1].read_text());assert config['selected']['enable_shared_expert_dp'] is True
 raw=json.loads(zlib.decompress(paths[2].read_bytes()));bindings=json.loads(paths[3].read_text());windows=json.loads(paths[4].read_text());profiles=[]
 for src in raw:
  b=next(p for p in bindings['profiles'] if p['ep']==src['ep']);w=next(p for p in windows['profiles'] if p['ep']==src['ep']);by={r['csv_row']:r['data'] for r in src['selected']};records=[]
  for item in b['deliveries']:
   pre,norm,dfc,post=[by[item[k]] for k in ['HcPre','RmsNormCast','DFC','FFNHcPost']];stream=dfc['Stream ID'];assert stream==norm['Stream ID']==post['Stream ID']
   seq=[(i,d) for i,d in sorted(by.items()) if item['DFC']<i<item['FFNHcPost'] and d['Stream ID']==stream]
   types=[d['Type'] for i,d in seq];assert types==['DynamicQuant','QuantBatchMatmulV3','DequantSwigluQuant','QuantBatchMatmulV3','ForeachAddList'],types
   quant,up,act,down,add=[d for i,d in seq];assert first_shape(quant)=='3,5120' and first_shape(up)=='3,5120' and first_shape(down)=='3,2304' and first_shape(add)=='3,5120'
   assert '4608' in up['Output Shapes'] and first_shape(post)=='1,3,5120'
   end=start(post)+dur(post);dfc_end=start(dfc)+dur(dfc);down_end=start(down)+dur(down)
   assert start(add)>=down_end and start(post)>=start(add)+dur(add) and start(quant)>=dfc_end
   records.append(dict(global_rows=item['rows'],main_stream=stream,source_rows={**item,'shared_DynamicQuant':seq[0][0],'shared_GateUp':seq[1][0],'shared_ActQuant':seq[2][0],'shared_Down':seq[3][0],'ForeachAdd':seq[4][0]},
       norm_ready_to_Post_start_us=float(start(post)-start(norm)-dur(norm)),norm_ready_to_Post_end_us=float(end-start(norm)-dur(norm)),HcPre_start_to_Post_end_us=float(end-start(pre)),
       retained_DFC_us=float(dur(dfc)),remaining_containing_span_keep_DFC_us=float(end-start(norm)-dur(norm)-dur(dfc)),
       shared_start_to_Post_end_us=float(end-start(quant)),DFC_end_to_Post_end_us=float(end-dfc_end),Down_ready_to_Post_end_us=float(end-down_end),
       shared_quant_us=float(dur(quant)),shared_GateUp_us=float(dur(up)),shared_ActQuant_us=float(dur(act)),shared_Down_us=float(dur(down)),ForeachAdd_us=float(dur(add)),HCPost_us=float(dur(post)),
       arithmetic_native_union_us=float(union([add,post])),explicit_shared_TP_collective_in_selected_main_sequence=False))
  assert len(records)==516 and Counter(x['global_rows'] for x in records)=={24:480,20:36}
  groups=[]
  for rows in [24,20]:
   rs=[x for x in records if x['global_rows']==rows];keys=[k for k in rs[0] if k.endswith('_us')];sums={k:sum(x[k] for x in rs) for k in keys}
   groups.append(dict(global_rows=rows,calls=len(rs),sum_us=sums,median_us={k:median(x[k] for x in rs) for k in keys},window_pct={k:v/w['window_us']*100 for k,v in sums.items()},conditional_foreach_removal_keep_HCPost=[dict(added_nonoverlap_us_each=h,net_us=sums['ForeachAdd_us']-len(rs)*h,net_window_pct=(sums['ForeachAdd_us']-len(rs)*h)/w['window_us']*100) for h in [0,2,5,10,20]]))
  profiles.append(dict(ep=src['ep'],window_us=w['window_us'],groups=groups,deliveries=records))
 summaries=[]
 for rows in [24,20]:
  gs=[next(g for g in p['groups'] if g['global_rows']==rows) for p in profiles]
  summaries.append(dict(global_rows=rows,representative_ep=[p['ep'] for p in profiles],mean_window_pct={k:mean(g['window_pct'][k] for g in gs) for k in gs[0]['window_pct']},mean_foreach_removal_scenarios=[dict(added_nonoverlap_us_each=h,net_us_mean=mean(g['conditional_foreach_removal_keep_HCPost'][j]['net_us'] for g in gs),net_window_pct_mean=mean(g['conditional_foreach_removal_keep_HCPost'][j]['net_window_pct'] for g in gs)) for j,h in enumerate([0,2,5,10,20])]))
 out=dict(status='OLD_TWO_RANK_COMPLETED_MOE_HC_OUTPUT_AND_ACTUAL_SHARED_SEDP_SEQUENCE',profiles=profiles,summaries=summaries,
   corrected_previous_scope='Prior5.746362% was norm-ready to HC Post START minus DFC. New remaining_containing_span_keep_DFC_us includes HC Post duration. Preserve old artifact; do not treat its incomplete endpoint as full output cap.',
   qualification='Only old balanced mirrored C8 EP0/EP8 localM3 windows. Typed main-stream sequences support replicated shared weights/no shared TP collectives there; not loaded config/current binary, all16 ranks, whole request or sixordinary. Spans include necessary math/waits; whole-allfree is not removable cost. Foreach-only scenarios retain original HCPost cost and charge incremental fused overhead; not measured speedup. Outside-stream allgather is not assigned to shared experts.',request_credit=0,
   inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in paths],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
 # Independently price the original C8 first-token cohort. Do not transfer the
 # mirrored-window percentage, or incorrectly use current-layer EP arrivals
 # for arithmetic that executes AFTER that layer's EP/DFC join.
 base=Path('/Users/wio/work/branch-fusion-audit-20261104');cp=base/'.cache/first-token-v168-protocol-bodies.json.zlib';ap=base/'full-review/PREFILL-OA-JOINT-ARRIVAL.json'
 compute=json.loads(zlib.decompress(cp.read_bytes()));arrival=json.loads(ap.read_text());native=[]
 for p in compute['profiles']:
  for body in p['bodies']:
   by={r[0]:r for r in body['rows']};stream,={by[i][3] for i in body['HcPre_TASK_ROWS']};rows=sorted((r for r in body['rows'] if r[3]==stream),key=lambda r:r[1]);posts=[r for r in rows if p['regimes'][r[-1]][1]=='HcPost'];assert len(posts)==80
   for layer in range(40):
    post=posts[2*layer+1];i=rows.index(post);add=rows[i-1];ga,gp=[p['regimes'][r[-1]] for r in [add,post]]
    assert ga[1]=='ForeachAddList' and gp[1]=='HcPost' and ga[3].startswith('DT_BF16') and gp[3].startswith('DT_BF16')
    native.append(dict(dp=p['DP'],tp=p['TP'],carrier=body['F21_carrier_rows'],layer=layer,add_task=add[0],post_task=post[0],add_us=(add[2]-add[1])/1000,post_us=(post[2]-post[1])/1000,add_input_shapes=ga[2],post_input_shapes=gp[2]))
 assert len(native)==2560 and len(compute['profiles'])==16
 carriers=[s['M'] for s in arrival['sections']];scenarios=[]
 for scope,h in [('add_only',0),('add_only',5),('add_and_Post_allfree',0)]:
  budget={m:0. for m in carriers}
  for m in carriers:
   for layer in range(40):
    rs=[r for r in native if (r['carrier'],r['layer'])==(m,layer)];assert len(rs)==16
    # Source-selected SEDP has no inter-rank dependency between same-layer
    # adds. This deliberately grants the largest rank credit to all peers,
    # then serializes every layer/carrier to relax every exposure wait.
    values=[r['add_us']+(r['post_us'] if scope=='add_and_Post_allfree' else 0)-h for r in rs]
    budget[m]+=max(values)/1000
  req=[]
  for r in arrival['requests']:
   credit=sum(budget[m] for m in carriers[:carriers.index(r['first_eager_M'])+1]);req.append(dict(slot=r['slot'],DP=r['DP'],TTFT_ms=r['TTFT_ms'],conditional_relaxed_ms=credit,conditional_relaxed_pct=credit/r['TTFT_ms']*100))
  scenarios.append(dict(scope=scope,incremental_fused_overhead_us_each=h,carrier_ms=budget,requests=req,maximum_request_percent=max(r['conditional_relaxed_pct'] for r in req)))
 out['original_C8_first_token']=dict(typed_deliveries=2560,ranks=16,carriers=carriers,scenarios=scenarios,qualification='Separate originalC8 first-token typed tasks. A deliberately relaxed all-peer maximum serialized by source layer/carrier with original request order, no changed EP/schedule model. Grants full exposure and no extra host/copy/capture cost except stated h. Not the next-EP-entry arrival model or a measured/request prediction, not full decode/makespan or sixordinary. Whole add+Post allfree includes mandatory HC math and output writes; cannot be called achievable saving. Negative per-rank cases are not silently clipped; largest-peer relaxation can hide regressions, so native cases must still pass independently.')
 out['inputs'] += [dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [cp,ap]]
 portable=json.dumps(dict(tasks=native,source_pins=out['inputs'][-2:]),ensure_ascii=False,separators=(',',':')).encode()
 (R/'moe-output-original-C8-tasks.json').write_bytes(portable+b'\n')
 (R/'moe-output-original-C8-tasks.json.zlib').write_bytes(zlib.compress(portable,9))
 (R/'moe-output-price.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summaries,ensure_ascii=False))
 print(json.dumps(dict(original_C8_first_token=[dict(scope=s['scope'],h=s['incremental_fused_overhead_us_each'],maximum_request_percent=s['maximum_request_percent']) for s in scenarios])))
if __name__=='__main__':main()
