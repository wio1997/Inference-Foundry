#!/usr/bin/env python3
"""Price matched whole DFC parents; never assign whole duration to input quant."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
from statistics import median
import json,zlib,itertools
from analyze_o_projection_parent_price import start,dur,first_shape
ROOT=Path('audit/execution-search')
def main():
    paths=[ROOT/n for n in ['swa-publication-parent-calls.json.zlib','hc-early-delivery-price.json','qkv-joint-delivery-price.json','dfc-gemm-tile-profile.json','dfc-input-host-cpu.json']]
    raw=json.loads(zlib.decompress(paths[0].read_bytes()));bindings=json.loads(paths[1].read_text());windows=json.loads(paths[2].read_text());old=json.loads(paths[3].read_text());cpu=json.loads(paths[4].read_text());profiles=[]
    assert next(r for r in cpu['cases'] if r['M']==3)['quant_compute_calls']==18
    for src in raw:
        p=next(r for r in bindings['profiles'] if r['ep']==src['ep']);w=next(r for r in windows['profiles'] if r['ep']==src['ep']);by={r['csv_row']:r['data'] for r in src['selected']};records=[]
        for r in p['deliveries']:
            d=by[r['DFC']];norm=by[r['RmsNormCast']];post=by[r['FFNHcPost']]
            assert d['Type']=='DispatchFFNCombine' and first_shape(d)=='3,5120' and norm['Type']=='RmsNormCast' and post['Type']=='HcPost'
            assert d['Stream ID']==norm['Stream ID']==post['Stream ID']
            native=float(dur(d));parent=float(start(post)-start(norm)-dur(norm));assert parent>=native
            records.append(dict(global_rows=r['rows'],local_DFC_M=3,DFC_csv_row=r['DFC'],norm=r['RmsNormCast'],next_FFNHcPost=r['FFNHcPost'],DFC_us=native,norm_ready_to_FFNPost_us=parent,other_parent_span_us=parent-native,DFC_input_shape=d['Input Shapes'],input_dtype=d['Input Data Types']))
        assert len(records)==516 and Counter(r['global_rows'] for r in records)=={24:480,20:36}
        groups=[];scenarios=[]
        for m in [24,20]:
            rs=[r for r in records if r['global_rows']==m];s=sum(r['DFC_us'] for r in rs);span=sum(r['norm_ready_to_FFNPost_us'] for r in rs)
            groups.append(dict(global_rows=m,local_M=3,calls=len(rs),DFC_sum_us=s,parent_sum_us=span,DFC_per_call_median_us=median(r['DFC_us'] for r in rs),whole_DFC_allfree_window_pct=s/w['window_us']*100,whole_norm_to_Post_allfree_window_pct=span/w['window_us']*100))
            for f,h in itertools.product([.025,.05,.1,.25,1],[0,5,20]):
                net=f*s-len(rs)*h
                scenarios.append(dict(global_rows=m,fraction_of_entire_DFC_removed=f,added_nonoverlap_us_each=h,conditional_net_us=net,conditional_window_pct=net/w['window_us']*100,scope='Sensitivity to whole DFC share; input quant/transport phase share unmeasured. Negative results retained; no per-layer free fallback inferred.'))
        profiles.append(dict(ep=src['ep'],window_us=w['window_us'],groups=groups,scenarios=scenarios,deliveries=records))
    result=dict(status='MATCHED_WHOLE_DFC_PARENT_AND_INPUT_GEOMETRY_SCREEN',profiles=profiles,original_C8_prefill_common_suffix_reference=dict(source='dfc-gemm-tile-profile.json',clock_endpoint_allfree_P90_percent=[x['P90_percent'] for x in old['conditional_values'] if x['kind']=='common_suffix' and x['common_critical_stage_fraction']==1 and x['removable_or_hidden_fraction']==1 and x['added_us_per_layer']==0],one_percent_reference_phase_thresholds=old['one_percent_stage_thresholds'],qualification='Original 2560 eager DFC tasks/24-expert populations, common suffix under archived inter-rank clock bounds; includes all DFC protocol/compute/return. Fixed old external arrival/order, not input quant phase or measured request speedup. Old1percent is prioritization reference, not user hard acceptance threshold.'),candidate_geometries=dict(local_decode_M=3,K=5120,topK=6,original_quant_compute_calls=18,unique_rows=3,original_expanded_sender_record_bytes=18*(5120+32),unique_sender_records_bytes=3*(5120+32),maximum_sender_write_credit_bytes=15*(5120+32),qualification='Only logical sender record payload. Network does not necessarily fall6x: different TopK destination ranks require separate transport; receiver firstGMM wants contiguous expert-ordered A and needs gather or changed native operand loader. Three active row owners instead of18 also reduce parallelism. Allactive illustration, not actual masks/routing distribution or HBM transactions.'),findings=['Inputquant has real source duplicate arithmetic on localM3, but cannot multiply whole DFC by5/6: original calls run on18cores and phase time is unmeasured.','Retaining expandedrecords eliminates only redundant compute/read; unique sender records remove expansion writes but require stable source-row map, owner lifetime and remote gather. Eliminating receivergmA materialization additionally changes GMM operandcopy and publication/flag protocol.','Native quant uses Div(maxAbs/127), FP32->FP16 CAST_TRUNC then INT8 CAST_ROUND. Shared DynamicQuant is a separate primitive and cannot be assumed bitwise equal; same BF16 input and scale shape are insufficient.','Whole DFC remains materially larger than independent Engram smallchildren. Retain whole quant+route+dispatch+GMM parent pending sourcequalified byte/flag scheduling and paid phase share.'],limits=['Two representative mirrored-load short windows, not whole request or all16 rank averages; target24 and draft20 reported separately.','No native internal timestamp or PMU attribution to quant/dispatch is available here; actual phase share, new overlap, extra indirect address/copy costs remain unknown.','CPU host sourcekey is not loaded devicekey. No bitwise, implementation or end-to-end admission.'],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in paths],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    (ROOT/'dfc-input-delivery-price.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(ep=p['ep'],groups=p['groups'],target10percent_h5=[s for s in p['scenarios'] if s['global_rows']==24 and s['fraction_of_entire_DFC_removed']==.1 and s['added_nonoverlap_us_each']==5]) for p in profiles],ensure_ascii=False))
if __name__=='__main__':main()
