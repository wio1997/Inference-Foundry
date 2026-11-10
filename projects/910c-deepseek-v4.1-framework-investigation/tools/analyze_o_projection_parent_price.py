#!/usr/bin/env python3
"""Bind saved SMLA->OA/OB->SP->HC/FFN deliveries and price complete parents."""
from pathlib import Path
from hashlib import sha256
from decimal import Decimal as D
import json,zlib,itertools,statistics,struct

ROOT=Path('audit/execution-search')
def start(a):return D(a['Start Time(us)'].strip())
def dur(a):return D(a['Duration(us)'])
def union(rows):
    spans=sorted((start(a),start(a)+dur(a)) for a in rows);total=D(0);last=None
    for a,b in spans:
        if last is None:total+=b-a;last=b
        elif b>last:total+=b-max(a,last);last=b
    return total
def first_shape(a):return a['Input Shapes'].strip('"').split(';')[0]
def bf16_round(x):
    u=struct.unpack('<I',struct.pack('<f',x))[0];return (u+0x7fff+((u>>16)&1))>>16
def bf16_float(u):return struct.unpack('<f',struct.pack('<I',u<<16))[0]
def main():
    rawpath=ROOT/'swa-publication-parent-calls.json.zlib';raw=json.loads(zlib.decompress(rawpath.read_bytes()))
    parentpath=ROOT/'qkv-joint-delivery-price.json';parents=json.loads(parentpath.read_text());profiles=[]
    for p in parents['profiles']:
        src=next(r for r in raw if r['ep']==p['ep']);byid={r['csv_row']:r['data'] for r in src['selected']};calls=sorted(byid);records=[];used=set()
        for r in p['deliveries']:
            after=[i for i in calls if r['smla']<i<r['smla']+32]
            main=[i for i in after if byid[i]['Stream ID']==r['main_stream']]
            get=lambda typ,lo:next(i for i in main if i>lo and byid[i]['Type']==typ)
            inv=get('InplacePartialRotaryMul',r['smla']);oa=get('TransposeBatchMatMul',inv);ob=get('MatMulV2',oa);post=get('HcPost',ob);pre=get('HcPre',post);norm=get('RmsNormCast',pre)
            rs,=[i for i in after if ob<i<post and byid[i]['Name']=='AivKernel' and byid[i]['Type']=='hcom_reduceScatter_']
            assert first_shape(byid[oa])==f"{r['rows']},1,4096" and first_shape(byid[ob])==f"{r['rows']},1024"
            assert all('BF16' in byid[i]['Input Data Types'] for i in [oa,ob,post,pre,norm])
            assert first_shape(byid[pre])=='3,4,5120' and first_shape(byid[norm])=='3,5120'
            ids={inv,oa,ob,rs,post,pre,norm};assert not ids&used;used|=ids
            body=[byid[i] for i in [oa,ob,rs,post,pre,norm]];core=[byid[i] for i in [oa,ob,rs]]
            end=start(byid[norm])+dur(byid[norm]);oa_begin=start(byid[oa]);ob_begin=start(byid[ob]);rs_end=start(byid[rs])+dur(byid[rs]);post_begin=start(byid[post])
            assert start(byid[ob])>=start(byid[oa])+dur(byid[oa]) and start(byid[post])>=rs_end
            records.append(dict(rows=r['rows'],main_stream=r['main_stream'],smla=r['smla'],inverse_rope=inv,OA=oa,OB=ob,RS=rs,HcPost=post,HcPre=pre,RmsNormCast=norm,OA_blocknum=byid[oa]['Block Num'],OB_blocknum=byid[ob]['Block Num'],OA_to_FFNnorm_span_us=float(end-oa_begin),OA_OB_span_us=float(start(byid[ob])+dur(byid[ob])-oa_begin),OA_OB_handoff_gap_us=float(start(byid[ob])-oa_begin-dur(byid[oa])),OA_OB_RS_retained_union_us=float(union(core)),OA_to_FFNnorm_native_union_us=float(union(body)),free_others_and_gaps_keep_OA_OB_RS_us=float(end-oa_begin-union(core)),OB_RS_span_us=float(rs_end-ob_begin),Post_Pre_norm_span_us=float(end-post_begin),OA_us=float(dur(byid[oa])),OB_us=float(dur(byid[ob])),RS_us=float(dur(byid[rs])),HcPost_us=float(dur(byid[post])),HcPre_us=float(dur(byid[pre])),Norm_us=float(dur(byid[norm])),draft_output_pad_before_RS_csv_rows=[i for i in main if ob<i<post and byid[i]['Type'] in ['ZerosLike','ConcatD']],start_us=str(oa_begin),end_us=str(end)))
        groups=[];scenarios=[]
        for rows in [24,20]:
            vals=[r for r in records if r['rows']==rows]
            sums={k:sum(r[k] for r in vals) for k in ['OA_to_FFNnorm_span_us','OA_OB_span_us','OA_OB_handoff_gap_us','OA_OB_RS_retained_union_us','free_others_and_gaps_keep_OA_OB_RS_us','OB_RS_span_us','Post_Pre_norm_span_us','OA_us','OB_us','RS_us','HcPost_us','HcPre_us','Norm_us']}
            groups.append(dict(rows=rows,deliveries=len(vals),sum_us=sums,per_delivery_median_us={k:statistics.median(r[k] for r in vals) for k in ['OA_to_FFNnorm_span_us','OB_RS_span_us','Post_Pre_norm_span_us']},free_parent_window_pct=sums['OA_to_FFNnorm_span_us']/p['window_us']*100))
            for scope,f,h in itertools.product(['OA_to_FFNnorm_span_us','free_others_and_gaps_keep_OA_OB_RS_us','OB_RS_span_us','Post_Pre_norm_span_us'],[1,.5,.25],[0,5,20]):
                net=sums[scope]*f-len(vals)*h
                scenarios.append(dict(rows=rows,scope=scope,fraction_of_reference_span_removed=f,added_nonoverlap_us_per_delivery=h,conditional_net_us=net,conditional_window_pct=net/p['window_us']*100))
        profiles.append(dict(ep=p['ep'],window_us=p['window_us'],source_consistent_deliveries=len(records),groups=groups,scenarios=scenarios,deliveries=records))
    # Exact two-term typed formula: preserving OA BF16 changes a precomposed product.
    x=[1.,2**-8];wa=[1.,1.];wb=3.;oa_exact=sum(a*b for a,b in zip(x,wa));oa_bits=bf16_round(oa_exact);original_bits=bf16_round(bf16_float(oa_bits)*wb)
    collapsed_bits=bf16_round(sum(a*(b*wb) for a,b in zip(x,wa)));assert original_bits==0x4040 and collapsed_bits==0x4041
    result=dict(status='BOUND_OA_OB_SP_HC_FFN_PARENT_AND_TYPED_FORMULA_COUNTEREXAMPLE',profiles=profiles,
      algebraic_counterexample=dict(scope='Finite IEEE FP32/BF16 RNE formula, not Ascend/model observed result. K2 example can be padded with zeros; not a nativeMMA proof.',x=x,wa=wa,wb=wb,OA_exact=oa_exact,OA_BF16_hex=hex(oa_bits),keep_OA_rounding_BF16_hex=hex(original_bits),precompose_without_OA_rounding_BF16_hex=hex(collapsed_bits),stops='Universal unconditional OA/OB weight precomposition that erases OA BF16 rounding, not an original-rounding fused kernel.'),
      typed_payload_per_rank=dict(OA_weight_bytes=4096*1024*2,OB_weight_bytes=1024*5120*2,precomposed_weight_bytes=4096*5120*2,original_MAC_per_row=4096*1024+1024*5120,precomposed_MAC_per_row=4096*5120,precomposed_compute_ratio=(4096*5120)/(4096*1024+1024*5120),OA_intermediate_roundtrip_bytes_per_row=1024*2*2,OB_partial_roundtrip_bytes_per_global_row=5120*2*2,HC_retained_residual_write_required=True,qualification='Nominal typed tensor payload/compute only, not HBM transactions/VRAM release. Fullrank native matching and bitwise gates remain.'),
      limits=['Saved balanced mirrored C8 steady2representative ranks only; full request and sixordinary/prefill prices remain separate oldqualified evidence, not extrapolated from this window.','Rows20draft pads four output rows to24 beforeRS; target24 has no such pair. Never reuse target24 rank/row price for draft or zero-query automatically.','Source-consistent native row sequences and types, not runtime pointer/event proof; communication AivKernel is used once, matching HCCL aggregate not charged twice.','Spans include necessary computation/communication and gaps. All-free cases are screening envelopes only; retained-core remainder and fraction/cost cases are conditional, not removable instructions or native/request prediction.','CANN activekey/originalMMA/HC rounding and collective reduction order still unqualified. Finite formula is not nativebit; library fused MM+RS cannot be accepted by dtype alone.'],
      original_large_prefill_reference=['Attention输出到FFN交付判断.md','attention-output-value.json','attention-output-portable.json'],whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [rawpath,parentpath]],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'o-projection-parent-price.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps([dict(ep=p['ep'],groups=p['groups'],target_half_h5=[s for s in p['scenarios'] if s['rows']==24 and s['fraction_of_reference_span_removed']==.5 and s['added_nonoverlap_us_per_delivery']==5]) for p in profiles],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
