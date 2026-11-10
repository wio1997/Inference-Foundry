#!/usr/bin/env python3
"""Price complete Q/KV input-to-query/cache delivery and paid head repartition cases."""
from bisect import bisect_left,bisect_right
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json,statistics,zlib

ROOT=Path('audit/execution-search')
def start(r):return Decimal(r['data']['Start Time(us)'].strip())
def dur(r):return Decimal(r['data']['Duration(us)'])
def end(r):return start(r)+dur(r)
def clean(s):return s.strip('"')
def union(intervals):
    e=None;out=Decimal(0)
    for a,b in sorted(intervals):
        if e is None or a>=e:out+=b-a;e=b
        elif b>e:out+=b-e;e=b
    return out

def main():
    archive=ROOT/'swa-publication-parent-calls.json.zlib';join=ROOT/'swa-publication-join-price.json'
    raw=json.loads(zlib.decompress(archive.read_bytes()));joined=json.loads(join.read_text());results=[]
    for p in raw:
        old=next(x for x in joined['profiles'] if x['ep']==p['ep']);rows=sorted(p['selected'],key=start);byid={r['csv_row']:r for r in rows};byStream=defaultdict(list)
        for r in rows:byStream[r['data']['Stream ID']].append(r)
        qa_pool=defaultdict(list);kv_pool=defaultdict(list)
        for r in rows:
            if r['data']['Type']!='MatMulV2':continue
            shape=clean(r['data']['Input Shapes'])
            if shape=='3,5120;1280,5120':qa_pool[r['data']['Stream ID']].append(r)
            if shape in ['24,5120;512,5120','20,5120;512,5120']:kv_pool[r['data']['Stream ID']].append(r)
        a2as=[r for r in rows if r['data']['Type']=='hcom_alltoall_' and r['data']['Stream ID']!='N/A'];ats=[start(r) for r in a2as]
        ags=[r for r in rows if r['data']['Type']=='hcom_allGather_' and r['data']['Stream ID']!='N/A'];ags_t=[start(r) for r in ags]
        records=[]
        for b in old['binds']:
            main=byStream[b['main_stream']];qb=byid[b['qb_csv_row']];qrope=byid[b['query_rope_csv_row']];cache=[byid[i] for i in b['cache_csv_rows']];smla=byid[b['smla_csv_row']]
            pool=qa_pool[b['main_stream']];qa=pool[bisect_right([end(r) for r in pool],start(qb))-1]
            between=[r for r in main if start(r)>=end(qa) and end(r)<=start(qb)]
            qnorm,=[r for r in between if r['data']['Type']=='RmsNorm' and clean(r['data']['Input Shapes'])=='3,1280;1280']
            dq,=[r for r in between if r['data']['Type']=='DynamicQuant' and clean(r['data']['Input Shapes'])=='3,1280']
            pool=kv_pool[b['cache_stream']];kv=pool[bisect_right([end(r) for r in pool],start(cache[0]))-1];assert start(kv)>=end(qa) and start(qb)>=end(kv)
            assert start(cache[0])-end(kv)<Decimal(10) and start(qb)-end(dq)<Decimal(100)
            a2a=a2as[bisect_left(ats,end(smla))];ag=ags[bisect_right(ags_t,start(qa))-1]
            assert end(ag)<=start(qa) and start(qa)-end(ag)<Decimal(10)
            assert start(a2a)-end(smla)<Decimal(100)
            tail=[r for r in main if start(r)>=end(smla) and end(r)<=start(a2a)]
            inverse,=[r for r in tail if r['data']['Type']=='InplacePartialRotaryMul']
            transpose,=[r for r in tail if r['data']['Type']=='Transpose']
            oproj=next(r for r in main if start(r)>=end(a2a) and r['data']['Type']=='TransposeBatchMatMul')
            assert start(oproj)-end(a2a)<Decimal(10)
            index_owned=any(r['data']['Type']=='VllmAscendQuantLightningIndexerV2' and start(r)>=end(qrope) and end(r)<=start(smla) for r in main)
            ops=[qa,kv,qnorm,dq,qb,*cache,qrope];span=end(qrope)-start(qa);keep3=dur(qa)+dur(kv)+dur(qb)
            assert span>=keep3
            records.append(dict(rows=b['row_count'],main_stream=b['main_stream'],cache_stream=b['cache_stream'],qa=qa['csv_row'],kv=kv['csv_row'],qnorm=qnorm['csv_row'],dq=dq['csv_row'],qb=qb['csv_row'],query_rope=qrope['csv_row'],cache=b['cache_csv_rows'],smla=smla['csv_row'],matched_attention_a2a=a2a['csv_row'],preceding_input_ag=ag['csv_row'],
                root_start=str(start(qa)),root_end=str(end(qrope)),span_us=float(span),native_union_us=float(union((start(r),end(r)) for r in ops)),
                keep_original_serial_three_gemms_us=float(keep3),free_non_gemm_and_gaps_ceiling_us=float(span-keep3),
                qa_us=float(dur(qa)),kv_us=float(dur(kv)),qb_us=float(dur(qb)),qnorm_us=float(dur(qnorm)),dq_us=float(dur(dq)),qrope_us=float(dur(qrope)),swa_us=b['native_sum_us'],
                qnorm_dq_ready_us=str(end(dq)),kv_ready_us=str(end(kv)),cache_end_us=b['cache_end_us'],input_ag_reference_us=float(dur(ag)),same_payload_attention_a2a_reference_us=float(dur(a2a)),
                smla_us=float(dur(smla)),inverse_rope_us=float(dur(inverse)),transpose_us=float(dur(transpose)),oproj_ready_us=str(start(oproj)),query_rope_to_smla_retained_us=float(start(smla)-end(qrope)),owned_QLI=index_owned))
        assert len(records)==516 and len({r['qa'] for r in records})==516 and len({r['kv'] for r in records})==516 and len({r['matched_attention_a2a'] for r in records})==516
        # Paid scenarios restricted to 480 24-row target-like stream instances.
        eligible=[r for r in records if r['rows']==24];assert len(eligible)==480
        scenarios=[]
        for qb_ratio in [0.125,0.25,0.5]:
            for ag_factor in [0.5,1.0,2.0]:
                for swa_saved_fraction in [0.0,0.5,1.0]:
                    for added in [5,20]:
                        values=[]
                        for r in eligible:
                            qready=Decimal(r['qnorm_dq_ready_us'])+Decimal(str(r['input_ag_reference_us']*ag_factor))
                            qdone=max(qready,Decimal(r['kv_ready_us']))+Decimal(str(r['qb_us']*qb_ratio+r['same_payload_attention_a2a_reference_us']))
                            cache=Decimal(r['cache_end_us'])-Decimal(str(r['swa_us']*swa_saved_fraction))
                            finish=max(cache,qdone)+Decimal(str(r['qrope_us']+added))
                            values.append(float(Decimal(r['root_end'])-finish))
                        net=sum(values);scenarios.append(dict(new_qb_duration_ratio=qb_ratio,new_packed_qr_ag_cost_vs_larger_input_ag=ag_factor,SWA_native_fraction_saved=swa_saved_fraction,added_pack_sync_us_per_delivery=added,
                            net_us=net,window_pct=net*100/old['window_us'],negative_deliveries=sum(v<0 for v in values),positive_deliveries=sum(v>0 for v in values)))
        groups=[]
        for n in [24,20]:
            rr=[r for r in records if r['rows']==n]
            groups.append(dict(rows=n,deliveries=len(rr),full_span_free_us=sum(r['span_us'] for r in rr),full_span_free_window_pct=sum(r['span_us'] for r in rr)*100/old['window_us'],
                retain_original_serial_three_gemms_free_other_ceiling_us=sum(r['free_non_gemm_and_gaps_ceiling_us'] for r in rr),retain_original_serial_three_gemms_free_other_window_pct=sum(r['free_non_gemm_and_gaps_ceiling_us'] for r in rr)*100/old['window_us'],
                qb_median_us=statistics.median(r['qb_us'] for r in rr),attention_a2a_reference_median_us=statistics.median(r['same_payload_attention_a2a_reference_us'] for r in rr),input_ag_reference_median_us=statistics.median(r['input_ag_reference_us'] for r in rr)))
        wide=[]
        for qb_ratio in [0.125,0.25,0.5]:
            for ag_factor in [0.5,1.0,2.0]:
                for swa_saved_fraction in [0.0,0.5]:
                    for smla_ratio in [1.0,1.05,1.1]:
                        for added in [5,20]:
                            values=[]
                            for r in eligible:
                                g=Decimal(str(r['input_ag_reference_us']*ag_factor))
                                qb_new=max(Decimal(r['qnorm_dq_ready_us'])+g,Decimal(r['kv_ready_us']))+Decimal(str(r['qb_us']*qb_ratio))
                                cache=Decimal(r['cache_end_us'])-Decimal(str(r['swa_us']*swa_saved_fraction))
                                query_ready=max(cache,qb_new)+Decimal(str(r['qrope_us']))
                                idx_g=g if r['owned_QLI'] else Decimal(0)
                                finish=query_ready+Decimal(str(r['query_rope_to_smla_retained_us']))+idx_g+Decimal(str(r['smla_us']*smla_ratio+r['inverse_rope_us']+added))
                                values.append(float(Decimal(r['oproj_ready_us'])-finish))
                            net=sum(values);wide.append(dict(new_qb_duration_ratio=qb_ratio,new_qr_and_own_index_ag_cost_vs_larger_input_ag=ag_factor,SWA_native_fraction_saved=swa_saved_fraction,new_SMLA_duration_ratio=smla_ratio,added_pack_metadata_sync_us_per_delivery=added,net_us=net,window_pct=net*100/old['window_us'],negative_deliveries=sum(v<0 for v in values),positive_deliveries=sum(v>0 for v in values)))
        results.append(dict(ep=p['ep'],dp=p['dp'],window_us=old['window_us'],groups=groups,scenarios=scenarios,head_partition_through_attention=dict(eligible_deliveries=480,own_QLI_gathers=sum(r['owned_QLI'] for r in eligible),scenarios=wide),deliveries=records))
    out=dict(status='FULL_QKV_DELIVERY_AND_PAID_HEAD_REPARTITION_SENSITIVITY_NOT_PREDICTION',profiles=results,whole_chain_complete=False,every_replacement_bitwise=True,
        geometry=dict(original_Qb=dict(local_M=3,N=32768,K=1280,weight_bytes=41943040),head_partition_target24=dict(TP=8,logical_M=24,N_per_rank=4096,K=1280,weight_bytes_per_rank=5242880,weight_bytes_difference_per_rank=36700160,packed_quant_query_payload_bytes_per_rank=3852,gathered_payload_bytes_per_rank=30816,A2A_per_rank_BF16_bytes=196608)),
        limits=['Two saved representative ranks and source-consistent native geometry; not all-rank/runtime event or request estimate.',
            'Full span free includes necessary work and gaps. Keep-three-GEMMs ceiling assumes original duration and serial ordering; not arbitrary joint resource bound.',
            'Qb time ratios1/8,1/4,1/2 are explicit sensitivities, not weight-byte performance forecasts. Added AG costs use .5/1/2 times existing larger245760B inputAG as reference, not an upper bound or matched-new32KB timing.',
            'A2A references same existing196608B/rank BF16 attention head restore in same iteration; new reverse permutation/extra phase may have different scheduling/contention. All added pack/sync and negative cases retained.',
            'Only 24-row480 instances eligible in this sensitivity. Draft20/padded-tail/zero-query ranks use original fallback; extra collectives require common batch qualification across every TP participant. Precision and output mapping not admitted.',
            'Native QB INT8 exact integer accumulation and identical epilogue remain unproved; BF16 WKV M-change counterexample does not prove or disprove this INT8 case.',
            'Wide head-partition extends through SMLA/inverseRoPE to original TP O-projection layout, conditionally deletes original transpose+A2A rather than adding query A2A. Keeps original queryRoPE-to-SMLA whole remainder translated sequentially; adds an index gather at each actual QLI producer with persistent global-index reuse for consumers. Native bit/metadata/ownership and contention unproved; SMLA duration1/1.05/1.1 sensitivity, no predicted speedup.'],
        inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [archive,join]],new_NPU_work=0,new_profile=0,new_primitive_admissions=0,analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    p=ROOT/'qkv-joint-delivery-price.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(ep=p['ep'],groups=p['groups'],wide_own_QLI=p['head_partition_through_attention']['own_QLI_gathers'],wide_selected=[s for s in p['head_partition_through_attention']['scenarios'] if s['new_qb_duration_ratio']==.25 and s['new_qr_and_own_index_ag_cost_vs_larger_input_ag']==1 and s['added_pack_metadata_sync_us_per_delivery']==5]) for p in results],indent=2))
if __name__=='__main__':main()
