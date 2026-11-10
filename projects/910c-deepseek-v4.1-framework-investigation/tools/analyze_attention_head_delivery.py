#!/usr/bin/env python3
"""Derive CP causal/output coordinates, head-schedule traffic and decisive cost margins."""
from pathlib import Path
from hashlib import sha256
import itertools,json,zlib,collections

ROOT=Path('audit/execution-search')
def main():
    # Source CP suffix formula: request_end - clamp(request_end,actual_start,actual_end).
    mask_cases=mask_coordinates=0
    for lengths in [(6,6,6,6),(5,5,5,5),(1,2,0,5),(6,1,4,2),(0,0,0,0)]:
        ends=list(itertools.accumulate(lengths));starts=[0]+ends[:-1];total=sum(lengths);per=(total+7)//8
        for seq in [128,129,32768,33792]:
            seen=[]
            for rank in range(8):
                lo=rank*per;hi=min((rank+1)*per,total)
                lo=min(lo,hi)
                for b,(a,z,q) in enumerate(zip(starts,ends,lengths)):
                    first=max(a,lo);last=min(z,hi);localq=max(last-first,0)
                    suffix=z-min(max(z,lo),hi)
                    local_seq=seq-suffix
                    if localq==0:continue
                    for t in range(localq):
                        global_j=first-a+t
                        right_local=local_seq-localq+t
                        right_global=seq-q+global_j
                        assert right_local==right_global
                        assert max(right_local-127,0)==max(right_global-127,0)
                        seen.append(first+t);mask_coordinates+=1
            assert sorted(seen)==list(range(total));mask_cases+=1
    # Source restore_tp_heads recv concatenates source-rank token slices at destination head rank.
    coordinate_count=0
    for dest in range(8):
        for source in range(8):
            for token in range(3):
                for local_head in range(8):
                    global_q=source*3+token;global_head=dest*8+local_head
                    for d in range(512):
                        old_recv_address=((source*3+token)*8+local_head)*512+d
                        new_tp_address=(global_q*8+local_head)*512+d
                        assert old_recv_address==new_tp_address
                        original_cp_value_coordinate=(source,token,global_head,d)
                        desired_new_value_coordinate=(global_q//3,global_q%3,dest*8+local_head,d)
                        assert original_cp_value_coordinate==desired_new_value_coordinate;coordinate_count+=1
    traffic=[]
    for K in [0,64,128,256,512]:
        # DenseSWA is read directly for QK and PV; compressed KV goes cache->GM merge->QK/PV.
        per_query=(2*128+4*K)*512*2
        old=3*per_query;new=24*per_query
        traffic.append(dict(compressed_valid_rows=K,SWA_rows=128,bytes_per_query=per_query,
            old_uniform_rank_bytes=old,new_uniform_rank_bytes=new,extra_uniform_rank_bytes=new-old,
            ratio=8,weight_read_payload_difference=35*1024*1024,net_extra_vs_nominal_Qb_weight_read_difference=new-old-35*1024*1024))
    packet=ROOT/'attention-head-schedule-cpu-a3/result.json';schedule=json.loads(packet.read_text());assert schedule['status'].startswith('PASS')
    summaries=[]
    for cmp in [0,1]:
        rr=[r for r in schedule['cases'] if r['per_request_q']==6 and r['ratio']==2 and r['compressed']==cmp and not r['batch_consistency']]
        cp=[r for r in rr if r['heads']==64];glob,=[r for r in rr if r['heads']==8]
        summaries.append(dict(compressed=bool(cmp),CP64_FA_cores=[r['fa_cores'] for r in cp],global_head8_FA_cores=glob['fa_cores'],CP64_q_rows=[r['q_rows'] for r in cp],global_q_rows=glob['q_rows'],all_default_noFD=all(r['fd_heads']==0 and not r['S2split'] for r in rr),CP64_mBase=cp[0]['mBase'],head8_mBase=glob['mBase']))
    pricefile=ROOT/'qkv-joint-delivery-price.json';prior=json.loads(pricefile.read_text());margins=[];branches=[]
    rawfile=ROOT/'swa-publication-parent-calls.json.zlib';raw=json.loads(zlib.decompress(rawfile.read_bytes()))
    for p in prior['profiles']:
        native={r['csv_row']:r['data'] for r in next(r for r in raw if r['ep']==p['ep'])['selected']}
        counts=collections.Counter();durations=collections.Counter()
        for r in p['deliveries']:
            if r['rows']!=24:continue
            fields=native[r['smla']]['Input Shapes'].strip('"').split(';')
            kind='compressed_CSA' if fields[2] else 'dense_SWA_only'
            counts[kind]+=1;durations[kind]+=r['smla_us']
        assert counts=={'compressed_CSA':456,'dense_SWA_only':24}
        branches.append(dict(ep=p['ep'],target_delivery_counts=dict(counts),native_SMLA_sum_us=dict(durations),qualification='Actual third-input presence + unchanged Python has_compressed and arch22 template dispatch; not runtime tiling-key identity. All-parent cost margins below include both branches.'))
        T=sum(r['smla_us'] for r in p['deliveries'] if r['rows']==24)
        for swa in [0,0.5]:
            base=next(s for s in p['head_partition_through_attention']['scenarios'] if s['new_qb_duration_ratio']==.25 and s['new_qr_and_own_index_ag_cost_vs_larger_input_ag']==1 and s['SWA_native_fraction_saved']==swa and s['new_SMLA_duration_ratio']==1 and s['added_pack_metadata_sync_us_per_delivery']==5)
            for target_pct in [0,1]:
                required=p['window_us']*target_pct/100
                margin=base['net_us']-required
                margins.append(dict(ep=p['ep'],SWA_native_fraction_saved=swa,window_reference_pct=target_pct,original_target_SMLA_sum_us=T,base_conditional_net_us=base['net_us'],remaining_extra_SMLA_and_metadata_budget_us=margin,
                    extra_budget_per_delivery_us=margin/480,max_new_SMLA_duration_ratio_if_no_other_extra=1+margin/T,
                    max_total_pack_metadata_sync_us_per_delivery_if_SMLA_unchanged=5+margin/480))
    out=dict(status='BOUNDED_HEAD_LAYOUT_METADATA_COORDINATES_AND_RESOURCE_GAP',whole_chain_complete=False,every_replacement_bitwise=True,
        source_coordinate_checks=dict(causal_CP_mask_frames=mask_cases,causal_query_coordinates=mask_coordinates,output_TP_coordinates=coordinate_count,
            scope='Integer source-coordinate identities only; not native MMAD/SoftmaxFlashV2 result/complete CP builder or backend test. Newroute guardtarget24 only; extra tail/empty examples explain fallback.'),
        original_scalar_schedule=summaries,
        native_branch_inventory=branches,
        dense_SWA_geometry=dict(scope='Source mask and scalar schedule geometry only, not complete copy count or native cost. mBase256 permits several queries in one block, unlike CSA.',CP_uniform_local_q=3,global_per_request_q=6,CP_FA_cores=1,global_FA_cores=4,CP_mask_union_rows=130,global_mask_union_rows_per_request=133,CP_active_M=192,global_active_M_per_request=48,global_total_active_M=192,limits='Do not apply CSA8x visits or2x paddedM to these24calls. N-buffer splitting, exact read count, softmax row tiling and native bit/cost still unqualified.'),
        padded_Cube_rows=dict(scope='Compressed CSA path only; dense-SWA branch has mBase256 and requires separate consumer analysis',original_per_uniform_rank=3*64,new_per_uniform_rank=24*16,ratio=2,active_queryhead_rows_same=192,reason='Compressed mBase=G; both QK/PV align actualM to16. Head8 usesM16 for8 active rows; source shape plan only, not measured FLOP utilization.'),
        traffic=traffic,
        traffic_scope='Compressed CSA path: TP aggregate KV-query visits grow8x for same24 queries when each of8headrank visitsallqueries instead ofoneCP owner/query; perrank8x onlyuniformR reference. WithinCSA, SWA part two direct PA reads, compressedpart4 logicalGM handoffs. Firsttwo denseSWA-only layers have distinct mBase256 branch, not covered by thistraffic multiplier. No L2/HBM/padding/miss-rate or physicaltransaction claim; zero/negative sentinel and masks may reduce actual validK.',
        cost_margins=margins,
        value_interpretation=['Previous loop038 source-consistent prices remain immutable conditional sensitivities. SMLA ratio1/1.05/1.1 was not a native prediction; this review exposes 8x logicalKV visits and2x paddedCube M, offset by more activeFA cores inCPUplan.',
            '1%window line is a sensitivity reference only, not original fullrequest1% acceptance gate. Margins include all additional exposed globalmetadata/packing/flags; baseline assumptions Qb-quarter and addedAG references remain unmeasured.',
            'New globalquerymetadata cannot reuse cache-only packet or blindlyreplace CPU seq-lens mirror. Preserve original device-authoritative suffix and perhead sink/selected-ID generation; dynamic row count feeds SoftmaxFlashV2 tiler, so perhead independence doesnot prove nativebit.',
            'Further headlayout development needs globalquery commonplan reuse and measured/bit-qualified SMLA/epilogue. Keep larger original-order KV/L1/cache reuse parent pending, do not spend standalone NPUprobes before these decisive gaps.'],
        next_large_delivery='Existing scoped prepare input metadata delivery: coordination-end tofirst native metadata submit~31ms/243where periteration, Free overlap~27ms. Not exact function or unprofiled/request saving; trace common CP query/suffix/length/mask plan togroup publishers and actual consumers.',
        inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [packet,pricefile,rawfile]],new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],new_primitive_admissions=0,analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    p=ROOT/'attention-head-layout-delivery.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(coordinates=out['source_coordinate_checks'],schedules=summaries,trafficK512=traffic[-1],margins=[r for r in margins if r['window_reference_pct']==1]),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
