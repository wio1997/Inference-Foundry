"""MC2 return V1/V2 selection,tile/rank writes and explicit mask-gated work;CPU only."""
from pathlib import Path
from collections import Counter
from itertools import product
from fractions import Fraction
from math import ceil
import hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write

K=N/'MC2-operator-source-v56/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel'
F=W/'frozen-source-snapshot/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel'

def pin(p):
    b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def part(p,a,b):return dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b])
def microtiles(rows):
    result=[]
    for base in range(0,rows,128):
        m=min(128,rows-base);chunks=ceil(m/16);first=chunks//2
        for sub,count in [(0,first),(1,chunks-first)]:
            start=base+(first*16 if sub else 0)
            for j in range(count):
                length=m-first*16-j*16 if sub==1 and j==count-1 else 16
                result.append(dict(subcore=sub,start=start+j*16,length=length))
    return result
def segments(counts,start,length):
    result=[];prefix=0;tile_offset=0
    for peer,count in enumerate(counts):
        lo=prefix;hi=prefix+count;prefix=hi
        if lo>=start+length:break
        if hi<=start:continue
        a=max(lo,start);b=min(hi,start+length)
        if b<=a:continue
        result.append(dict(peer=peer,destination_row_offset=max(0,start-lo),source_tile_row_offset=tile_offset,rows=b-a))
        tile_offset+=b-a
    return result

def main():
    assert not(O/'combine-inputs-v116.json').exists()
    indexes=[read(O/'combine-sdk-source-index-v116.json'),read(O/'combine-sdk-forward-source-index-v116.json')]
    sdk=[r for idx in indexes for r in idx['files']if r['status']=='READ']
    assert len(sdk)==10
    vv=Path(next(r['local_path']for r in sdk if r['path'].endswith('dav_c220/kernel_operator_vec_vconv_impl.h')))
    fw=Path(next(r['local_path']for r in sdk if r['path'].endswith('basic_api/kernel_operator_vec_vconv_intf_impl.h')))
    parts=[part(K/'dispatch_ffn_combine_kernel.hpp',213,257),part(K/'dispatch_ffn_combine_kernel.hpp',590,637),
        part(K/'dispatch_ffn_combine_kernel.hpp',793,879),part(K/'dispatch_ffn_combine_kernel.hpp',895,1083),
        part(K/'dispatch_ffn_combine.h',199,274),part(F/'utils/block_epilogue_pertoken_v2.hpp',65,247),
        part(vv,132,146),part(vv,555,565),part(vv,993,1061),part(fw,31,42),part(fw,107,142)]
    paths=[O/'REVIEW-CHECKPOINT-v115.json',O/'coverage-current-obligations-v115.json',O/'MC2-operator-profile-v56.json',
        O/'MC2-kernel-boundary-and-tiling-contract-v56.json',O/'receiver-budget-operator-source-contract-v97.json',
        O/'MC2-attribute-to-ceiling-contract-v98.json',O/'typed-capacity-primary-contract-v89.json',
        O/'combine-sdk-source-index-v116.json',O/'combine-sdk-forward-source-index-v116.json',O/'combine-mask-primary-receipt-v116.json',
        N/'combine-sdk-inventory166-v116.json',N/'combine-sdk-source166-v116.json',N/'combine-sdk-forward-source166-v116.json']
    parents=[F/'dispatch_ffn_combine_kernel.hpp',F/'dispatch_ffn_combine.h']
    assert all(pin(p)['sha256']==pin(K/p.name)['sha256']for p in parents)
    paths+=list(dict.fromkeys(Path(p['path'])for p in parts))+[Path(r['local_path'])for r in sdk]+parents
    paths+=[Path(read(O/'combine-mask-primary-receipt-v116.json')['path'])]
    write(O/'combine-inputs-v116.json',dict(inputs=[pin(p)for p in dict.fromkeys(paths)],
        profile_question='Which fixed native input geometries select each return branch,thenwhich source tile/byte/mask conditions affectfurtherfusion/workbounds?',
        new_device_compute=False,new_model_or_capture=False,overall_complete=False))
    shapes=[];counts=Counter()
    for r in read(O/'MC2-operator-profile-v56.json')['shape_timings']:
        dims=[list(map(int,x.split(',')))for x in r['input_shapes'].strip('"').split(';')]
        m,h=dims[0];k=dims[3][1];assert dims[3][0]==m and h==5120
        branch='V1'if m*k>4096 else'V2';counts[branch]+=r['executions']
        shapes.append(dict(M=m,H=h,topK=k,M_times_topK=m*k,executions=r['executions'],source_predicted_branch=branch,
            full_kernel_native_duration_ns=r['duration_ns'],input_shapes=r['input_shapes']))
    assert dict(counts)==dict(V2=3512,V1=960)
    write(O/'combine-profile-branch-map-v116.json',dict(cohorts=shapes,source_predicted_counts=dict(counts),existing_executions=4472,
        guard_uses_allocated_local_M_not_active_rows_or_received_rows=True,
        source_compatible_loaded_object_only=True,internal_branch_observed_directly=False,
        actual_received_expert_rows_or_masks_unrecorded=True,new_regime_or_execution_promotions=0,overall_complete=False))
    write(O/'combine-source-contract-v116.json',dict(parts=parts,source_tree_parent_controls=[pin(p)for p in parents],source_tree_parent_hashes_match_v56=True,
        facts=['initBuffer setsisCombineV1 true,thenfalsewhenlocalallocatedM*topK<=4096. V1andV2 dispatch afterSwiGLU aredistinctreturnpaths.',
            'V2reusesGMM2expert/blockgeometry with128x256tiles,then divides eachMtile into16rowmicrotiles overpairedAIVsubcores. Thelastpartialmicrotileisassignedtosubcore1.',
            'For1..16receiverrowsinoneMtile,subcore0 hasnochunk andsubcore1 hasone;thisisnot anallpossiblefusionparallelismcap. LocalM3 doesnotmean3receiverrows.',
            'V2epilogue loadsactualm*actualn FP16values withUBrowstride256;scalesonlyactualrows,andwritesactualrow/rankintersectionsasBF16,keepingFP16→FLOAT→scale→BF16rounding.',
            'ItsUBallocationmax_len=8192,repeat=128 forbothCastcalls;Mulsusesfalsemask too. These literalsdonotproveactiveelements becauseisSetMaskfalseuses externalmask/mode.',
            'InstalledSDK2201selector forwardsCastbool toCastImpl;false bypassesSetMask andcallsunchangedf162f32/f322bf16r intrinsics. Level2Cast(count) setsCounter thenrestoresNormal/fullmask;thehigh-dimensionfalseoverloaddoesnotdo so.',
            'Ranksegments areintersections ofexpert-relativemicrotilewithprefixsumofrawper-sourceexpertcounts;capacityclippinglimitsreceivertiles,not peerprefixidentity.',
            'ExistingC/C2andA/activationaliases,expert/ranklogicalclippingorder,crosscoreflags/scaleepochs/returnpeerbanks andfinalweightedunpermute remain constraints.'],
        SDK_source_selection_not_historical_loaded_binary_or_mask_state=True,new_framework_audit=False,overall_complete=False))
    tilecases=[]
    for m in range(0,514):
        ts=microtiles(m);covered=[x for t in ts for x in range(t['start'],t['start']+t['length'])]
        assert sorted(covered)==list(range(m))and len(covered)==len(set(covered))
        assert all(1<=t['length']<=16 for t in ts)and len(ts)==ceil(m/16)
        tilecases.append(dict(rows=m,chunks=len(ts),subcore_chunk_counts=[sum(t['subcore']==s for t in ts)for s in [0,1]]))
    segment_cases=0
    for cc in product(range(4),repeat=4):
        for capacity in range(sum(cc)+1):
            clipped=min(sum(cc),capacity)
            for t in microtiles(clipped):
                seg=segments(cc,t['start'],t['length']);seen=[];pref=[]
                for rank,c in enumerate(cc):pref.extend((rank,j)for j in range(c))
                for s in seg:
                    seen.extend((s['peer'],s['destination_row_offset']+j)for j in range(s['rows']))
                assert seen==pref[t['start']:t['start']+t['length']]
                assert sum(s['rows']for s in seg)==t['length'];segment_cases+=1
    for cc in [(3,0,17,5),(0,0,48,0),(1,129,0,3),(128,1,17,257)]:
        pref=[(i,j)for i,c in enumerate(cc)for j in range(c)]
        for cap in range(sum(cc)+1):
            for t in microtiles(min(sum(cc),cap)):
                seg=segments(cc,t['start'],t['length']);seen=[(s['peer'],s['destination_row_offset']+j)for s in seg for j in range(s['rows'])]
                assert seen==pref[t['start']:t['start']+t['length']];segment_cases+=1
    write(O/'combine-row-and-peer-interval-proof-v116.json',dict(tile_cases=tilecases,row_partition_cases=len(tilecases),peer_segment_reference_cases=segment_cases,
        formula='EachsourceMtilem<=128 hasq=ceil(m/16)chunks;subcore0gets floor(q/2),subcore1getsceil(q/2). Theirrelativeintervals partition[0,m)once.',
        peer_rule='Fornonnegativeconsistentpeercountswithprefixp_s,sendtile[a,b)intersection[p_s,p_s+c_s) topeer s atpreSumBeforeRank[s,e]+max(a-p_s,0);UBoffsetsumsprecedingintersectionlengths.',
        limitations='Exactsourcecoordinate/semanticstore-spanreference,notcompiledDMAwriteextent,rawactualcounts,loadedABIornewcache/lifetimeproof.',overall_complete=False))
    examples=[]
    for rows in [[1],[3],[16],[17],[48],[2]*24,[25]+[1]*23,[128],[129]]:
        r=sum(rows);s=sum(ceil(x/16)for x in rows);active=5120*r;full=20*8192*s
        examples.append(dict(receiver_expert_rows=rows,R=r,S=s,source_microtile_calls=20*s,
            conditional_full_Normal_cast_elements_per_direction=full,
            active_materialized_value_conversions_per_direction=active,
            conditional_padding_work_ratio=str(Fraction(full,active)),
            old_repeat_per_call=128,new_full_Normal_repeat_per_call='4*actual_microtile_rows',
            GM_input_and_return_value_bytes_unchanged=True,actual_profile_receiver_counts=False))
    frag=[]
    for z in range(1,7):
        for total in range(z,65):
            low=max(z,ceil(total/16));high=z+(total-z)//16
            assert low<=high;frag.append(dict(R=total,positive_experts=z,S_min=low,S_max=high))
    write(O/'combine-mask-gated-work-models-v116.json',dict(status='CONDITIONAL_CEILING',examples=examples,fragmentation_bounds=frag,
        FULL_NORMAL_case='OnlyifexternalhardwaremodeisNormalandall64lanesactiveatbothfalseCast/Mullexecution,eachCastrepeat128submits8192elements. Actualsavedmaskstateunknown.',
        full_Normal_work='H5120,Ntile256⇒20columnblocks;S=sum_e ceil(r_e/16). Perdirection sourceCastwork=163840*S versusactiveFP16/FLOAT/BF16valueconversion5120*R. R=sumr_e=min(A,C)onlyundersamevalidsource/countdomain.',
        fragment_range='GivenR>0andZpositiveexperts, max(Z,ceil(R/16))<=S<=Z+floor((R-Z)/16),achievablewithoutadditionalexpertcapacitybounds. Therefore2<=32*S/R<=32 inthisfullNormal16rowlowering.',
        candidate='ForH5120full256columnblocks,actualm*256isdivisibleby64. InprovenfullNormaldomainreduceCastrepeat128to4*actualm withsamefalsemask/strides andno maskmutation;keeporiginalrounding/scalemultiply/GMstoreandflags.',
        UB_compaction=dict(current_max_len_elements=8192,active16x256_max_elements=4096,current_two_bank_layout_bytes=2*(8192*8+32*4),
            conditional_active_only_two_bank_layout_bytes=2*(4096*8+16*4),
            dependency='TrimallpaddingCastaccessesbefore reducingmax_len;otherwiseoldrepeat128canexceedcompactbuffers. Keep16row/256colassumption,scaleepochs/events andallSwiGLU/returnUBaliasreaders.',
            benefit_scope='OnlythisprivateUBlayoutreference,notglobalGMworkspace/HBM/VRAMrelease orprovedoccupancygain;resourceallocator/kernelreservation maystillfixed.'),
        Counter_case='isSetMaskfalseinheritsCounterstate;repeatmaynotdeterminetotalcount. A repeatonlyrewrite canfailtoelideanywork orchangebehavior. Active-elementworkisunboundwithoutmaskstate.',
        mask_mutation_counterexample='ChangingCasttoCounter(m*n)butleavingfollowingMuls<false>unchangedcanmaketheper-rowMulsapplym*n insteadofn. Needexplicitoldmask/mode preservationorrestoration before everyaffectedconsumer.',
        capacity_cuts='InthestatedCastinstructionloweringanddocumented64elements/AIVcyclecasestudy,useC>=max(W_F162F32/(48*64),W_F322BF16/(48*64),othernecessarycuts). Do notsumtypedtableswithoutsharedissueproof;F_AIVupper/time/MFUunknown.',
        unavoidable_work_not_allalgorithms='Activeconversioncountsprice thisexplicitstoredFP16→FLOATscale→BF16lowering. Alegalnewfusedinstructioncouldrequireadifferentcapacitymodel;noall-algorithmsworklowerboundclaimed.',
        gain='Element-workratiosarenotwholekernel/sourceTASK/requestspeedups;sourcecastsareUBwork,notadditionalHBMbytevolume.',
        required_observers='PaddedUBlanes musthave no subsequent reader,FPstatus/errorobserver,aliasoraddressreuse beforeoverwrite. PreserveFP16round,FLOATscale andBF16CAST_RINT atactivepositions.',overall_complete=False))
    write(O/'combine-guards-and-invalid-rewrites-v116.json',dict(invalid_rewrites=[
        dict(id='LOCAL_M_NOT_RECEIVER_ROWS',local_M=3,possible_receiver_expert_rows=[0,1,3,17,48,128],actual_receiver_rows_unbound=True),
        dict(id='REPEAT_IS_NOT_ACTIVE_COUNT',isSetMask=False,placeholder=-1,repeat=128,Counter_count_example=768,
            consequence='CounterexampleisSDKmaskmodel,notactualmeasuredmask;8192cannotbeinferredfromliteralalone.'),
        dict(id='MASK_LEAK_INTO_MULS',microtile_rows=3,N=256,Counter_Cast_count=768,subsequent_perrow_Muls_intended=256,
            consequence='FalseMulsinheritsCounter768unlessrestored;blindmaskchangeisnotdrop-in.'),
        dict(id='LOGICAL_CLIPPING_ORDER',expert_demands=[2,2],capacity=3,old_selected=[['e0',0],['e0',1],['e1',0]],
            after_blind_expert_reorder_selected=[['e1',0],['e1',1],['e0',0]],
            consequence='SameR3notidenticalselectedassignments;freezeoldkeeprows/offsets beforephysicalschedulingchanges.'),
        dict(id='GM_BYTES_NOT_UB_CAST_PADDING',gm_active_rows=3,N=256,source_UBcast_capacity_rows=32,
            consequence='ActiveonlycastdoesnotremoveGMread/returnstoresalreadylimitedtoactualtilegeometry.'),
        dict(id='VALUE_OR_SCALE_EPOCH',same_scale_address=True,changed_contents=True,
            consequence='Perbuffer source_scale_offset cachingiswithin invocation;cannotcarryitacrossiterationsbyaddressalone.')],
        refined_existing_variants=['F21-FINE','F21-CUT'],new_main_candidates=0,new_pairs=0,new_groups=0,
        remaining=['actualloadedheader/compiler/operator/maskstate','same-callreceivercounts/clipmask','allGMcopylibraryactualwriteextents','padlane/FPstatus/lifetimeobservers','newprecision/event/protocol/capacityevidence'],
        no_native_bug_or_speedup_claim=True,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',source_predicted_branches=dict(counts),row_partition_cases=len(tilecases),peer_segment_cases=segment_cases,
        planned_new_semantic_rows=2,new_device_work=False,new_native_promotions=0,overall_complete=False)))

if __name__=='__main__':main()
