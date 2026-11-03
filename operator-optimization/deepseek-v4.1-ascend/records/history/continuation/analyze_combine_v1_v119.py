"""V1 peer-owned return work and within-expert row redistribution, no target execution."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import gzip,hashlib,json,math,random,statistics
from finalize_origin_coverage_v9 import N,O,W,read,write

def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def section(p,a,b):return dict(**pin(p),first_line=a,last_line=b,source=Path(p).read_text().splitlines()[a-1:b])

def stat(xs):return dict(n=len(xs),minimum=min(xs),median=statistics.median(xs),maximum=max(xs))if xs else None

def split_peer_rows(counts,capacity,workers):
    """Keep original peer prefix clipping; split clipped row ordinals across workers."""
    clipped=[];remaining=capacity
    for n in counts:
        keep=min(n,max(0,remaining));clipped.append(keep);remaining-=keep
    R=sum(clipped);prefix=[0]
    for n in clipped:prefix.append(prefix[-1]+n)
    segments=[]
    for worker in range(workers):
        lo=R*worker//workers;hi=R*(worker+1)//workers
        for peer,n in enumerate(clipped):
            a=max(lo,prefix[peer]);b=min(hi,prefix[peer+1])
            if a<b:segments.append(dict(worker=worker,peer=peer,source_row=a,peer_row=a-prefix[peer],rows=b-a))
    return clipped,segments

def main():
    assert not(O/'REVIEW-CHECKPOINT-v119.json').exists()
    root=W/'frozen-source-snapshot/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel'
    kernel=root/'dispatch_ffn_combine_kernel.hpp';header=root/'dispatch_ffn_combine.h';row=root/'utils/block_epilogue_pertoken_row.hpp'
    const=root/'utils/const_args.hpp'
    dist=N/'MC2-operator-source-v56/saved-observer/EP-count-distribution.json'
    route=N/'MC2-operator-source-v56/saved-observer/EP-routes.json'
    rawpath=N/'MC2-V1-observer-source-v119/decoded-records.jsonl.gz'
    old=[O/'MC2-kernel-boundary-and-tiling-contract-v56.json',O/'MC2-receiver-work-and-communication-model-v56.json',
        O/'combine-profile-branch-map-v116.json',O/'combine-source-contract-v116.json',O/'combine-mask-gated-work-models-v116.json',
        O/'operator-shape-ceiling-ledger-v55.json']
    write(O/'combine-v1-inputs-v119.json',dict(inputs=[pin(p)for p in [kernel,header,row,const,dist,route,rawpath,
        N/'v1-observer-joint166-v119.json',O/'combine-v1-observer-source-index-v119.json',N/'derive_EP_routes_and_ownership_v17_166.py']+old],
        questions=['How are V1 rows assigned to AIV workers and returned?',
            'Can the saved independent count capture bind peer-owned row work without joining later TASK timings?',
            'Which within-expert redistribution preserves clipped rows,scale/destination identity and completion while changing only scheduling?'],
        prior_answers='v56 already describes V1 caller andgenericFINE/CUT;v116 expands V2. This expands the V1 row epilogue and peer-owner load model,not a new overall fused boundary.',
        remote_scope='Readoneexisting1.69MBcompressedmetadatafilewithoriginalv17SHAcontrol;SSHclosed. No newcontainer/model/device work.',new_device_work=False,overall_complete=False))
    controls=read(old[3])['source_tree_parent_controls']
    for c in controls:assert pin(c['path'])==c
    rt=row.read_text();kt=kernel.read_text()
    assert 'uint32_t tileLoops = blockM;'in rt and 'loopIdx ++'in rt
    assert 'AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, blockN)'in rt
    assert 'AscendC::Cast(ubD, ubCFp32, AscendC::RoundMode::CAST_RINT, blockN)'in rt
    assert 'dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum'in kt
    write(O/'combine-v1-source-contract-v119.json',dict(parts=[section(kernel,191,204),section(kernel,958,1023),
        section(row,41,len(rt.splitlines())),section(header,214,254),section(const,1,len(const.read_text().splitlines()))],
        parent_controls_match_v116=True,
        peer_assignment='OnAIV coreIdx=blockIdx+subblockId*blockNum,coreNum=blockNum*subblockDim. V1assigns peer d to worker d modcoreNum. IfEP<=coreNum,eachpeerhasoneworkerandworkers>=EPdonopeerrowwork.',
        return_rows='For each local expert,waitoldcross-coreflag andSyncAll,then rawexpert/peerprefix srcRowOffset,capacity clip,dataRows,preSumBeforeRank destination. Keeporiginalexpert/peerclippingselection.',
        row_epilogue='Two stages. Every row loads5120FP16 fromC2,widens5120toFP32,loadsoneFP32scale,performs5120Muls,casts5120toBF16RINT,andissuesfullrowpeerstore. Per-stage events andS_V/Vector barriers remain.',
        counts_not_V2='Count-form Cast/Muls use blockN5120;V2repeat128/externalfalsemask/8192 padding model is not transferred toV1.',
        private_UB_layout_bytes=2*5120*(2+2+4),
        completion='Finalize per-stage pendingreads/writes,then allAIVSyncAll,ResetTokenPerExpert,CrossRankSync,weightedunpermute. Keepallworkersparticipating;redistributingrowwriters doesnotdeletethecompletionprotocol.',
        flags='Oldexpertflag/barrierprotocol is preserved opaque. No tile-ready or hardwarememory-fence semantics inferred from source naming;no native synchronization bug asserted.',
        historical_loaded_ABI_or_actual_internal_branch_attested=False,overall_complete=False))

    regimes=[r for r in read(old[-1])['observed_regimes']if r['operator']=='DispatchFFNCombine']
    assert sum(r['executions']for r in regimes)==4472
    assert all((r['signature']['blockNum'],r['signature']['mixBlockNum'])==(24,48)for r in regimes)
    selected=[c for c in read(old[2])['cohorts']if c['source_predicted_branch']=='V1']
    assert sum(c['executions']for c in selected)==960
    write(O/'combine-v1-later-profile-scope-v119.json',dict(cohorts=selected,
        all_MC2_executions_reused=4472,source_predicted_V1_executions=960,
        recorded_launch_shape=dict(blockNum=24,mixBlockNum=48),
        source_compatible_AIV_workers=48,EP_case=16,
        worker_model_assumption='The48-worker model uses24blocks*2AIVsubblocks,consistentwithlatermixBlockNum48. Runtimeget_block_num/subblockmappingandtheearlierobserverlaunchare not independentlyattested;thisisnotaphysicalcapacityproof.',
        structural_limit='Given48AIVworkers andEP16,atmost16workers can own anyV1 peer rows. This is not wholekernel hardware utilization or a measured bottleneck.',
        separate_capture='The following observer counts are fromobserver-20261030T160315Z,not laterhostprof-20261030T212812Z. No crosscapture timestamp/ordinal/TASKlatency join.',
        no_V1_internal_phase_time_or_speedup=True,overall_complete=False))

    d=read(dist);q=read(route)
    rb=rawpath.read_bytes();assert hashlib.sha256(rb).hexdigest()=='d6e787e5bd8498c8e68ccd3f2d1874a8dfd20260071fae7f16ad6876ec3c3029'
    raw=[json.loads(line)for line in gzip.decompress(rb).splitlines()];assert len(raw)==4736
    rawgroups={}
    for r in raw:
        assert r['capacity_max_output_size']==65536 and not r['receiver_count_sum_above_capacity_argument']and r['group_binding_complete']
        rawgroups.setdefault((r['role'],r['layer'],r['ordinal']),[]).append(r)
    assert len(rawgroups)==296 and all(len(rs)==16 and {r['ep_rank']for r in rs}==set(range(16))for rs in rawgroups.values())
    key=lambda c:(c['role'],c['layer'],c['ordinal'],c['phase'])
    dq={key(c):c for c in d['cohorts']};qq={key(c):c for c in q['cohorts']};assert len(dq)==len(qq)==296 and dq.keys()==qq.keys()
    rows=[];cohorts=0;actual_partition_cases=0;actual_segment_total=0;actual_digest=hashlib.sha256()
    for ident,c in dq.items():
        if c['phase']!='formal_C8_measure':continue
        cohorts+=1;qr=qq[ident];assert qr['all_group_binding_complete']and len(qr['group_ranks'])==16
        mat=qr['sender_receiver_assignment_matrix'];E=c['E_local'];assert len(mat)==16 and all(len(x)==16 for x in mat)
        rawrank={r['ep_rank']:r for r in rawgroups[(c['role'],c['layer'],c['ordinal'])]}
        assert sum(map(sum,mat))==c['total_assignments']
        for receiver in range(16):
            peers=[mat[sender][receiver]for sender in range(16)]
            experts=c['receiver_expert_rows_by_logical_id'][receiver*E:(receiver+1)*E]
            assert sum(peers)==sum(experts)==c['receiver_rows_by_rank'][receiver]
            if c['M_by_rank'][receiver]*c['K']<=4096:continue
            R=sum(peers);assert 0<R<65536
            assert rawrank[receiver]['M']==c['M_by_rank'][receiver]and rawrank[receiver]['receiver_expert_counts']==experts
            joint=[[int(rawrank[sender]['sender_expert_histogram'].get(str(receiver*E+e),0))for sender in range(16)]for e in range(E)]
            assert [sum(v)for v in joint]==experts and list(map(sum,zip(*joint)))==peers
            old_lb=max(max(peers),sum((x+15)//16 for x in experts))
            balanced_quanta=sum((x+47)//48 for x in experts)
            old_exact=sum(max(v)for v in joint)
            old_ub=min(R,sum(min(x,max(peers))for x in experts))
            assert old_lb<=old_exact<=old_ub and balanced_quanta>0
            remaining=65536
            for e,peer_counts in enumerate(joint):
                clipped,segments=split_peer_rows(peer_counts,remaining,48);assert clipped==peer_counts
                loads=[0]*48;by_peer=[[]for _ in range(16)]
                for seg in segments:
                    loads[seg['worker']]+=seg['rows'];by_peer[seg['peer']].append((seg['peer_row'],seg['rows']))
                for peer,n in enumerate(peer_counts):
                    cursor=0
                    for start,length in sorted(by_peer[peer]):assert start==cursor;cursor+=length
                    assert cursor==n
                assert max(loads)-min(loads)<=1 and max(loads)==(sum(peer_counts)+47)//48
                assert len(segments)<=max(0,sum(x>0 for x in peer_counts)+sum(x>0 for x in loads)-1)
                actual_partition_cases+=1;actual_segment_total+=len(segments)
                actual_digest.update(repr((ident,receiver,e,peer_counts,segments)).encode());remaining-=sum(peer_counts)
            rows.append(dict(role=c['role'],layer=c['layer'],observer_ordinal=c['ordinal'],receiver_EP_slot=receiver,
                local_M=c['M_by_rank'][receiver],topK=c['K'],source_predicted_branch='V1',
                peer_owned_return_rows=peers,expert_return_rows=experts,R=R,
                positive_peer_owners=sum(x>0 for x in peers),old_max_worker_total_rows=max(peers),
                unit_row_expert_barrier_old_quanta_lower=old_lb,unit_row_expert_barrier_old_quanta_upper=old_ub,
                expert_peer_return_rows=joint,unit_row_expert_barrier_old_exact_quanta=old_exact,
                unit_row_balanced48_perexpert_feasible_quanta=balanced_quanta,
                abstract_lower_ratio=str(Fraction(old_lb,balanced_quanta)),
                abstract_exact_row_quanta_ratio=str(Fraction(old_exact,balanced_quanta)),
                per_peer_BF16_return_payload_bytes=[10240*x for x in peers],
                current_source_C2_read_and_peerwrite_logical_bytes=20480*R,
                current_source_FP32_scale_scalar_refs=R,
                source_F16toF32_elements=5120*R,source_F32toBF16_elements=5120*R,source_Muls_elements=5120*R))
    assert cohorts==87 and len(rows)==640 and all(r['role']=='target'for r in rows)
    summary=dict(source_predicted_V1_receiver_records=len(rows),formal_cohorts_reused=87,
        R=stat([r['R']for r in rows]),positive_peer_owners=stat([r['positive_peer_owners']for r in rows]),
        old_max_worker_total_rows=stat([r['old_max_worker_total_rows']for r in rows]),
        abstract_balanced_perexpert_row_quanta=stat([r['unit_row_balanced48_perexpert_feasible_quanta']for r in rows]),
        abstract_old_lower_over_constructed_row_schedule=stat([float(Fraction(r['abstract_lower_ratio']))for r in rows]),
        abstract_old_exact_over_constructed_row_schedule=stat([float(Fraction(r['abstract_exact_row_quanta_ratio']))for r in rows]),
        abstract_strict_lower_ratio_above1=sum(Fraction(r['abstract_lower_ratio'])>1 for r in rows))
    write(O/'combine-v1-observer-peer-loads-v119.json',dict(summary=summary,rows=rows,
        evidence='MEASURED_SAVED_OBSERVER_COUNT_MARGINS_PLUS_DERIVED_CONDITIONAL_SOURCE_ASSIGNMENT',
        count_matrix_axes='matrix[sender][receiver]. Return atreceiver writes backtosender,sooldworker_d_rows=matrix[d][receiver]whenworkerCount>=16.',
        perexpert_peer_joint_counts_recovered_in_observer=True,raw_saved_metadata_records=4736,
        later_hostprofile_joint_counts_still_unbound=True,
        capacity='AllselectedobserverR<65536;priorv56savedcapacityreceipt65536 andnooverflow remain source/observerdomain. Do not promote capacitiesof1272otherlaterprofilecalls.',
        model='48identicalunit-rowworkers,unlimitedbandwidth/noextraDMAorsegmentcost,expertbarriers,no releases. OldL=sum_e max_d x[e,d]isnowcomputedfromrawhistograms. Marginsalonegive max(max_d S_d,sum_eceil(n_e/16))<=L<=min(R,sum_emin(n_e,max_dS_d)). Balancedwithinexperte givesnewfeasiblequanta=sum_eceil(n_e/48).',
        ratio_qualification='These are abstract rowservice quanta and counterfactual schedule counts,not microseconds,physicalspeedup/upperbound/utilization or actualV1stagebottleneck. Peerlinkbytes/readiness/fragmentoverhead may dominate.',
        saved_ordinal_used_only_within_same_observer_artifact_key=True,no_later_profile_time_or_phase_join=True,
        no_new_native_attribution_or_branch_attestation=True,overall_complete=False))

    # Independent logical source/destination row identity and clipping references.
    rng=random.Random(119);samples=[];cases=0;digest=hashlib.sha256()
    for peers in [1,2,3,8,16]:
        for workers in [1,2,3,7,16,24,48]:
            for trial in range(24):
                counts=[rng.randrange(33)for _ in range(peers)];capacity=rng.randrange(sum(counts)+1)
                clipped,segments=split_peer_rows(counts,capacity,workers)
                expected=[]
                for p,n in enumerate(clipped):expected.extend((p,j)for j in range(n))
                actual=[];loads=[0]*workers
                for s in segments:
                    actual.extend((s['peer'],s['peer_row']+j)for j in range(s['rows']));loads[s['worker']]+=s['rows']
                assert sorted(actual)==sorted(expected)and len(actual)==len(set(actual))
                assert max(loads)-min(loads)<=1
                nonempty=sum(x>0 for x in clipped);active_workers=sum(x>0 for x in loads)
                assert len(segments)<=max(0,nonempty+active_workers-1)
                cases+=1;digest.update(repr((counts,capacity,workers,clipped,segments)).encode())
                if len(samples)<6:samples.append(dict(counts=counts,capacity=capacity,workers=workers,clipped=clipped,segments=segments))
    assert cases==840
    write(O/'combine-v1-row-redistribution-proof-v119.json',dict(cases=cases,digest=digest.hexdigest(),samples=samples,
        actual_observer_expert_partition_cases=actual_partition_cases,actual_logical_segments=actual_segment_total,actual_partition_digest=actual_digest.hexdigest(),
        construction='For each expert afteroriginalreadiness,freezeclippedpeer counts. Splitordinal[0,R)byfloor(wR/W)endpoints;intersecteachworkerintervalwithpeerprefixintervals. MapcutstoexactoldC2row,scaleindex,andpeerrow. No expert/rank selection change.',
        properties=['Exactold keptrow set/source-scale-destination identities,no duplicates/holes.',
            'Withinexpert max/minworker rowcounts differbyatmost1;donotmoveacrossexperts or relaxoldflag/barrier/completion.',
            'TwointervalpartitionshaveatmostpositivePeerCount+activeWorkerCount-1logicalpeersegments. Moreepiloguefunctionsegments/metadataworkarepossible,butretainingtheoldone-rowclasskeepstheRrowcopycountunchanged. BatchingDMAneedsseparatefragment/tailcost.'],
        missing_joint_count_witness=dict(expert_by_peer_A=[[4,0],[0,4]],expert_by_peer_B=[[2,2],[2,2]],
            both_expert_margins=[4,4],both_peer_margins=[4,4],unit_row_old_L_A=8,unit_row_old_L_B=4,
            conclusion='Same expert and peer margins cannot identify exact per-expert maximum sum. This prompted retrievaloforiginalsender_expert_histogram;exactjointisnowrecoveredonlyintheoldobservercapture,notlaterTASKprofile.'),
        logical_not_compiled_extent=True,overall_complete=False))
    write(O/'combine-v1-composition-and-theory-gates-v119.json',dict(
        variants=[dict(id='F21-V1-WITHIN-EXPERT-ROWS',change='Redistribute keptrows across48workerswithinexistingexpertreadiness/completion,usingoriginalrowepilogue.',
            old_work_unchanged='SameFP16C2rows/scales/BF16peerdataandweightedunpermute;sameVectorconversions/Muls andGMlogicalcontent. No new arithmetic savings or return-byte reduction.',
            extra_cost='Existingprefixmetadata can be readondevice,not newhostsync;chargeintervalcuts/lookup/functionentries,pipelineoccupancy,peercontention andintegerdivision. Withb1theoriginalRrowDMAs/per-roweventcountstayunchanged;batchingrequiresseparatefragment/tailcost.'),
            dict(id='F21-V1-ROW-BATCH',change='Separateoption:batchbrowsperprivateUBstage to reducehighlevelCast/copy/event grouping.',
                cost='CurrentprivateUBlayout81920B;keepingallthreebuffersandtwofullstageswoulduse81920*b B. Stillbrow-scales andHRMuls;no b-foldinstructionthroughput orfit/occupancy claim.',
                dependency='Newcopy/broadcast/perrowscale/event/mask/UBalias designandtypedcapacity required. Can retainb1forloadbalance;V2paddingtrimcreditnotapplicable.')],
        symbolic_original_source_bound='ForoldfixedpeerownershipandRNErowpath,per-workernecessarysourceconversions/Muls areH*S_d each. Givenindependentlyvalidper-workertypedcapacityupperboundsU_e,T_old_source>=max_d,e(H*S_d/U_e). This is not a universalboundonreorganizedimplementations.',
        transformed_bound='ReorganizedrowpathstillprocessesH*Rsourceelements/type,unlessseparatelyvalidatedrepresentation/fusionremoveswork. Useglobaltypedresourcecuts plusper-peer2HRwrite/requiredstorage/linkcuts;do notaddoverlappableresourcecuts or treatcounterfactualscheduleasphysicaltime lowerbound.',
        preserved=['FP16GMM2round→FP32scale→BF16RINTreturn→originalweightedunpermuteorder/probmetadata.',
            'Originalexpert/peerclipselection,allsource-scale-peerrowmaps,remotealiases,capacityandoldC2/Aretirement lifetimes.',
            'AllAIVexpertflags/barriers andper-stageFinalize,allrankcompletion beforeunpermute;multiplewritersneedrealcompiledextent/disjointstore/protocolproof.',
            'Peerreturnpayload/content unchanged;onehotpeerlinkinjectioncanremainserialdespitemoreAIVworkers.',
            'F21-FINEonchip handoff/F21-CUTexpertpartition arepriorproposals;withinexpertrowbalancecannotdouble-counttheirinterface/work savings.'],
        unknown=['Actualhistoricalcompiledbranch/coremapping andV1phase times','Perexpert-perpeerjointcountsforlaterhostprof andallsource/destallocatorlifetimes',
            'TypedVector/Cast/Muls/GM-L2-HBM/link/injectioncapacitiesandnewfragmentcost','Nonlinearclipping/fullbankreadersandphysicalstoreextents'],
        no_new_pair_group_or_candidate_root=True,new_native_promotions=0,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',observer_source_predicted_V1_rank_records=640,
        later_profile_source_predicted_V1_executions=960,row_identity_reference_cases=cases,
        observer_peer_owners=summary['positive_peer_owners'],abstract_row_ratio=summary['abstract_old_exact_over_constructed_row_schedule'],
        actual_V1_stage_speedup_established=False,new_device_work=False)))

if __name__=='__main__':main()
