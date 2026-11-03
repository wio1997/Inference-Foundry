"""Operator-only head, Markov, sampling and acceptance work/traffic models.

All calculations are conditional analytical decisions, not device experiments.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib, math, struct
from finalize_origin_coverage_v9 import N, O, W, read, write
from model_router_vector_v63 import f32, bf16
from analyze_state_engram_v62 import dimensions
S = W / 'frozen-source-snapshot'
V, H, R, P, D = 129280, 5120, 256, 8, 5

def packed(score, col):
    assert math.isfinite(score) and 0 <= col < V
    score = f32(score)
    if score == 0: score = 0.0  # baseline equality treats -0 and +0 as a tie
    bits = struct.unpack('<I', struct.pack('<f', score))[0]
    ordered = (~bits & 0xffffffff) if bits & 0x80000000 else bits ^ 0x80000000
    key = (ordered << 17) | (V - 1 - col)
    assert 0 <= key < 2**49 < 2**63
    return key

def main():
    assert not (O / 'REVIEW-CHECKPOINT-v65.json').exists()
    old_design=None
    if (O/'heads-sampling-fusion-costs-v65.json').exists():
        assert not (O/'heads-sampling-design-refinement-history-v65.json').exists()
        old_design={name:read(O/name) for name in ['heads-sampling-fusion-costs-v65.json','heads-sampling-operator-contract-v65.json','heads-sampling-fusion-variants-v65.json']}
    for a in read(O / 'checkpoint-manifest-v64.json')['artifacts']:
        q = Path(a['path']); data = q.read_bytes()
        assert len(data) == a['bytes'] and hashlib.sha256(data).hexdigest() == a['sha256'], q
    regs = read(O / 'heads-sampling-regime-models-v65.json')['regimes']
    chains = read(O / 'heads-sampling-native-chains-v65.json')
    assert len(regs) == 166 and sum(x['executions'] for x in regs) == 9792
    pins = []
    h32 = read(O / 'M16-head-partition-postprocess-and-layout-contract-v32.json')['source_files']
    h36 = read(O / 'M20-Markov-confidence-and-verification-contract-v36.json')['source_files']
    h28 = {r['path']:r['sha256'] for r in read(O / 'M17-sampling-source-and-liveness-v28.json')['source_files']}
    refs = [
        ('vllm-ascend/vllm_ascend/models/deepseek_v41/model.py', h32),
        ('vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py', h32),
        ('vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py', h32),
        ('vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py', h36),
        ('vllm-ascend/vllm_ascend/spec_decode/utils.py', h36),
        ('vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py', h32),
        ('vllm/vllm/model_executor/layers/logits_processor.py', h32),
        ('vllm/vllm/model_executor/layers/vocab_parallel_embedding.py', h32),
        ('vllm/vllm/model_executor/models/qwen3_dspark.py', h32),
        ('vllm-ascend/vllm_ascend/sample/rejection_sampler.py', h28),
        ('vllm-ascend/vllm_ascend/ops/triton/reject_sample.py', h28),
    ]
    for rel, old in refs:
        path = S / rel
        if not path.exists():
            prefix = 'contract-source-supplement' if rel.endswith('qwen3_dspark.py') else 'sampler-source-supplement'
            path = W / prefix / 'frozen-source-snapshot' / rel
        expected = old['/vllm-workspace/' + rel]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == expected, path
        pins.append(dict(path=str(path), sha256=sha, original166_hash_binding=True))
    # Pure mathematical witnesses; these are not native-kernel correctness tests.
    scores = [-1e30, -20., -1., -0., 0., 1e-30, 1., 20., 1e30]
    for a in scores:
        for b in scores:
            for i, j in [(0,1), (17,17), (V-1,0)]:
                assert (packed(a,i) > packed(b,j)) == ((f32(a),-i) > (f32(b),-j))
    cap = [f32(math.tanh(f32(x))) for x in [20.,30.]]
    assert cap == [1.,1.]  # rounded monotonic processing can create ties
    old_comp, valid, pos = [10,20], [1,1], [1,0]
    simultaneous = [old_comp[g]+valid[g] for g in pos]
    wrong = old_comp.copy()
    for i,g in enumerate(pos): wrong[i] = wrong[g]+valid[g]
    assert simultaneous == [21,11] and wrong == [21,22]
    tail=[]
    for n in sorted({r['N'] for r in chains['target_tails']}):
        snapshot=16*n*H; post=86*n*H+20*n+2*H; fused=10*n*H+20*n+2*H
        assert post-fused==76*n*H
        tail.append(dict(N=n,HC=4,H=H,raw_HC_snapshot_copy_value_bytes=snapshot,
            source_post_snapshot_value_bytes=post,source_with_snapshot_value_bytes=post+snapshot,
            fused_collapse_round_norm_value_bytes=fused,fused_with_snapshot_value_bytes=fused+snapshot,
            preserved_snapshot_fusion_interface_budget_bytes=post-fused,
            snapshot_dead_total_interface_budget_bytes=post+snapshot-fused,
            collapse_ordinary_ops=7*n*H,RMS_ordinary_ops=4*n*H+n,RMS_rsqrt=n,
            current_raw_HC_AllGather_ring_send_bytes=7*n*H,current_premix_AllGather_ring_send_bytes=14*n,
            new_normalized_BF16_AllGather_ring_send_bytes=1.75*n*H,
            live_snapshot_local_collapse_norm_total_send_bytes=7*n*H+1.75*n*H,
            contract='Nisphysicaltargettailrows,notselectedheadK. PreserveFP32premix/sum4,collapsedBF16round,RMSBF16round,gamma/epsilon/rstd/allnormalizedcontentandrawHCsnapshotreaders. Logicalinterfacesarevalueaccessmodels,notcompulsoryHBM.'))
    assert next(r for r in tail if r['N']==6400)['raw_HC_snapshot_copy_value_bytes']==524288000
    target_mul_keys={(r['database_index'],t['TASK_rowid']) for r in chains['target_tails'] for t in r['stages'] if t['operator']=='Mul'}
    helper_counts=Counter()
    for r in regs:
        if r['module_candidates']==['M15'] and r['operator']=='Mul':
            helper_counts[dimensions(r['signature']['inputShapes_text'])[0][0]]+=sum(tuple(k) not in target_mul_keys for k in r['native_TASK_keys'])
    helper_counts=+helper_counts
    assert sum(helper_counts.values())==104
    helper=[dict(N=n,groups=count,source_collapse_value_bytes=82*n*H+16*n,
        fused_preserving_BF16_collapse_value_bytes=10*n*H+16*n,interface_budget_bytes=72*n*H,
        ordinary_ops=7*n*H,contract='Separate104draft/helper4TASKcollapses,identifiedbysubtractingtargetMulTASKkeys. NoRMSpairedbyshape;allconsumerrows/normalizedvsrawconfidenceandHCstateobserversremainconditions.') for n,count in sorted(helper_counts.items())]
    base_shapes=sorted({r['M'] for r in chains['base_heads']})
    heads=[]
    for kind,k,n,ms in [('TP8_BASE',H,V//P,base_shapes),('REPLICATED_MARKOV',R,V,[2,4])]:
        for m in ms:
            f=2*m*k*n; weight=2*k*n; q=2*(m*k+k*n+m*n)
            heads.append(dict(role=kind,M=m,K=k,N=n,ops_2perMAC=f,weight_bytes=weight,
                one_read_write_cold_value_bytes=q,published_BF16_compute_reference_us=f/376e12*1e6,
                published_1p6TB_cold_value_reference_us=q/1.6e12*1e6,
                conditional_reference_floor_us=max(f/376e12,q/1.6e12)*1e6,
                rank_major_gather_transpose_interface_bytes=4*m*V if kind=='TP8_BASE' else 0,
                contract='Publishedperdie376TF/1.6TBreference,notactualclock/mode/cachetraffic/latencyproof. Weightcoldvaluationappliesonlydeclaredcoldcase;actualHBM/L2lambdaandCubeoccupancymaydominate.'))
    comm=[]
    for m in sorted(set(base_shapes+[2,4])):
        comm.append(dict(M=m,P=P,V=V,full_BF16_vocab_AG_ring_send_bytes=2*m*V*7/8,
            packed_INT64_MAX_AllReduce_ideal_ring_send_bytes=8*m*2*7/8,
            INT64_key_AllGather_then_local_max_send_bytes=(P-1)*8*m,
            FLOAT_score_INT32_column_two_AllReduce_send_bytes=2*4*m*2*7/8,
            FLOAT_score_INT64_column_two_AllReduce_send_bytes=(4+8)*m*2*7/8,
            two_INT32_word_key_AllGather_then_local_lexmax_send_bytes=(P-1)*8*m,
            full_vocab_to_two_word_AG_payload_ratio=(2*m*V*7/8)/((P-1)*8*m),
            contract='Allarelogicalringpayloadmodels;collectives/latencies/GMalignandranktopologycostremain. PackedINT64MAXsupportisNOTinferredfromenum/generalAPI:official9.0A3expansionconstraintsomitINT64reduction;fixed9.2loadedmodeunbound. Smalltwo-INT32-wordAllGatherpluslocalunsignedlexmaxisfallbackwithoutINT64reduction;copy/ABI/captureandNaN/tiesstillgated.'))
    assert comm[0]['full_vocab_to_two_word_AG_payload_ratio']==4040
    steps=[]
    for b in [2,4]:
        source=26*b*V+32*b; ideal=2*b*V+8*b
        assert source-ideal==24*b*V+24*b
        steps.append(dict(B=b,V=V,R=R,steps=D,
            source_score_GEMM_output_3Slices_Add_scorecopy_argmax_value_bytes=26*b*V,
            source_argmax32_cast64_IDcopy_value_bytes=32*b,
            fused_base_read_ID64_store_value_bytes=ideal,private_interface_budget_per_step_bytes=source-ideal,
            selected_embedding_lookup_value_bytes=4*b*R+8*b,
            retained_confidence_embedding_buffer_bytes=2*b*D*R,
            original_W2_weight_bytes_per_step_cold=2*V*R,
            W2_weight_bytes_per_5step_loop_cache_lambda_1=2*V*R,
            W2_weight_bytes_per_5step_loop_cache_lambda_5=5*2*V*R,
            fully_warm_loop_HBM_weight_lambda_may_be_zero=True,
            baseline_5step_ops_per_rank=5*2*b*R*V,sharded_5step_ops_per_rank=5*2*b*R*(V//P),
            replicated_combined_HplusR_vs_existing_HoverP_plusR_ops_ratio=(H+R)/(H/P+R),
            sharded_dual_accumulator_vs_base_plus_replicated_Markov_ops_ratio=(H+R)/(H+P*R),
            combined_full_base_weight_bytes=2*H*V,baseline_sharded_base_weight_bytes=2*H*(V//P),
            contract='OneIDSliceBEFOREGatherplus3scoreSlicesAFTERGather. 520source-ordered10TASKsteps:9M20+1M16,not5completeper-requestepochsattested. B×VscoreViewCopy/BIDcopyactualpayload,notB×5VorBmax×6base. OnlyID-only/privatecorrectedscorebranchpermitsideal;preservebiasBF16,postprocess,baseBF16,AddBF16,tie/NaN/remapandnext-stepIDdependency. Weightlambda1..5ONLYcold-firstloop;warmHBMlambda0possible,L2trafficseparate.'))
    assert steps[0]['private_interface_budget_per_step_bytes']==6205488
    assert steps[0]['replicated_combined_HplusR_vs_existing_HoverP_plusR_ops_ratio']==6
    assert steps[0]['sharded_dual_accumulator_vs_base_plus_replicated_Markov_ops_ratio']==.75
    memo=dict(V=V,R=R,full_BF16_transition_table_bytes=2*V*V,
        TP8_vocab_column_sharded_table_bytes=2*V*(V//P),precompute_dense_ops=2*V*V*R,
        dense_table_compute_reference_us=2*V*V*R/376e12*1e6,
        table_output_only_cold_value_reference_us=2*V*V/1.6e12*1e6,
        cache_row_bytes=2*V,cache1024rows_bytes=1024*2*V,
        full_precompute_ops_to_one_B2_step_ops_ratio=V/2,
        block_size=256,block_count=math.ceil(V/256),FP32_block_norm_bounds_bytes=4*math.ceil(V/256),
        contract='MemoizepostprocessedBF16biasrowonlyforfixedtoken+W1/W2/postprocessor/nativeaccumulation/rowpositionandshape-generation;B1memoizedGEMMneednotmatchbaselineB2/B4. Hit/missuniqueIDs/availablecapacity/buildamortizationunknown. Fulltable33.43GBreplicated/4.178GBshardnotfreefit. Certifiedblockpruneusesoutwardnativeerror-boundandstrictupper<currentbest;baseTopKaloneinexact. FuturetokenIDremainserial. Existingapply_bias_gatheredtruncatesdenseproposal(-infelsewhere),notexactmemoization.')
    assert memo['full_BF16_transition_table_bytes']==33426636800
    sampling=[]
    for b,t in sorted({(r['B'],r['T_capacity']) for r in chains['greedy_rejections']}):
        sampling.append(dict(B=b,T_capacity=t,V=V,Kmax=D,
            Index_Cast_workingclone_argmax_value_bytes=22*t*V,
            pure_fused_BF16_source_argmax_score_read_bytes=2*t*V,
            pure_ID_only_interface_budget_bytes=20*t*V,
            raw_FLOAT_clone_value_bytes=8*t*V,
            full_raw_regeneration_BF16_to_FLOAT_value_bytes=6*t*V,
            net_clone_regeneration_budget_if_full_raw_later_needed_bytes=2*t*V,
            current_INT32_output_minus1_Fill_bytes=4*b*(D+1),
            contract='Existinggreedyrejectionwritesaccepted/first-rejectprefixandbonusonlyifallaccepted. Fillalsoservesunaccepted/non-greedyrows;newfullwritekernelcanfoldFill,existingpartialwritercannot. Temp0/positivegreedykerneldoesnotproveallrawlogprob/entropy/processors/RNGbranchesinactive. Syntheticacceptancepreservessamerandomdraws/rates/negativeIDguard.'))
    lazy=[]
    for selected,active,waves,label in [(12,2,1,'B2D5_first_reject_all'),(12,12,6,'B2D5_all_accept')]:
        weight=2*H*(V//P)
        lazy.append(dict(case=label,current_batched_head_rows=selected,lazy_active_head_rows=active,
            lazy_dependency_waves=waves,baseline_dense_ops=2*selected*H*(V//P),lazy_dense_ops=2*active*H*(V//P),
            baseline_one_cold_weight_bytes=weight,lazy_weight_reload_lambda_min=1,lazy_weight_reload_lambda_max=waves,
            lazy_worst_case_cold_weight_bytes=waves*weight,
            fully_warm_HBM_weight_lambda_may_be_zero=True,
            contract='Illustrativeacceptancepatterns,notobservedrowvalidity. Lazyargmax/recoverydoesnotnecessarilymeanlazyGEMM;currentheadalreadyselectsrowsbeforeGEMM. FirstrejectcanreduceF6xbutretainone165.48MBcoldweight. Allaccept6serialheadwavescanreload6x/losebatchtiling;cachelambdaandRNG/bonus/rawconsumersmandatory. Decoderfuturequeriesnotprovenprivate.'))
    stages={'clamp':16,'three_IndexCheck_index_input_reference':24,'valid64_gather':24,'computed32_gather':16,
        'draft32_gather':16,'position_GE0':9,'draft32_to64':12,'draft_GT0':9,'bool_AND':3,
        'valid64_to32':12,'INT32_add':12,'computed_Select32':13,'oldAccepted32_to64':12,
        'accepted_Select64':25,'accepted64_to32':12}
    assert sum(stages.values())==215
    state=[dict(N=n,stages_value_bytes_per_row=stages,source_conventional_value_bytes=215*n,
        fused_snapshot_reads_bytes=32*n,fused_two_INT32_stores_bytes=8*n,interface_budget_bytes=175*n,
        contract='17TASK/call,72calls. BeforeANYstoreloadALLgatheredoldcomputed/valid/draft,prevpos,CPUcomputed,oldAcceptedforwholeNinonesynchronizedprogram;otherwiseimmutableoldbuffer/doublebufferneeded. INT32add/wrap/64→32acceptedcast,currentprefixwriteandbounds/errors/aliases/asyncgenerationretained. Checkcomparisons/scalar/errorflagsnotcompulsoryGM;215Nisdeclaredvalue-reference,notHBM/time.') for n in sorted({r['N'] for r in chains['state_corrections']})]
    if old_design is not None:
        write(O/'heads-sampling-design-refinement-history-v65.json',dict(previous_uncheckpointed_design=old_design,
            refinement='Addedexplicit104separatedraft/helpercollapsebudget;clarifiedcold-firstlambda1..5vswarmHBMlambda0;removedhypotheticalhiddenlayoutvariantwithoutfixedprofileboundary. PriorPASSwasmathematicalmodelgeneration,notnativevalidation.',overall_complete=False))
    write(O/'heads-sampling-fusion-costs-v65.json',dict(target_tail_models=tail,separate_draft_helper_collapse_models=helper,head_models=heads,greedy_vocab_communication_models=comm,
        Markov_step_models=steps,Markov_memoization_and_certified_pruning=memo,sampling_models=sampling,
        lazy_target_head_models=lazy,state_correction_models=state,
        hardware_reference=dict(BF16_ops_per_s=376e12,HBM_bytes_per_s=1.6e12,scope='Publishedperdie910Creference,actualclock/mode/HBM/L2andVector/INT64/specialratesunbound.',url='https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1'),
        roofline='Withinexplicittypedalgorithm:T>=max(Ftyped/Ptyped,QHBM/BHBM,QL2/BL2,comm/latency,dependency);serialized/sharedresourcesequencesaddappropriately. Valueinterfaces/coldvaluations/nativeTASKtimesnotcausalcriticalpathgain.',overall_complete=False))
    write(O/'heads-sampling-operator-contract-v65.json',dict(source_files=pins,new_remote_source_read=False,
        local_clause_review='Onlyoperatorconsumer/rounding/partition/recurrence/outputwrites/snapshotclauses;reuseold166SHA-boundfiles. Noframework/API/HTTP/SSE/lifecycleaudit.',
        facts=[
            'M15has104target6TASKtailsand104separatedraft/helper4TASKcollapses;208collapsegroupsnotpairedto104normsbyshape. FullnormalizedtargetcontentandrawHCsnapshotreadersremainexplicit.',
            'HeadrowselectionalreadyBEFOREbaseLMGEMM;F26pruningwouldmovebeforeHCcollapse/normwithunionofallactivecontentreaders,promptflagsnotattested. BufferNonebranchalreadycollapseslocallybeforeAllGather;donotcountasnew.',
            '192baseheadGEMMsuseTP8BF16[M,5120]×[16160,5120];520MarkovGEMMsusefullreplicatedBF16[B,256]×[129280,256]. 129280physicalconfigwidthdoesnotattestvalidcolumns/added-vocab/softcap/scale/runtimeflags.',
            'Headpostprocessingorderdividecap→tanh→multiplycap→scaleonroundedGEMMvaluesmustprecedebest-columnselection;monotonicroundingcancreateties. Vocabmappingaftercolumnargmax.',
            '520MarkovstepshaveonepreembeddingIDSliceand3postGEMMscoreSlices;BF16base/bias/AddandnextIDdependencyremain. Native520countsnotper-requestlogicalepochs.',
            'ConfidenceusesrawhiddenplusorderedinputembeddingsincludingseedT[:,:D],notoutputT[:,1:D+1]ornormalizedhidden. GlobaldynamicTopKbudget/prefixconditionsalreadyv36,notuniformper-requestthreshold.',
            'CurrentgreedyTritonpartialwritecontractrequires-1suffix/non-greedyoutputinitialization. Fullwritefusionrequiresnewkernel,notjustFilldeletion. Random/recovery/RNG/state/forcedbonusandrawreadersretained.',
            'M24bothdestinationarraysINT32;validcountsummaryINT64. Alloldgathersprecedewrites. Per-rowinplacegather-mapracesunderbatchpermutation;wholeNpreloadoroldbufferrequired.',
            'Next-IDvalidcountliteralpredicate(x!=-1)&(x<V)includesnegative<-1values;optimizationmustnotreplacewith0<=x<Vwithoutvalid-domainproof.' ],
        witnesses=dict(softcap_rounded_tie=dict(raw_scores=[20.,30.],cap=1.,postprocessed_scores=cap,raw_winner=1,processed_first_column_winner=0),
            snapshot_alias_race=dict(old_computed=old_comp,valid=valid,previous_positions=pos,required=simultaneous,naive_inplace=wrong),
            base_TopK_not_Markov_bound=dict(base=[1,0],bias=[-100,100],base_winner=0,corrected_winner=1),
            packed_finite_FLOAT32_lexicographic=dict(combinations_checked=len(scores)**2*3,key_bits=49,tie='firstvalidcolumn',NaN='notcovered;preservebaselineNaNpolicyorfallback',signed_zero='canonicalizedbeforebitpacking')),
        collective_primary_references=[
            dict(url='https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/hcclug/hcclcpp_07_0021.html',version='CANN9.0',finding='GeneralAPIlistsA3INT64andMAX;doesnotbyitselfbindallmodes.'),
            dict(url='https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/hcclug/hcclenvref_07_0009.html',version='CANN9.0',finding='A3AI_CPU/AIVreductiondtypeconstraintsomitINT64;AIVAllGatheralsodoesnotlistINT64. ThereforepackedINT64MAXisconditionalwithtwoINT32wordAGfallback,notassumedlegalfixed9.2.')],
        current_9p2_loaded_collective_dtype_mode_support_proved=False,physical_numeric_ceiling_proved=False,overall_complete=False))
    vs=[]
    def variant(i,c,choice,effect,gates,decision='CONDITIONAL'):
        vs.append(dict(id=i,candidate=c,choice=choice,work_IO_effect=effect,conditions=gates,decision=decision,
            conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,critical_path_gain_us=None))
    variant('TAIL-COLLAPSE-RMS','F26','FuseFP32weightedHCcollapse→BF16round→RMS→BF16','76NHlogicalinterfacebudgetwithrawsnapshotpreserved','FullHnormwithinresourcebudget,gamma/eps/rstd/nativeprecision,allnormalizedandcollapsedreaders;fusedcompletion/readiness.')
    variant('TAIL-SP-LOCAL-NORM','F26','Localcollapse/normbeforeBF16normalizedAllGather','HCAG7NH+premix14N→1.75NHonlyifrawHCsnapshotdead;BufferNonealreadylocalcollapse','Sameownerrows/inputgeneration/allrawHCobservers,fullnormalizedreaders,commordering/capture;live-snapshotcasekeepHCAGandpriceextraHAG.')
    variant('TAIL-ALL-READER-ROW-UNION','F26','Moveorderedselectionbeforecollapse/norm','N→Uallactivecontentreaderrows,notjustselectedheadK','Iavailableearly/ownerplan/duplicates/bounds,promptandhiddenbankconsumers,SPpaddedgather8max(U_p)notfreeU/N;104draftcollapseconsumerdomainsseparate.')
    variant('TAIL-SNAPSHOT-DIRECT-OUT','F26','HCAllGatherwritespersistentrawsnapshotbankdirectly','16NHread/writecopybudget','NewoutABI/alignment/capacity/captureaddress,allaliases/lastreaders/prefixandunchangedtail;notdeletecurrentlylivecopywithoutproof.')
    variant('TAIL-NORM-HEAD','F26','HeadtileconsumesonchipBF16normalizedvalues','Normoutputinterfaceprivatebudgetonly','Fullrow5120statisticandBF16roundbeforeGEMM;manyVtilesmayrepeatnorm/inputloads,raw/normalizedotherreadersandCube/Vectorresourcecost.')
    variant('HEAD-EPILOGUE-SELECT','F35','Eachvocabshardemitsprocessedroundedbestscore/column','AvoidfullscoreplaneplusfullVAGinID-onlydomain','KeepBF16GEMMround/postprocesscap/scale/validcolumn/NaN/firsttieandglobalremapafterselection;otherprob/rawreadersrequireownprotocol.')
    variant('HEAD-PACKED-MAX','F35','49bitfiniteFP32score/columnkeyglobalMAX','Ideal8MINT64ARsend14Mvs226240MfullVAG','Current9.2loadedmodeINT64MAXnotbound;generalAPIenumnotenough,9.0A3modeconstraintsrestrict;NaN/signedzero/padding/proc-tiesand8Balignment.', 'CONDITIONAL_SUPPORT_UNBOUND')
    variant('HEAD-SMALL-KEY-GATHER','F35','AllGather2INT32wordkeythenlocalunsignedlexmax','56Msendvs226240MfullVAG;4040xpayloadratioonly','INT32copy/ABI/capture/rankmappingandallfinite/tie/NaNconstraints;localmerge/launchlatencyand2wordrepresentationcost,not4040xspeedup.')
    variant('HEAD-TWO-STAGE-BEST','F35','FLOATscoreMAXthenINT32minwinningcolumn','Two4MARsend14M,but2latencies','INT32protocolonlyprivatevalidV129280thenrestorecallerINT64/32ABI;exactroundedscoreequality/NaN/padding/postprocessandorderedcollectives.')
    variant('HEAD-DISTRIBUTED-PROBABILITY','F35','Retainpartitionedscoreswithglobalnormalizers/selectedscores','Globalmax+sumexp+selectedp/logprob,entropyadditionalfullpartitionpass','ExactH=-sum(p*log(p+1e-10))notordinarylogZmomentformula;recoveryfullresidual/RNGQglobalmappingandnativeprecision,notservedbybestIDprotocol.')
    variant('MARKOV-ADD-ARGMAX','F30','FuseW2epilogue→BF16bias→BF16Add→argmax→IDstore','24BV+24Blogicalbytes/stepprivateinterfacebudget','Allcorrected-scoreobserversdead,processor/BF16barriers/tie/NaN/remap/embeddingnextdependency;fullscorefallbackretainscorrectedplane.')
    variant('MARKOV-FULLSCORE-FUSED','F30','EmitcorrectedBF16planeandbestIDtogether','RemoveseparateSlice/Add/ViewCopy/readonce;fullplaneoutputremains','Fullscore/probability/recoveryreaderABI/aliases/lifetimeandBF16bias/Add;cannotclaimID-only26BV→2BVbudget.')
    variant('MARKOV-VOCAB-SHARD','F30','ShardW2matchingbaseVpandglobalbesteverystep','W2compute/weight8xsmallerperrank;new5serialglobalIDprotocols','Sameprocessedcolumnvalues/mapping,eachIDbeforedependentembed;prob/fullscorebranchadditionalnormalizer/gather/RNGcost.')
    variant('MARKOV-OWNER','F30','OwnerretainsfullW2/baseandbroadcastsIDsafterrecurrence','Skip7replicacomputations;gatherbaseonceownervsall-rankAG','Ownercriticalpathstill5GEMMs;all-rankconfidence/budgets/RNG/state/fullscoreobserversandownerbroadcast/scatter/liveness;alternativevocabsharding.')
    variant('MARKOV-COMBINED-REPLICATED','F30','Naivereplicatedfull(H+R)GEMMperstep','6xexistingpercardcombinedopsandbaseW1.324GBvs165.48MB','OneaccumulatorroundoncealsochangesBF16base+bias+Add;repeatedbasetiling/weightreads.', 'EXCLUDED_AS_UNCONDITIONAL_OPTIMIZATION')
    variant('MARKOV-DUAL-ACCUMULATOR','F30','ShardedtwoaccumulatorGEMMretainingseparaterounds','Matrixops.75currentbase+repMarkov,fromMarkovsharding','Preservebase/biasBF16postprocess/Add,prebatchedBDbasevsBserialstepweightreuse/tiling;compareagainstexistingbatchedbase+fusedshardedMarkov,notclaimnovelalgebra.')
    variant('MARKOV-EMBED-REUSE','F30','Retaininputembeddingsforconfidence','2BDRbufferbytes,selectedw1readsavoided','ConfidenceusesT[:,:D]includingseedandrawhidden,notoutputtokens/normhidden;BF16andversion/order/lifetime,actualflagnotattested.')
    variant('MARKOV-VERSIONED-MEMO','F30','CachepostprocessedBF16biasrowbytoken','258560Cbytes;costmissuniqueUvsallB,hitunknown','W1/W2/postprocessor/nativeaccum/shape/rowgenerationkeys,B1vsB2/B4precisionandcapacity/lifetime. 33.43GBwholetransitiontableor4.178GBshardprecomputepriced,notfree.')
    variant('MARKOV-CERTIFIED-BLOCK-PRUNE','F30','Strictnative-error-certifiedbase+biasblockupperbound','505blocksat256;FP32W2normmaxmetadata2020B','Outwarddotaccum/BF16/postprocesserrorbound,finiteinputs/tie-safeupper<best;weight/token-generationandboundevaluationcost,unknownsurvivors. BaseTopKaloneinexact.')
    variant('MARKOV-BASE-TOPK-ONLY','F30','KeepbaseTopKthencomputeMarkovonlythere','Counterexamplebase[1,0]+bias[-100,100]changeswinner','Existingapply_bias_gatheredtruncatedproposalAPIisnotexactfull-vocabsubstitute.', 'EXCLUDED_FROM_EXACT_FUSION')
    variant('SAMPLE-INDEX-CAST-ARGMAX','F35','GatherselectedBF16sourceandargmaxwithoutFLOATplanes','20TVinterfacebudgetonlypureIDbranch','Allprocessors/raw/probability/entropy/logprob/forcedbonusreadersinactiveornewexactprotocol;selection/order/errors/nativeNaN/tieandRNGunchanged.')
    variant('SAMPLE-RAW-REGENERATION','F35','RegeneraterawFLOATonactualdemandinsteadofclone','Fullrawlaterneedednet2TVbudget,not8TV','RetainunalteredBF16source/indexgeneration;selectedlogprob/entropyneedsseparateworkandstream/liveness;rawclonecurrentlypositive72.')
    variant('SAMPLE-LAZY-ARGMAX-HEAD','F35','Firstmismatchstopforargmax/recovery;headlazinessseparatelycosted','B2D5firstrejectF/6butoneweightload;allaccept6wavescanreload6x','Actualprefix/budget/bonus/raw/probability/RNGdraworder;currentheadalreadyselected;decoderqueriesnotdeadbyheadprune.')
    variant('SAMPLE-FULLWRITE-REJECT','F35','NewgreedyrejectionwritesallB×6with-1suffix','Fold24BoutputFillbytesandlaunch','Mixednon-greedyoutputownership/bonus/syntheticguards/zeroDraftmustmatch;currentTritonpartialwriterFillcannotdrop.')
    variant('STATE-ONE-PROGRAM-SNAPSHOT','F31','Fuse17stageswithwholeNoldvaluepreloadbeforestores','215N→40Nlogicalreference,175Nbudget','B2/4singleprogramallsourcepreload/barrier,INT32addwrap/64→32cast,bounds/errors/allaliases/asyncgeneration/writeprefix;noarbitraryper-rowinplacegather-map.')
    variant('STATE-IMMUTABLE-OLD-BUFFER','F31','Useimmutableoldstate/doublebufferforlargerN','Fusedmapneedsadditionalbuffer/publicationcost','Old/newgenerationandallreaders/swapordered;notcombinewithsingleprogramzero-extra-storagebudget.')
    variant('STATE-VALIDCOUNT-EPILOGUE','F31','Fullwriteacceptedoutputandvalidcountsummarytogether','Avoidseparatepredicate/cast/reduceplanes','Exactliteral(x!=-1)&(x<V),countsINT64anddestINT32distinct,outputsuffix/RNG/mixedgreedyownership;consumerreadinessandbatchmapgeneration.')
    variant('MARKOV-DYNAMIC-PREFIX','F30','Exactglobalconfidencebudgetbeforehead/recurrenceprefixprune','Onlyknownselectedprefix;globalTopK/histogramv36contractretained','Budgetperiod/allrankcontrol/orderedTseedembeddingsandrawhidden;noncausaldraftdecoderfuturequeriesremainlive;notperrequestuniformthreshold.')
    write(O/'heads-sampling-fusion-variants-v65.json',dict(variants=vs,
        interaction_rules=['Tail76NHpreservedsnapshotvs92NHdead-snapshotbudgetsnotadditive;localSPnormhaslive-snapshotcommunicationpenalty. ExistingBufferNonecollapsealreadybaseline.',
            'HeadbestIDprotocolrequiresallscoreconsumersclosure;distributedprobability/entropy/recoveryareseparateexactalgorithmswithRNG/roundingcost.',
            'PackedINT64MAXnotassumednativelegal;twoINT32wordAGandtwo-stageFLOATMAX/INT32MINarealternatives,notadditivebenefits.',
            'Markovowner/sharded/replicatedmemochoicesmustincludeeachserialIDreadiness,newcollectives/cachelambdaandallconfidence/fullscoreobservers.',
            'Biascachebuild/capacity/hitmiss/nativegenerationandcertifiedpruneboundscostexplicit;currentselected-vocabapproximationnotexactreuse.',
            'SamplingID-onlyandfull-raw-regenerationbudgetsaredifferentreaderbranches;lazyGEMMcanlosebatchweightreuseevenifFfalls.',
            'StatewholeNpreloadvslargerNimmutableoldbufferarealternatives;17TASKcountor17launchesdoesnotimply17xgain.'],
        variant_count=len(vs),whole_modules_closed=0,overall_complete=False))
    print(dict(regimes=166,executions=9792,source_pins=len(pins),variants=len(vs),tail_shapes=len(tail),head_shapes=len(heads),state_models=len(state),packed_reference_checks=len(scores)**2*3,snapshot_witness=simultaneous,naive_wrong=wrong))

if __name__ == '__main__': main()
