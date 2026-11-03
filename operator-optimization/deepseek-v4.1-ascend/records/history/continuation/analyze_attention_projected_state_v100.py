"""Existing attention/output frontiers and headwise state/projection legality and cost domains."""
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import hashlib,itertools,json
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_norm_quant_v94 import bf16
from analyze_HC_attention_fork_v93 import stats

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(O/'REVIEW-CHECKPOINT-v100.json').exists()
    inputnames=['attention-indexer-profile-and-edges-v60.json','attention-indexer-native-numerical-contract-v60.json',
      'projection-graph-chains-v61.json','projection-operator-numerical-contract-v61.json','V41-query-operator-path-contract-v75.json',
      'attention-joint-interface-models-v83.json','attention-transfer-credit-ledger-v83.json','KV-publication-shared-reader-topology-v87.json',
      'A3-datapath-primary-evidence-v80.json','whole-scope-requirement-audit-v99.json']
    dsa=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/attention/dsa_v41.py'
    cp=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py'
    old=N/'projection-operator-source-v61/vllm_ascend/attention/context_parallel/dsa_cp.py'
    native=N/'attention-operator-source-v60/vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_mla'
    parts=[]
    for p,segs in [(dsa,[(290,295),(613,686),(704,724)]),(cp,[(209,225)]),(old,[(77,93),(1754,1799)]),
      (native/'sparse_flash_mla_torch_adpt.h',[(123,145)]),(native/'op_kernel/arch22/sparse_flash_mla_csa_block_vector.h',[(843,861)]),
      (native/'op_kernel/arch22/sparse_flash_mla_swa_block_vector.h',[(1030,1046)])]:
        ls=p.read_text().splitlines();parts.append(dict(path=str(p),sha256=sha(p),segments=[dict(first_line=a,last_line=b,source=ls[a-1:b])for a,b in segs]))
    write(O/'attention-projected-state-source-contract-v100.json',dict(parts=parts,facts=[
      'CurrentV41attentionreturnsBF16SMLAoutput;theninverse64tailRoPEandBF16outputrestoreTPheads;thelegacyoutputprojectoronlyisreused. NolegacyqueryheadRMSisintroduced.',
      'return_softmax_lse=FalsegivesanemptyFLOATtensor,currentAPIdoesnotreturnperbranchm/l/numeratorneededbyanewsplitmerge.',
      'CSA/SWAvectorBmm2CastAndCopyOutexplicitlyRINTstoBF16beforewrite;thisintermediateroundmustnotbecancelledsolelybyreal-linearidentity.',
      'restore_tp_headspermuteslocaltokens/allheadsintosendchunksandall-to-allreturnsalltokens/localheads. OAgroupedprojectionthenWOB/TPprojectionremains;earlyplacementchangesownership/weights/ABI.',
      'SWA/selectedcompressedKVvaluesremainreadable,currentandfuturecachepublicationspersist,andTopKpreparationdoesnotprovideattentionheadsoftmaxmasses.'
    ],new_target_source_files=0,new_public_documents=0,new_server_or_device_work=False,no_framework_research=True,overall_complete=False))
    a=read(O/inputnames[0]);proj=read(O/inputnames[2]);pidx={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in proj['graph_projection_chains']}
    qi={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in a['graph_QLI_chains']}
    gs=read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json')['graph_executions'];blocks={(g['database_index'],g['graph_ordinal'],b['block_ordinal']):b for g in gs for b in g['blocks']}
    bydb=defaultdict(list)
    for r in a['graph_SMLA_chains']:bydb[r['database_index']].append(r)
    fronts=[];inputs=[];used=set();types=Counter()
    for db,rs in sorted(bydb.items()):
        path=CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json';raw=read(path);inputs.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),original_database_sha256=raw['database_sha256']))
        tasks={r['task_record_rowid']:r for r in raw['TASK_records']};defs={r['globalTaskId']:r for r in raw['COMPUTE_definitions']}
        for r in rs:
            key=(db,r['graph_ordinal'],r['block_ordinal']);p=pidx[key];assert p['root_bound']==r['root_bound']
            lo=r['send_layout']['endNs'];hi=p['stages']['WO_A']['startNs']
            candidates=[tasks[tid]for tid in blocks[key]['task_record_rowids']if tasks[tid]['globalTaskId']in defs and defs[tasks[tid]['globalTaskId']]['opType_text']=='hcom_alltoall_'and tasks[tid]['startNs']>=lo and tasks[tid]['endNs']<=hi]
            assert len(candidates)==1;t=candidates[0]
            stages={k:r[k]for k in ['SMLA','inverse_RoPE','send_layout']}
            stages['restore_heads_A2A']=dict(task_record_rowid=t['task_record_rowid'],op='hcom_alltoall_',startNs=t['startNs'],endNs=t['endNs'],streamId=t['streamId'],duration_ns=t['endNs']-t['startNs'])
            stages.update({k:p['stages'][k]for k in ['WO_A','WO_B','REDUCE_SCATTER']})
            typed={}
            for k,s in stages.items():
                t=tasks[s['task_record_rowid']];d=defs[t['globalTaskId']];assert(t['startNs'],t['endNs'])==(s['startNs'],s['endNs']);used.add((db,t['task_record_rowid']))
                typed[k]=dict(globalTaskId=t['globalTaskId'],task_record_rowid=t['task_record_rowid'],input_shapes=d['inputShapes_text'],output_shapes=d['outputShapes_text'],input_types=d['inputDataTypes_text'],output_types=d['outputDataTypes_text'])
            assert typed['SMLA']['output_shapes']=='"3,64,512;"'and typed['SMLA']['output_types']=='DT_BF16;FLOAT'
            assert typed['send_layout']['input_shapes']=='"3,8,8,512;4"'and typed['send_layout']['output_shapes']=='"8,3,8,512"'
            assert typed['WO_A']['input_shapes']=='"24,1,4096;1,4096,1024"'and typed['WO_A']['output_shapes']=='"24,1,1024"'
            order=['SMLA','inverse_RoPE','send_layout','restore_heads_A2A','WO_A','WO_B','REDUCE_SCATTER']
            assert all(stages[x]['endNs']<=stages[y]['startNs']for x,y in zip(order,order[1:]))
            qli=None
            if key in qi:
                q=qi[key];assert q['SMLA_task_rowid']==stages['SMLA']['task_record_rowid'];qli={k:q[k]for k in ['mode','QLI','prepare']}
                for k in ['QLI','prepare']:used.add((db,q[k]['task_record_rowid']))
            types[r['family']]+=1
            fronts.append(dict(database_index=db,graph_ordinal=key[1],block_ordinal=key[2],root_bound=r['root_bound'],family=r['family'],stages=stages,typed=typed,index_selection=qli,
              SMLA_to_WOA_end_observed_span_ns=stages['WO_A']['endNs']-stages['SMLA']['startNs'],
              restore_protocol_or_pointer_identity_proved=False,partial_attention_state_in_current_API=False))
    assert len(fronts)==3200 and len(used)==23680
    write(O/'attention-projected-state-inputs-v100.json',dict(inputs=[dict(path=str(O/n),bytes=(O/n).stat().st_size,sha256=sha(O/n))for n in inputnames],native_caches=inputs,
      graph_block_index=dict(path=str(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json'),sha256=sha(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json')),overall_complete=False))
    write(O/'attention-output-owner-frontiers-v100.json',dict(frontiers=fronts,count=3200,root_bound=2560,phase_unbound=640,index_selection_frontiers=640,reused_unique_TASKs=len(used),families=dict(types),
      recorded_local_query_output=dict(rows=3,heads=64,D=512,BF16_value_bytes=3*64*512*2),recorded_projection_input=dict(rows=24,groups=1,flattened_K=4096,heads_per_group=8,R=1024,BF16_value_bytes=24*4096*2),
      work_volume_cardinality_equal_not_row_or_rank_causal_identity=True,
      root_SMLA_to_WOA_end_observed_span_ns=stats([r['SMLA_to_WOA_end_observed_span_ns']for r in fronts if r['root_bound']]),
      native_task_sums_not_critical_path_or_speedup=True,new_regime_or_assignment_promotions=0,overall_complete=False))

    # Samehead-massesandprojectedbranchsums,yetdifferentcorrectheadwiseweightedmerge.
    alpha=[F(1,4),F(3,4)];A1=[F(1,4),F(3,4)];B1=[F(3,4),F(1,4)];A2=B1;B2=A1
    merge=lambda A,B:sum(x*a+(1-x)*b for x,a,b in zip(alpha,A,B))
    witness=dict(heads=2,projector_each_head=[1,1],head_A_mass_weights=list(map(str,alpha)),same_B_weights_complement=True,
      branchA_case1=list(map(str,A1)),branchB_case1=list(map(str,B1)),branchA_case2=list(map(str,A2)),branchB_case2=list(map(str,B2)),
      projected_branch_sums_case1=[str(sum(A1)),str(sum(B1))],projected_branch_sums_case2=[str(sum(A2)),str(sum(B2))],
      final_case1=str(merge(A1,B1)),final_case2=str(merge(A2,B2)),same_perhead_masses_but_collapsed_projection_insufficient=True)
    write(O/'attention-headwise-merge-and-linear-state-v100.json',dict(witness=witness,
      unnormalized_counterexample=dict(shared_reference_m=0,perhead_l_A=[1,1],perhead_l_B=[1,3],perhead_final_denominator=[2,4],
        u_A_case1=['1/4','3/4'],u_B_case1=['3/4','1/4'],u_A_case2=['3/4','1/4'],u_B_case2=['3/4','1/4'],
        same_projected_unnormalized_branch_sums=[1,1],final_case1='3/4',final_case2='7/8',
        interpretation='Evenunnormalizedcollapsedbranchsumswiththefullsameperheadm/lmetadataareinsufficientwhenfinalheadnormalizersdiffer. Theseareexactrealstatecounterexamples,notcurrentcheckpointstatereachabilityproof.'),
      stable_state='Perheadperbranch(m,l,u);emptybranchisidentitywithl=0,u=0andguardedm=-inf. Mergefiniteheadstateswithm=maxm_i,l=sum exp(m_i-m)l_i,u=sum exp(m_i-m)u_i. Sinkzero-valuepseudoentrycontributesonce. NativeBF16exp/outputandmergetreefidelityremainunproved.',
      projected_state='ForWblockedbyhead,W_h:[D,R],unnormalizedheadstateu_hcanlinearlyprojectu_hW_hbeforemergeinexactarithmeticifkepthead-separated. Afterallmassesready,combinehead-specificcoefficientsbeforeheadsum.',
      forbidden_unqualified_reduction='Collapsedz_i=sum_h u_ih W_hlosesheadidentity;head-specificmergeweights cannotgenerallybeappliedtoz_iwithonescalar,evenwhenallperheadLSE/massesareavailable.',
      linear_commutation='Foranyhead-diagonalD_alpha,anoutputspaceB_alphaexistsforallrowstatesiffD_alpha W=W B_alpha (equivalentlyleftnullspaceofWisD_alpha-invariant). W=[1;1],D=diag(1/4,3/4),x=[1,-1]givesxW=0butxD W=-1/2.',
      linear_state_dimension='Overreal-linearencodingsandindependentlyvariableheadstates/coefficients,retainatleastsum_h rank(W_h)DOFtosupportallheadweightings. If8headblockseachrank512,4096DOFcannotfitone1024Dcollapsedstate. Actualcheckpointblockranksandjointheadstatereachability/independenceareunknown;thisisnotanactualfixed-model,bit-levelorallalgorithmstoragebound.',
      special_cases='Equalrequiredrescalingacrossheads,projectionafterallmassesknown,structuredblocksubspaces/lowranksorretainedrecoverableheadstatesareseparatedomains;notruledoutbygenericcounterexample. Equalfinaldenominatorscanhelpunnormalizedstatesbutdoesnotrepairalreadyseparatelynormalizedcollapsedbranches.',
      state_storage_references=[dict(query_rows=M,heads=H,output_projector_groups=G,FP32_head_D_numerator_bytes=4*M*H*512,
        FP32_head_R_projected_numerator_bytes=4*M*H*1024,FP32_perhead_m_l_bytes=8*M*H,
        FP32_collapsed_pergroup_R_bytes=4*M*G*1024,
        collapsed_state_not_generically_sufficient=True,
        dense_materialized_branch_value_reference_not_universal_live_storage_bound=True)for M,H,G in [(3,64,8),(24,8,1)]],
      source_returned_LSE=False,source_exposed_branch_states=False,native_precision_or_actual_rank_proved=False,overall_complete=False))

    values=[1.0,1.0,2.015625];den=4 # threeexp(0)keysandoneexp(0)zero-valuedsink
    raw=sum(values)/den;oldout=bf16(raw);oldproj=bf16(oldout*3)
    pushed_float=bf16(sum(v*3 for v in values)/den);pushed_bf16=bf16(sum(bf16(v*3)for v in values)/den)
    assert(oldout,oldproj,pushed_float,pushed_bf16)==(1.0,3.0,3.015625,3.015625)
    write(O/'attention-projection-rounding-and-position-v100.json',dict(BF16_RNE_model_counterexample=dict(key_values=values,all_key_logits=0,sink_logit=0,zero_sink_value=True,denominator=den,scalar_W=3,
      unrounded_attention=raw,rounded_attention=oldout,original_project_after_round=oldproj,preproject_FP32_values_then_round=pushed_float,preproject_BF16_values_then_round=pushed_bf16,
      embedding='PlacevaluesandWintheunrotated448dimensionblock,otherdims/headscontributezero. WitnessdoesnotdependoninverseRoPE.',native_model_inputs_or_device_test=False),
      inverse_RoPE_identity='Forfixedquerypositionqandinexactarithmetic,Y_q=sum_h(sum_k p_qhk v_k) Rinv(q) W_h=sum_hk p_qhk [v_k Rinv(q) W_h].',
      position_cache_gate='W_eff,h(q)=Rinv(q)W_hdependsonthequeryposition. Anordinarykey-position-onlyvaluecachecannotstoreonefixedvW_effforallfuturequerieswithoutadecomposition/invariance/recomputationcontract. KeypositionRoPEandqueryinverseRoPEarenotthesameposition.',
      simple_position_witness=dict(tail_value=[1,0],tail_weight=[1,0],query_inverse_identity_output=1,query_inverse_quarter_turn_output=0,illustrative_rotation_not_actual_model_positions=True),
      partial_nonrot_variant='448nonrotateddimensionsmayuseposition-independentWpartinrealalgebra;retain64tailattentionanditsqueryinverseRoPE/projection. ItstillcrossesBF16attentionroundingandaddshead-separatedstate/merges.',
      precision_rule='KeeporiginalattentionBF16round,inverseRoPEBF16roundandOAoutputroundinanyoutput-epiloguecandidateunlessactualtolerance/numericalequivalenceisproved. RealdistributionofWthroughPVdoesnotproveit.',overall_complete=False))

    work=[];H=8;D=512;RANK=1024
    for M,Nkeys in itertools.product([3,24,128,800,6400],[1,64,128,512,640,1024,2048]):
        for newtag,Unew in [('already_built_ideal',0),('same_distinct_selected_set',Nkeys),('no_cross_query_build_reuse',M*Nkeys)]:
            oldPV=2*M*H*Nkeys*D;oldOA=2*M*H*D*RANK;QK=2*M*H*Nkeys*D
            prebuild=2*Unew*H*D*RANK;prePV=2*M*H*Nkeys*RANK
            threshold=F(M*D*(RANK-Nkeys),D*RANK) # R=2D => M*(1-N/R)
            assert oldPV+oldOA-(prebuild+prePV)==2*H*(M*D*RANK-Unew*D*RANK-M*Nkeys*(RANK-D))
            work.append(dict(logical_projector_group_M=M,H=H,D=D,R=RANK,uniform_N_reference=Nkeys,U_new_reference=Unew,U_domain=newtag,
              old_PV_OPS=oldPV,old_OA_OPS=oldOA,unchanged_QK_OPS=QK,new_value_build_OPS=prebuild,new_projected_PV_OPS=prePV,
              old_QK_PV_OA_OPS=QK+oldPV+oldOA,new_QK_PV_and_build_OPS=QK+prebuild+prePV,
              preproject_lower_reference_GEMM_work=prebuild+prePV<oldPV+oldOA,
              strict_build_break_even_U='U_new < M*(1-N/1024) forD512,R1024',break_even_U_reference=str(threshold),
              projected_value_accumulation_width_ratio=2,
              stat_first_FP32_score_dense_value_bytes=4*M*H*Nkeys,stat_first_BF16_probability_dense_value_bytes=2*M*H*Nkeys,
              stat_first_recompute_QK_additional_reference_OPS=QK,
              nonrot448_build_OPS=2*Unew*H*448*RANK,nonrot_projected_plus_tail_PV_OPS=2*M*H*Nkeys*(RANK+64),nonrot_tail_postprojection_OPS=2*M*H*64*RANK,
              nonrot_PV_width_ratio='17/8',actual_native_case_or_legal_precision_or_free_placement_proved=False))
    topology=read(O/'KV-publication-shared-reader-topology-v87.json');assert topology['source20_longKV_consumer_layers']==list(range(20,40))
    memory=[]
    for Howner,placement in [(8,'ONE_HEAD_OWNER_GROUP'),(64,'CURRENT_QUERY_OWNER_ALL_HEADS')]:
        for consumers in [1,20]:
            for live,ebytes in itertools.product([128,640,34816],[2,4]):
                base=live*512*2;extra=live*Howner*1024*ebytes*consumers
                memory.append(dict(U_live_reference=live,projected_heads_per_owner=Howner,placement=placement,distinct_consumer_projectors_reference=consumers,
                  projected_V_element_bytes=ebytes,projected_V_dtype_reference='BF16'if ebytes==2 else'FLOAT',
                  original_BF16_K_plane_bytes=base,added_head_projected_V_plane_bytes=extra,total_value_bytes=base+extra,
                  total_to_original_ratio=str(F(base+extra,base)),all_projectors_or_positions_compatible=False,
                  original_K_retained_for_unchanged_QK=True,compression_lowrank_recompute_or_on_demand_retention_not_ruled_out=True,
                  actual_allocated_memory_or_feasibility_known=False))
    write(O/'attention-preprojected-value-work-and-storage-v100.json',dict(work_models=work,work_model_count=len(work),storage_models=memory,storage_model_count=len(memory),
      reference_scope='Exact-reallinearconventionaldenseGEMMrolemodel;oneW_OAgroup8heads×512→1024. M,N,U_new,U_livearedeclaredreferences,notactualsamecallvalidkeycounts. BF16/position/placementgateshavenotpassed.',
      current_vs_hypothetical_placement='ObservedattentionisM3/H64;OAownerisM24/H8. M24/H8attentioninthismodelwouldbeanewownershippartition,notacapturedkernel. EqualMHcardinalitydoesnotgrantquery/weight/readinessavailability.',
      storage_qualification='Projectedheadbanksareonechosenrepresentation,notuniversalminbytes. KeeporiginalsharedKforunchangedQK;U_newbuildworkandU_livelifetimeareseparate. Cachewarmbuildcreditmustdeclareprewindowworkandmemory,notfreefuturebank.',
      source20_consumers=topology['source20_longKV_consumer_layers'],projector_equality_or_block_ranks_or_query_position_invariance_attested=False,
      shared_KV_not_shared_output_weights='20source20consumerlayersmayuse20differentOAprojectors. SourceKVsharingdoesnotproveprojectedVsharing. Model20distinctprojectorsisconditional,notcheckedcheckpointvaluesequality.',
      FLOAT_cache_gate='FLOATprojectedVdoublesaddedbankbytesvsBF16. ExactrealOPSreferenceisunchanged,butmixedBF16/FLOATPVsupport,conversion/accuracyandtypedthroughputareunbound;donotchargeFLOATpathatBF16capacity.',
      stat_first_gate='AnothersplitcomputesallrequiredheadQK/statisticsbeforePV. Storefullscores/expandoriginaltile-statesequenceorpayQKrecomputation;currenttileGMqueuesarenotalreadyfull-Npersistentbanks. Densevaluefiguresarealternativebuffers,notadditiveextraHBMtraffic/universallivebounds. ReplayingoriginalBF16exp/scaling/PVorderorprovennewprecisionisrequired;knownmassesremoveoneinformationgatebutnotattention/OArounding.',
      theoretical_bound='Afteralegalgraph/physicalpartitionandmandatoryalgorithmworkarefixed,T_card>=max(F_BF16_card/P_BF16_upper,bytes_GMread/Bread_upper,bytes_GMwrite/Bwrite_upper,live_state_cut,true_tile_dependency_cut,link_cuts). SharedCubeQK/PV/build/OAworkcountedonce;placementandrate/algorithmgatesremain.',
      actual_numeric_physical_ceiling_or_speedup=False,overall_complete=False))
    write(O/'attention-projected-state-variants-v100.json',dict(variants=[
      dict(id='AP100-PRESERVE-ROUND-OUTPUT-EPILOGUE',state='CONDITIONAL',contract='RetainSMLABF16/inverseRoPEBF16andCPpack/restore/OAdataownership;priceA3Vector→CubeGMinterfaceandcommunication. Sameoutputtransfercreditasv83,notanewadditivebudget.'),
      dict(id='AP100-SPLIT-HEAD-STATE',state='CONDITIONAL_NEW_STATE_ABI',contract='StartSWAattentionwhileQLIpreparescompressedselectiononlywithcorrectSWAreadyinputs. Keepperheadm/l/unroundedstate,sinkonce,currentnativeexp/outputroundingorprovennewprecision. CurrentreturnLSEFalseAPIcannotbedirectlycomposedfromBF16outputs.'),
      dict(id='AP100-STAT-FIRST-PV',state='CONDITIONAL_SPLIT_WITH_RETAIN_OR_RECOMPUTE',contract='FinishheadQK/statisticsbeforePVusingretainedscores/probabilities/originaltilestatesequenceoradditionalQKpass. Do notreplaceonlineBF16exp/scaling/PVsequencewithgloballynormalizedBF16Pwithoutfidelityproof. Knownmassesalone donotremovefinalattentionBF16orOAordergates.'),
      dict(id='AP100-PROJECT-BRANCH-AND-COLLAPSE-HEADS',state='REJECT_UNQUALIFIED_ALGEBRA',contract='Perheadmassesarenotgenerallyonepostprojectionscalar. ActualstructuredW/rank/commutationexceptioncouldbeaseparatecandidate;notprovedforcheckpoint.'),
      dict(id='AP100-PROJECT-VALUES-BEFORE-PV',state='CONDITIONAL_DIFFERENT_ALGORITHM',contract='Exactreallinearityneedsfixedq;BF16roundcounterexample,queryinverseRoPE,headexpandedVcache/buildwork,alllayerprojectorsandnewheadownershipmustbepriced. Noautomaticattentionorcachewriteelision.'),
      dict(id='AP100-NONROT-ONLY',state='CONDITIONAL_DIFFERENT_ALGORITHM',contract='Preprojectonly448nonrotateddimensions,retain64tailqueryrotationandprojection. PVwidth1088vs512,headstate/roundingandsharedKstillrequired;notautomaticallycheaper.')],
      no_native_implementation_or_test=True,overall_complete=False))
    evidence=['attention-projected-state-source-contract-v100.json','attention-output-owner-frontiers-v100.json','attention-headwise-merge-and-linear-state-v100.json',
      'attention-projection-rounding-and-position-v100.json','attention-preprojected-value-work-and-storage-v100.json','attention-projected-state-variants-v100.json']
    claims=[('F08','F14','SWA_TO_PROJECTED_ATTENTION_STATE','SWAproducerreadinesscanfeedapartialattentionstate,notafinalOAcontributionwithoneglobalmass. RetainpersistentSWApublicationandnativeBF16/position/outputownership.'),
      ('F12','F14','SELECTION_READY_AND_HEADWISE_PROJECTED_MERGE','QLI/prepareprovidescompressedkeyIDs,nottheheadsoftmaxmasses. EarlySWAworkneedsanewheadstateABI;collapsedprojectedpartialsarenotgenerallymergeableevenwithperheadLSE.'),
      ('F14','F19','SHARED_KV_VS_LAYER_AND_QUERY_SPECIFIC_OUTPUT_PROJECTOR','C1/C2longKVbanksharingdoesnotmakeOAweights/queryinverseRoPEthesame. Preprojectedvaluesarehead/layer/positionqualified,differentfromthepreRoPEIndexKeyandpublishedKV;keepunchangedQKcache.')]
    write(O/'attention-projected-state-composition-clauses-v100.json',dict(clauses=[dict(id=f'REL100-{i:02d}',a=a,b=b,relation=kind,operator_claim=claim,evidence=evidence,
      applicable_domain='Exactcurrentrow/position/cache/readergenerationwithchosenphysicalpartition anddeclaredprecision/algorithm;notallruntimepaths.',conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None)for i,(a,b,kind,claim)in enumerate(claims,1)],
      extend_group='HG83-ATTENTION-OUTPUT',new_members=['F08','F19'],new_groups=0,new_main_candidates=0,new_native_promotions=0,overall_complete=False))
    print(json.dumps(dict(frontiers=3200,TASKs=len(used),families=dict(types),work_models=len(work),storage_models=len(memory),new_pairs=3,new_groups=0,rounding_witness=[oldproj,pushed_float],overall_complete=False)))

if __name__=='__main__':main()
