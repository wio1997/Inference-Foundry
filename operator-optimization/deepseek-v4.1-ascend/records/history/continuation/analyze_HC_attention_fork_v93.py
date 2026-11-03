"""Bind existing HC/projection records to operator deadlines and shared weight-service cuts."""
from pathlib import Path
from copy import deepcopy
from collections import defaultdict,Counter
from math import ceil,prod
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions


def stats(x):
    x=sorted(x);n=len(x)
    return dict(n=n,minimum=x[0],median=(x[n//2]if n%2 else(x[n//2-1]+x[n//2])/2),maximum=x[-1])if x else None


def main():
    assert not(O/'REVIEW-CHECKPOINT-v93.json').exists()
    p=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py';sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha=='101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91'
    lines=p.read_text().splitlines()
    prior=read(O/'HC-operator-numerical-and-dependency-contract-v59.json');assert prior['model_config']['hc_mult']==4
    write(O/'HC-attention-fork-source-contract-v93.json',dict(path=str(p),sha256=sha,first_line=879,last_line=920,source=lines[878:920],
      prior_contracts=['HC-operator-numerical-and-dependency-contract-v59.json','HC-fusion-split-reorganization-variants-v59.json','projection-operator-numerical-contract-v61.json','typed-capacity-primary-contract-v89.json'],
      facts=[
        'AttentionHcPre receivesoriginalHCresidualXandexternaloldpre_mix. ItsBF16Y feedsinputRMS/attention;currentpost/comb feedsameattentionHcPost;currentpre feedsfollowingFFNHcPre.',
        'FollowingFFNPre receivesupdatedattentionPostresidualandcurrentattn_pre coefficients,thenproducesYffnforRmsNormCast. F04isnotanotherconsumer oforiginalattentionY.',
        'All8capturedPre regimes includeexternalpremix. CurrentY/currentcoeffsplitindependence alreadyv58/v59;ordinarysplitDAGisreusednotanewoptimizationdiscovery.',
        'HcPrecoefficientprojection/normalization usesfull4HoriginalX,FLOAT/HF32weightplane. Q_A/SWAuseBF16normalizedcollapsedY withdifferentlocal/globalrows. Theyarenotonesame-inputwideGEMM.',
        'HcPrecompositeF01 andF02/othercompositions mayreferthesameHCwork. Countthecurrentcoefficient/collapse/norm once;newpairclauses arecoverage reconciliation,notsevenindependentgains.'
      ],
      no_new_target_source_files=True,new_native_or_Torch_execution=False,all_readers_aliases_or_precision_closed=False,overall_complete=False))
    hc=read(O/'HC-operator-profile-and-edges-v58.json')['exact_graph_role_edges'];bykind=defaultdict(dict)
    for e in hc:bykind[e['kind']][(e['database_index'],e['graph_ordinal'],e['block_ordinal'])]=e
    chains=read(O/'projection-graph-chains-v61.json')['graph_projection_chains'];bydb=defaultdict(list)
    for c in chains:bydb[c['database_index']].append(c)
    fronts=[];used=set()
    for db,cs in sorted(bydb.items()):
        d=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');tasks={r['task_record_rowid']:r for r in d['TASK_records']};defs={r['globalTaskId']:r for r in d['COMPUTE_definitions']}
        for c in cs:
            key=(db,c['graph_ordinal'],c['block_ordinal']);a=bykind['ATTENTION_PRE_TO_NORM'][key];b=bykind['INTRALAYER_POST_TO_PRE'][key];f=bykind['FFN_PRE_TO_NORM'][key]
            pre=tasks[a['producer_task_rowid']];norm=tasks[a['consumer_task_rowid']];post=tasks[b['producer_task_rowid']];ffnpre=tasks[b['consumer_task_rowid']];ffnnorm=tasks[f['consumer_task_rowid']]
            assert f['producer_task_rowid']==b['consumer_task_rowid']and c['root_bound']==a['root_bound']==b['root_bound']==f['root_bound']
            assert dimensions(defs[pre['globalTaskId']]['inputShapes_text'])==[[3,4,5120],[24,20480],[3],[24],[3,4]]
            assert [defs[t['globalTaskId']]['opType_text']for t in [pre,norm,post,ffnpre,ffnnorm]]==['HcPre','RmsNorm','HcPost','HcPre','RmsNormCast']
            stages=c['stages'];qa=tasks[stages['Q_A']['task_record_rowid']];qb=tasks[stages['Q_B']['task_record_rowid']];rs=tasks[stages['REDUCE_SCATTER']['task_record_rowid']]
            qdef=defs[qa['globalTaskId']];qbdef=defs[qb['globalTaskId']]
            assert dimensions(qdef['inputShapes_text'])==[[3,5120],[1280,5120]]and qdef['inputDataTypes_text']=='DT_BF16;DT_BF16'
            qbshapes=dimensions(qbdef['inputShapes_text'])
            assert qbshapes[0]==[3,1280]and prod(qbshapes[1])==32768*1280 and dimensions(qbdef['outputShapes_text'])==[[3,32768]]and qbdef['inputDataTypes_text'].startswith('INT8;INT8')
            sw=tasks[stages['SWA_KV']['task_record_rowid']];swdef=defs[sw['globalTaskId']];unique=c['KV_BF16_candidates']==1
            if unique:assert dimensions(swdef['inputShapes_text'])==[[24,5120],[512,5120]]and swdef['inputDataTypes_text']=='DT_BF16;DT_BF16'
            assert pre['endNs']<=norm['startNs']<=qa['startNs']and pre['endNs']<=post['startNs']and post['endNs']<=ffnpre['startNs']<=ffnnorm['startNs']
            tids={pre['task_record_rowid'],norm['task_record_rowid'],post['task_record_rowid'],ffnpre['task_record_rowid'],ffnnorm['task_record_rowid']}|{v['task_record_rowid']for v in stages.values()}
            for tid in tids:used.add((db,tid))
            fronts.append(dict(database_index=db,graph_ordinal=c['graph_ordinal'],block_ordinal=c['block_ordinal'],root_bound=c['root_bound'],
              attention_HcPre_TASK=pre['task_record_rowid'],input_norm_TASK=norm['task_record_rowid'],Q_A_TASK=qa['task_record_rowid'],Q_B_TASK=qb['task_record_rowid'],
              SWA_KV_candidate_TASK=sw['task_record_rowid'],SWA_role_unique=unique,KV_BF16_candidates=c['KV_BF16_candidates'],
              Q_B_recorded_weight_shape=qbshapes[1],Q_B_side_input_shapes=qbshapes[2:],
              output_reduce_scatter_TASK=rs['task_record_rowid'],attention_HcPost_TASK=post['task_record_rowid'],FFN_HcPre_TASK=ffnpre['task_record_rowid'],FFN_norm_TASK=ffnnorm['task_record_rowid'],
              HC_pre_duration_ns=pre['endNs']-pre['startNs'],HC_completion_to_Post_start_observed_span_ns=post['startNs']-pre['endNs'],
              Post_start_minus_reduceScatter_end_ns=post['startNs']-rs['endNs'],Post_end_to_FFNnorm_start_span_ns=ffnnorm['startNs']-post['endNs'],
              hypothetical_coefficient_internal_service_or_available_slack_ns=None,
              native_records_prove_new_branch_overlap=False,actual_pointer_ownership_or_currentCoefficient_input_equality_proved=False))
    assert len(fronts)==3200 and sum(r['root_bound']for r in fronts)==2560
    assert sum(r['SWA_role_unique']for r in fronts)==3120
    assert {r['block_ordinal']for r in fronts if not r['SWA_role_unique']}=={20}
    write(O/'HC-attention-fork-native-frontiers-v93.json',dict(frontiers=fronts,frontier_count=3200,root_bound_frontiers=2560,phase_unbound_frontiers=640,
      strict_unique_SWA_frontiers=3120,strict_root_bound=sum(r['root_bound']and r['SWA_role_unique']for r in fronts),ambiguous_SWA_C1_frontiers=80,
      reused_unique_native_TASKs=len(used),HC_completion_to_Post_start_observed_span_ns=stats([r['HC_completion_to_Post_start_observed_span_ns']for r in fronts if r['root_bound']]),
      observed_spans_are_not_new_overlap_slack_critical_path_or_gain=True,new_native_attribution_or_regime_promotion=False,overall_complete=False))

    planes=[dict(id='HC_current_coeff',dtype='FLOAT/HF32_source_mode',K=20480,N=24,bytes=20480*24*4),
      dict(id='Q_A',dtype='BF16',K=5120,N=1280,bytes=5120*1280*2),dict(id='SWA_KV',dtype='BF16',K=5120,N=512,bytes=5120*512*2),
      dict(id='Q_B',dtype='INT8',K=1280,N=32768,bytes=1280*32768)]
    cuts=[];R0=24*65536
    for count in [3,4]:
        selected=planes[:count]
        for layers in [1,40]:
            for deadline in ['LAST_ATTENTION_POST_REQUIRED_OUTPUTS_READY','FOLLOWING_FFN_PRE_REQUIRED_INPUTS_READY']:
                omitted=4*20480*4 if deadline.startswith('LAST_ATTENTION_POST')else 0
                Wb=sum(p['bytes']for p in selected)*layers-omitted
                # Onlythelastlayerpre4maybesetoutsideaPostdeadline;earlierpre4feedsinterveningFFN/nextlayers.
                perplane=[p['bytes']-(omitted if layer==layers-1 and p['id']=='HC_current_coeff'else 0)for layer in range(layers)for p in selected]
                independent=sum(max(0,w-R0)for w in perplane)/(24*128)
                cuts.append(dict(planes=[p['id']for p in selected],layers_in_window=layers,deadline=deadline,distinct_uncompressed_weight_bytes=Wb,
                  last_pre4_weight_bytes_omitted_from_guaranteed_work=omitted,earlier_pre4_column_groups_still_required=max(0,layers-1),
                  window_start='Beforeanychargedservice/computedpartialresultinthiswindow,withboundedinitialL0grant;notanobservedwhole-tensor-readybarrier.',
                  one_initial_L0B_capacity_grant_bytes=R0,joint_L1_fixedB_min_elapsed_cycles=ceil(max(0,Wb-R0)/(24*128)),
                  independent_per_plane_per_layer_warm_grants_sum_cycle_reference=independent,
                  flexible_orientation_one_L0A_L0B_grant_bytes=2*R0,
                  joint_L1_flexible_orientation_relaxed_min_elapsed_cycles=ceil(max(0,Wb-2*R0)/(24*(256+128))),
                  actual_native_L1_path_or_no_compression_and_weight_identity_proved=False))
    compute=dict(HC_semantic_HF32_ops=2*3*20480*24,Q_A_BF16_ops=2*3*5120*1280,SWA_global24_BF16_ops=2*24*5120*512,Q_B_INT8_ops=2*3*1280*32768)
    capacities=dict(HC_semantic_HF32_ops=24*4096,Q_A_BF16_ops=24*8192,SWA_global24_BF16_ops=24*8192,Q_B_INT8_ops=24*16384)
    column=[]
    for n in [24,20,4]:
        column.append(dict(columns=n,useful_HF32_MAC_ops=2*3*20480*n,
          independently_padded_M16_N16_service_ops=2*16*20480*(16*ceil(n/16)),
          actual_issued_native_count_or_all_fusions_lowerbound=False))
    ring=lambda rowbytes:24*7*rowbytes//8
    write(O/'HC-attention-shared-resource-models-v93.json',dict(planes=planes,cuts=cuts,
      QB_logical_KN_from_activation_output_not_weight_2D_layout=True,QB_recorded_blocked_weight_example=[1024,80,16,32],
      QB_scale_inputs_not_charged_to_L1_L0B_cut='PerchannelDT_BF16[32768]andperrowFLOAT[3]remainrequired. Theirservice/format/epiloguepaths areseparate,notzero;thecurrentweightcutisamandatorysubworkcut.',
      one_current_collapse_and_coefficient_counted_once=True,logical_owner_rows=dict(HC_current=3,Q_A=3,Q_B=3,SWA_KV=24),
      individual_compute_floor_scope='All24currentcoefficientcolumns;Postonlyurgent20hasseparatedeadlinecut. Earlierlayerspre4remainrequiredtofeedlaterlayers.',
      individual_useful_compute_cycle_floors={k:compute[k]/capacities[k]for k in compute},mixed_mode_capacity_simplex_proved=False,
      independent_column_split_reference=column,urgent_post_comb_columns=20,deferred_pre_columns=4,
      source_full_HC_gather_hypothetical_ring_send_bytes=ring(4*5120*2),current_normalized_H_gather_ring_send_reference_bytes=ring(5120*2),
      moving_HC_coeff_to_all24_rows_would_multiply_useful_work_by=8,
      domain='Conventional dense namedoperators;distinctuncompressed mandatoryweightplanes throughdeclared L1→L0 path,24AIC reference,oneinitialwarmgrant/noimportedL0flight. Both40layerdistinctplanes and fullphysicalrowdomain are assumptions,not allalgorithm lowerbounds. GM→L0/compression/pruning/parameterdedup alternatives require newcuts.',
      padding_is_independent_product_reference_not_native_tiling=True,
      counterpart80_layer20_ambiguities_not_relabelled=True,actual_HBM_or_physical_MFU_wallclock_or_speedup_proved=False,overall_complete=False))
    rules={
      'F04':('CURRENT_PRE_IS_FUTURE_FFN_INPUT_NOT_SAME_NORM', 'Currentattnpost/comb gateattentionPost;currentpre feedsfollowingFFNPre thenFFNRmsNormCast. NoCSE ofattention/FFNnorm orusingoldresidual.'),
      'F05':('COEFFICIENT_FORK_COMPETES_WITH_Q_NORM_DQ_QB','Afterexternal-premixY/RMSbranchready,Qnorm/DQ/QBcan beindependentofcurrentcoeff untilPost;shareoneHCwork and finiteVector/Cube/weightcapacities.'),
      'F06':('DIFFERENT_INPUTS_TYPES_AND_OWNER_ROWS','HCcurrentprojection usesfull4H X/FLOAT,QAusesnormalizedBF16Ylocal3,SWAglobal24. Theycannotunconditionallybeonewide same-inputGEMM;jointweightcut androwreplicationcostremain.'),
      'F08':('HC_INPUT_Y_AND_KV_STATE_FORK_NOT_DEAD_COEFFICIENT','SWA/C1norm/RoPE/cache readers retaintheirglobalrows/generations;coefficientYsplitdoesnotdeletepersistentKV/latentstateorchangeHCresidualepoch.'),
      'F14':('ATTENTION_OUTPUT_IS_COEFFICIENT_DEADLINE_PATH','Attention/inverseRoPE/restore/WOA canformtheotherforktosamePost;nativeexposureisnotfreeslack andnewcoeff mustreadybyactualrowread.'),
      'F15':('REDUCE_SCATTER_READINESS_AND_POST_COEFFICIENT_JOIN','OutputcollectiveandcurrentlocalHCcoeffjoinatPost;preserveallrankparticipation,rowplans/streamrdy/sharedmemoryservice,notfreecommoverlap.'),
      'F16':('OUTPUT_PROJECTIONS_SHARE_CAPACITY_BEFORE_HCPOST','WOA/WOB/reduce-scatter andcurrentcoeffsharefiniteAIC/Vector/memorybeforePost;Postcoeffnormalizationfusioncanloseoverlap orduplicateperDtilework.')}
    clauses=[dict(id=f'REL93-{i:02d}',a='F01',b=b,relation=t,operator_claim=claim,
      applicable_domain='Samecurrentattentionblockandversion,externalpremix,matchinglocal/globalrows/parameters/epochs,allPost/nextPre/readers andnativeprecision;noactualnewhandoffproved.',
      evidence=['HC-attention-fork-source-contract-v93.json','HC-attention-fork-native-frontiers-v93.json','HC-attention-shared-resource-models-v93.json','HC-operator-numerical-and-dependency-contract-v59.json'],
      conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None)for i,(b,(t,claim))in enumerate(rules.items(),1)]
    members=['F01','F02','F03','F04','F05','F06','F08','F14','F15','F16']
    packet=dict(clauses=clauses,higher_order_groups=[dict(id='HG93-HC-ATTENTION-FORK-DEADLINE',members=members,
      joint_contract='OneHCcurrentCoeff/externalY forkthroughattentiontoPostandfollowingFFN;differentinputs/rowownership andreaders,mixedresources/onewarmgrant. Neitherkernelcounts norindependentboundssumprovesgain.',all_variants_closed=False)],new_maincandidate=False,overall_complete=False)
    existing=read(O/'candidate-relation-clauses-v92.json')['clauses'];assert all(not any({r['a'],r['b']}=={c['a'],c['b']}for r in existing)for c in clauses)
    write(O/'HC-attention-composition-clauses-v93.json',packet)
    print(dict(frontiers=len(fronts),strictSWA=3120,ambiguous=80,reused_native_tasks=len(used),joint_weight_cuts=len(cuts),new_pairs=7,new_groups=1,overall_complete=False))


if __name__=='__main__':main()
