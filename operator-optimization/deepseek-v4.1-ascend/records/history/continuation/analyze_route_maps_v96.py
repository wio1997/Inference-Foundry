"""Offline analysis of MC2 grouping, inverse permutation and TopK handoff semantics."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,math
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_route_quant_v95 import BASE,chunks
from analyze_state_engram_v62 import dimensions

def inverse_scatter(sigma):
    out=[None]*len(sigma)
    for dst,src in enumerate(sigma):
        assert 0<=src<len(sigma)and out[src]is None
        out[src]=dst
    return out

def line_bytes(start,count,offset):
    return 0 if count==0 else ((offset+4*(start+count)-1)//32-(offset+4*start)//32+1)*32

def main():
    assert not(O/'REVIEW-CHECKPOINT-v96.json').exists()
    stage=read(O/'MC2-operator-inputs-v56.json')['operator_source']
    dispatch=Path(next(r['local_path']for r in stage if r['path'].endswith('dispatch_ffn_combine_kernel.hpp')))
    gating=Path(read(O/'router-vector-numerical-contract-v63.json')['parts'][0]['path'])
    unperm=BASE.parent/'unpermute/moe_token_unpermute.h'
    files={BASE/'moe_v2_fullload_dynamic_quant.h':[(86,218),(263,308),(318,394)],
      BASE/'moe_v2_src_to_dst_op.h':[(66,104)],dispatch:[(360,402),(445,462),(793,812),(965,978)],
      gating:[(244,270)],unperm:[(243,280)]}
    parts=[]
    for p,spans in files.items():
        lines=p.read_text().splitlines();parts.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
          segments=[dict(first_line=a,last_line=b,source=lines[a-1:b])for a,b in spans]))
    text=(BASE/'moe_v2_fullload_dynamic_quant.h').read_text()
    section=text[text.index('::SortCompute()'):text.index('::CopyOutIdx()')]
    assert section.count('Sort<float, true>')==2
    write(O/'route-map-operator-source-contract-v96.json',dict(parts=parts,
      facts=[
        'Full-load21000 each active quant partition reads the entire expert-ID vector,executes first expert grouping Sort,then second Sort on unique original assignment indices to construct inverse. Process invokes this before quantized row copies.',
        'The first output sigma maps sorted position to source assignment. Current second sort keys -sigma[j] and payload j;because sigma is a permutation,its valid result is inverse[sigma[j]]=j.',
        'Keeping each core first Sort sigma unchanged permits a direct local inverse construction without assuming native expert-tie stability. It does not identify a fast native implementation.',
        'Non-full gathered branch already has SrcToDst mapping by direct indexed writes;large816/834 cohorts are not credited with removal of these two full-load sorts.',
        'ApplyXActiveMask mutates expert IDs in GM to sentinel E for inactive token rows and synchronizes before routing initialization. Producer fusion must use effective post-mask IDs and preserve externally observed mutation/generation.',
        'Unpermute reads inverse indices and probs in original flattened token/topK order.Receiver maxOutputSize clipping makes arbitrary within-expert reorder unsafe unless nonbinding or exact original ordering retained.',
        'Router TopK sorts per-row biased scores;MC2 grouping sorts post-mask expert IDs across assignments. These are distinct key domains,not duplicate sorts of the same values.'
      ],algebraic_inverse_status='DERIVED_IDENTITY_IN_PERMUTATION_AND_EXACT_INDEX_DOMAIN',
      loaded_native_binary_or_new_scatter_performance_attested=False,new_target_source_files=0,overall_complete=False))
    old=read(O/'route-quant-tiling-and-work-models-v95.json');small=[r for r in old['cases']if r['expected_source_key']==21000]
    cases=[];costs=[];joint=[]
    for r in small:
        M,k=r['M'],r['topK'];T=M*k;C=r['used_quant_AIV_partitions'];p=r['per_core_assignment_rows'];assert T<32
        baseline=4*T*C+4*T;shared=12*T
        cases.append(dict(M=M,topK=k,T=T,current_partitions=C,saved_executions=r['saved_executions'],
          high_level_full_load_Sort_calls_reference=2*C,inverse_only_Sort_calls_reference=C,
          source_ID_copy_value_bytes=4*T*C,existing_global_inverse_value_write_bytes=4*T,
          shared_mapping_baseline_value_bytes=baseline,shared_mapping_transformed_value_bytes=shared,
          conditional_shared_mapping_GM_value_difference=baseline-shared,
          full_local_inverse_UB_value_writes=4*T*C,consumer_demand_local_inverse_UB_writes=4*(2*T-p),
          histogram_namespace_with_sentinel=385 if k==6 else 129,
          counts_have_one_writer_in_current_full_load=True,
          native_instruction_or_physical_speedup_claim=False))
        for ao in range(0,32,4):
            for bo in range(0,32,4):
                ids=line_bytes(0,T,ao);out=line_bytes(0,T,bo)
                new_reads=sum(line_bytes(a,b-a,bo)for a,b in r['core_assignment_intervals'])
                costs.append(dict(M=M,topK=k,ID_base_offset_mod32=ao,map_base_offset_mod32=bo,
                  baseline_copy_block_exposure_bytes=C*ids+out,shared_copy_block_exposure_bytes=ids+out+new_reads,
                  difference=(C*ids+out)-(ids+out+new_reads),
                  model='32B block touch per copy with no inter-copy cache/coalescing;notactualGM/HBMtransactions or a physical bound.'))
        for aligned in [False,True]:
            cs=(math.ceil(M/math.ceil(M/40))if aligned else C)
            qr=(M if aligned else r['quant_row_evaluations'])
            for shared_first in [False,True]:
                for scatter in [False,True]:
                    sorts=(1 if shared_first else cs)*(1 if scatter else 2)
                    joint.append(dict(M=M,topK=k,token_aligned=aligned,shared_first_permutation=shared_first,local_inverse_direct=scatter,
                      route_partition_count=cs,source_style_quant_row_evaluations=qr,high_level_Sort_calls_reference=sorts,
                      source_native_schedule_or_all_gates_closed=False))
    write(O/'route-map-source-work-and-interfaces-v96.json',dict(cases=cases,full_load_executions=sum(r['saved_executions']for r in cases),
      excluded_non_full_load_executions=old['saved_executions']-sum(r['saved_executions']for r in cases),
      current_source_weighted_Sort_call_reference=sum(r['saved_executions']*r['high_level_full_load_Sort_calls_reference']for r in cases),
      inverse_only_weighted_Sort_call_reference=sum(r['saved_executions']*r['inverse_only_Sort_calls_reference']for r in cases),
      block_exposure_cases=costs,joint_partition_and_map_cases=joint,
      scope='Source program work and value-interface alternatives,not profiled internal instruction counts. Existing globalinverse buffer can publish one common map only with deterministic matching grouping/order,all writers/readers/SyncAll and rank readiness.',
      original_MoE_GM_payload_and_weights_unchanged=True,actual_HBM_or_speedup_proved=False,overall_complete=False))
    # Native join binds only the visible producer/consumer boundary;inner Sorts are not separate profiled TASKs.
    tops=read(O/'router-vector-graph-chains-v63.json')['chains'];routes={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in read(O/'route-quant-native-frontiers-v95.json')['frontiers']}
    bydb=defaultdict(list)
    for r in tops:bydb[r['database_index']].append(r)
    fronts=[];used=set()
    for db,rs in bydb.items():
        d=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');ts={t['task_record_rowid']:t for t in d['TASK_records']};defs={t['globalTaskId']:t for t in d['COMPUTE_definitions']}
        for r in rs:
            key=(db,r['graph_ordinal'],r['block_ordinal']);rt=routes[key];top=ts[r['router_stages']['topk']['TASK_rowid']];route=ts[rt['MC2_TASK']];router=ts[r['router_stages']['router']['TASK_rowid']]
            td=defs[top['globalTaskId']];rd=defs[route['globalTaskId']]
            assert td['opType_text']=='MoeGatingTopKHash'and dimensions(td['outputShapes_text'])==[[3,6],[3,6],[3,384]]
            assert top['endNs']<=route['startNs']and top['streamId']==route['streamId']
            for t in [top,route,router]:used.add((db,t['task_record_rowid']))
            fronts.append(dict(database_index=db,graph_ordinal=key[1],block_ordinal=key[2],root_bound=rt['root_bound'],
              router_TASK=router['task_record_rowid'],topK_TASK=top['task_record_rowid'],MC2_TASK=route['task_record_rowid'],
              topK_output_types=td['outputDataTypes_text'],topK_output_shapes=dimensions(td['outputShapes_text']),
              full_score_descriptor_not_new_write_proof=True,topK_completion_to_MC2_start_observed_ns=route['startNs']-top['endNs'],
              inner_sort_timing_or_postmask_values_or_native_permutation_proved=False))
    write(O/'route-map-native-frontiers-v96.json',dict(frontiers=fronts,native_frontiers=len(fronts),root_bound=sum(r['root_bound']for r in fronts),reused_unique_native_TASKs=len(used),
      inner_sort_not_new_profile_TASKs=True,no_global_assignment_or_regime_promotions=True,observed_gaps_not_free_slack_or_gain=True,overall_complete=False))
    raw=[0,1];post=[384,1]
    rawsig=sorted(range(2),key=raw.__getitem__);postsig=sorted(range(2),key=post.__getitem__)
    model=dict(inverse_identity='For any permutation sigma of[0,T), original second-sort output pi[sigma[j]]=j. Direct scatter has exactly the same pi;first expert grouping and its ties stay unchanged.',
      valid_index_domain='T<2^24 for current float index conversions;current smallT=6/9/18. Valid expert IDs inclsentinel have higher sorting keys than MIN_FP32 padding.',
      inverse_prefix_reader_contract='GlobalCopyOutIdx writes onlyT int32s. Each partition quant loop reads inverse positions in its own[lo,hi) assignment interval. Core0 still publishes allT;othercores may use exact local windows if their indexing is adapted.',
      mask_witness=dict(raw_ids=raw,active_mask=[False,True],sentinel=384,effective_postmask_ids=post,
        inverse_before_mask=inverse_scatter(rawsig),inverse_after_mask=inverse_scatter(postsig),
        raw_nonzero_counts={'0':1,'1':1},effective_nonzero_counts={'1':1,'384':1}),
      capacity_witness=dict(expert_ids=[0,0],token_values=[1,2],receiver_capacity=1,
        first_tie_order=[0,1],reordered_tie_order=[1,0],outputs_after_inverse_first=[1,0],outputs_after_inverse_reordered=[0,2],
        scope='Generic rowwise identity-expert model with clipping;notcaptured overflow. Shows why arbitrary tie reordering needs nonbinding capacity or exact-order proof.'),
      equal_histogram_not_equal_inverse_witness=dict(effective_ids_a=[0,1],effective_ids_b=[1,0],histogram=[1,1],
        inverse_a=[0,1],inverse_b=[1,0]),
      counts_cache_scope='Current flag2 cumulative counts depend on exact effective-ID histogram and expert namespace/output mode,not assignment order. Verifying equality and publishing currentepoch still cost work. Equal counts do not imply equal inverse map.',
      known_inactive_suffix_formula='If P-N tail token masks are known false:hist(postmaskP)=hist(effective_prefixN)+(P-N)*k*unit_sentinel. Preserve all original map/record/mutation outputs;this formula alone is not row pruning.',
      cache_scope='Pure inverse mapping reuse needs equal effective post-mask ID vector,shape/k/expert namespace,group/drop/capacity/order policy and valid publication epoch. It doesnotreusehidden values,quantizedrows,expert outputs or router scores merelybecauseIDs match.',
      fixed_representation_symbolic_cut=dict(mandatory_fresh_global_inverse_prefix_bytes='4*T',rate_upper=None,
        time_lower_bound='T_publish >=4*T/B_upper_GM_map_publication',
        domain='Original fresh explicit inverse GM buffer for original unpermute readers;new implicit/permutation ABI needs a new model,not an unconditional allalgorithm/HBM bound.'),
      no_second_sort_is_semantically_necessary_in_this_inverse_problem=True,whole_MC2_or_native_legality_closed=False,overall_complete=False)
    write(O/'route-map-semantic-and-ceiling-contract-v96.json',model)
    write(O/'route-map-variants-v96.json',dict(variants=[
      dict(id='RM96-LOCAL-INVERSE',decision='SCOPED_ALGEBRAIC_IDENTITY;NATIVE_COST_PENDING',contract='KeepfirstSortsigmaunchanged;constructinverse[sigma[j]]=j locally. Remove second inverse Sort as necessary work in this representation domain;private alias/lifetime/backend cost still bind.'),
      dict(id='RM96-PARTIAL-INVERSE',decision='CONDITIONAL_READER_DEMAND_SPECIALIZATION',contract='Core0 constructs full inverse forglobaloutput;othercores onlytheir consumed assignment windows,with exactlocal indexadaptation. Full firstsort and all original outputs/counts remain.'),
      dict(id='RM96-SHARED-MAP',decision='CONDITIONAL_NEW_PUBLICATION',contract='One grouping/inverse producer publishes existingglobalmap;allquantcores readonlytheir partition. Requires matching deterministicties/mutation epoch/counters/visibility and costsnewreads/waits.'),
      dict(id='RM96-TOPK-METADATA',decision='CONDITIONAL_NEW_ABI',contract='TopK producer may build grouped map/histogram/prefix after preserving active-mask/sentinel transition. ScoreTopK and IDgroup sorts are different. Nativeorder/clipping/probability return and externally observedIDmutation retained.'),
      dict(id='RM96-CROSS-ITERATION-MAP',decision='CONDITIONAL_VALUE_KEYED_METADATA_ONLY',contract='Reuseinverse maps only for identical effectiveIDvectorandallrow/order/capacity/namespaceepochs. Recomputehidden quant/expertvalues;do notinferrepeatedIDsfromshape/addressorcacheweightoutputs.'),
      dict(id='RM96-COUNT-SUFFIX',decision='CONDITIONAL_FLAG2_HISTOGRAM_REUSE',contract='Cumsumcanuseexacthistogramequalitywithoutassignment-order equality. Knownfalsepaddingtailaddsk*(P-N)sentinelcounts;preservefullmaps/recordoutputs/mutationandcurrentpublication,priceverificationandnewbufferwork.')],
      new_maincandidate=False,implementation=False,native_test=False,overall_complete=False))
    print(json.dumps(dict(native_frontiers=len(fronts),native_tasks_reused=len(used),full_load_calls=sum(r['saved_executions']for r in cases),
      source_Sort_calls=sum(r['saved_executions']*r['high_level_full_load_Sort_calls_reference']for r in cases),
      inverse_only_Sort_calls=sum(r['saved_executions']*r['inverse_only_Sort_calls_reference']for r in cases),block_exposure_cases=len(costs),joint_cases=len(joint),overall_complete=False)))

if __name__=='__main__':main()
