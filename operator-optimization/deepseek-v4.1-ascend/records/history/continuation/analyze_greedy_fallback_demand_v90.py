"""Pin two greedy fallback operator bodies and model exact output/argmax demand."""
import ast,hashlib
from copy import deepcopy
from collections import Counter,defaultdict
from pathlib import Path
from math import prod
from finalize_origin_coverage_v9 import N,O,R,read,write
from reconcile_operator_graph_coverage_v88 import connected_sets


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not(O/'REVIEW-CHECKPOINT-v90.json').exists()
    prior=read(O/'M17-pytorch-fallback-source-v49.json');p=Path(prior['path']);text=p.read_text();lines=text.splitlines();tree=ast.parse(text)
    assert sha(p)==prior['sha256']=='bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda'
    wanted=['rejection_greedy_sample_spec_len_1_pytorch','rejection_greedy_sample_pytorch'];funcs=[];guards=[]
    for f in tree.body:
        if not isinstance(f,ast.FunctionDef)or f.name not in wanted:continue
        funcs.append(dict(name=f.name,first_line=f.lineno,last_line=f.end_lineno,source=lines[f.lineno-1:f.end_lineno]))
        for n in ast.walk(f):
            if isinstance(n,ast.If):guards.append(dict(function=f.name,line=n.lineno,test=ast.unparse(n.test),arms=['BODY','ORELSE_OR_FALLTHROUGH']))
    assert len(funcs)==2 and len(guards)==6
    caller=next(f for f in tree.body if isinstance(f,ast.FunctionDef)and f.name=='rejection_sample')
    route=next(n for n in ast.walk(caller)if isinstance(n,ast.If)and any(isinstance(c,ast.Call)and isinstance(c.func,ast.Name)and c.func.id==wanted[0]for stmt in n.body for c in ast.walk(stmt)))
    # Take the smallest enclosing condition: the immediate specialization,not outer !all_random.
    matches=[n for n in ast.walk(caller)if isinstance(n,ast.If)and any(isinstance(stmt,ast.Expr)and isinstance(stmt.value,ast.Call)and isinstance(stmt.value.func,ast.Name)and stmt.value.func.id==wanted[0]for stmt in n.body)]
    assert len(matches)==1;route=matches[0];route_test=ast.unparse(route.test)
    assert 'max_spec_len'not in {n.id for n in ast.walk(route.test)if isinstance(n,ast.Name)}
    write(O/'greedy-fallback-source-contract-v90.json',dict(path=str(p),sha256=sha(p),functions=funcs,local_if_cases=guards,
      local_if_sites=6,local_if_arms=12,source_capture_is_preexisting=True,new_target_source_files=0,
      caller_specialization=dict(line=route.lineno,test=route_test,source=lines[route.lineno-1:route.end_lineno],
        requires_output_capacity_equals_one=False,condition_requires_actual_counts_one=True,
        outer_reachability='not all_random AND not HAS_TRITON;specialization also all_greedy,B>0 for caller min/max;valid metadata n<=S.'),
      facts=[
        'SpecializedhelperassertsB==N,notoutputwidth==2. Callerchecksmin/max actual draftcounts==1 andall_greedy;doesnotcheckmax_spec_len==1. ForS>1onlycols0/1arewritten,suffixstilldependsontheoriginalfill.',
        'Specializednormalwritescol0targetArgmax;col1where(accept,bonus,oldcol1). Syntheticaccept=(U<rate0)&draft>=0;col0isacceptedDraftorfailedTarget. It writesallBentriesofcol1evenwhenunchanged.',
        'Generalhelperconstructsdevicecounts,startindices,repeat_interleavetokenRequestIDs andpositions,thenposition/mismatch/sentinelmatricesB*S forS>0;ordinaryusesfirstmismatch,syntheticfirstnotaccepted.',
        'Generalcopylength=min(firstMismatch+1,n);maskgreedyBEFOREgather. SyntheticcopiesdraftforacceptedpositionsandtargetArgmaxforfirstreject. Bonusiswrittenatnonlyifallaccepted;tailsandnongreedyrowsretainpreviousoutput.',
        'S==0branchsetsfirstMismatch0;withvalidn=0,greedyrowsgetbonusat0. B==0caller fallbackmin([]) isnot silently made valid by a body-only rewrite.',
        'Calleruniform_probs_for_greedy isgeneratedforallN andconvertedFLOAT32 beforeeitherbackend. Mixednon-greedypathlaterdrawsaseparateuniformbankandrequestQ;thoseRNGcallsareeffects,notCSE candidates.',
        'Only2directgreedybodiesare newly reviewedhere;4random/recoverybodieswerealreadyreviewedv49. Current72greedyTritonlaunchesdo notattestexecutionofthesePyTorchbodies or actualsynthetic/allgreedy flags.'
      ],
      preconditions=['B>0callerwhere min/max used;validboundedS/counts,cumulativepartitionN=sum(n),same ordered row and bonus plan.',
        'Exact provided row-argmax values/comparator/ties/NaNs preserved;inputs readonly and no unproved output/input alias.',
        'Synthetic uniforms retain the caller FLOAT32 conversion;rates retain original dtype/values and comparison promotion. All original RNGcalls/state advancement retained;keepvalidation/errors.',
        'Replacing old-output retention by PLACEHOLDER requires immediate caller-filled bank and no interveningwriter;generichelper retainsarbitraryexistingtails.'],
      whole_sampler_branch_universe_closed=False,native_backend_or_flags_attested=False,overall_complete=False))
    write(O/'greedy-fallback-demand-algebra-v90.json',dict(
      ordinary='For each greedy request withn drafts,letj befirstdraft!=Argmax position,or n. The demand for exact rowargmax ismin(j+1,n). Non-greedyrequestsneed0inthisstage;allnon-greedyprobability/scoreobserversareseparate.',
      synthetic='Acceptance depends on the existing uniform/rate and draft>=0,not targetArgmax. Letj befirstfailedaccept,or n. Onlywhenj<n is rowArgmax[j] needed. Acceptedprefix emitsdraft andallaccepted emitsbonus,soatmost one argmaxrow per greedy request.',
      distinction_from_prior_v33='v33 lazy random recovery selected full-vocabulary residual races. This demand projection concerns the greedy targetArgmax producer and the two newly reviewed fallback bodies;no residual probability sampler is replaced by argmax.',
      ordinary_prefix_priority='Ordinary lazy argmax/prefix demand already appears in v65. Here it is an independently checked body-specific refinement,not a new ordinary optimization discovery. New detailed domains include actualn1vsScapacity and synthetic zero/one targetArgmax demand.',
      exact_pointwise_domain='Forfixed bounded inputs,providedexactper-rowargmaxresults,existingrandomdraws and oldoutputbank. Native new-reduction parity,alias/capture and latency are separate gates.',
      full_write='For callerbank already initializedPLACEHOLDER,newwriter canwriteallB*(S+1)cells directly. Generalbaseline fill+Wsemanticupdates vsnewfullwrite saves4W outputstore-referencebytes,not automatically4B(S+1);specializedsourcewrites2B cells then leavesS-1tailcolumns.',
      input_work='Baselineworking-FLOATargmax references4NV scorebytes. Conditional row-demandplanreferences4AV. These are value queries/logicalreferences,not compulsoryHBMbytes or a latencygain.',
      composition_with_v65='In the guarded BF16Index→FLOATCast→workingclone→Argmax22NV reference,pruningonlyArgmax gives18NV+4AV. A joint pure-ID producer-consumer variant can read2AV,with22NV-2AV combinedcredit. Do notaddold20NV fusioncredit andnew4V(N-A) argmaxcredit;they shareonebaseline andcanovercount. This jointcase requiresallgreedy/nootherraw,processor,entropy,probability,logprob readers,retained source/index epoch and exactBF16widening/comparator semantics;mixedrows cannot inherit it globally.',
      theoretical_limit='Exact argmax over arbitrary unstructured V scores needs inspection of allV values in the worstcase unless an index/order/pruningcertificate isprovided. Each chosen row remains a full-vocab reduction. High-level argmax instruction capacity isnot inferredfromVectorMax64.',
      parallel_tradeoff='Ordinaryprefix demand can serialize row reductions and reduceparallelism. Syntheticdecisionmapcan precede onlyselectedrowargmax. Newlaunch/packing/handshake costs and otherlogitreaders must bepriced.',
      implementation=False,native_validation=False,overall_complete=False))
    cost=[]
    for x in read(O/'heads-sampling-fusion-costs-v65.json')['sampling_models']:
        B,T,V,S=x['B'],x['T_capacity'],x['V'],x['Kmax']
        # Shapes are recorded capacity;N and n are scenario parameters,not recovered flags.
        ns=[min(S,max(0,T-i*S))for i in range(B)];assert sum(ns)==T and all(n<=S for n in ns)
        scenarios=[('full_capacity_all_accept',ns,[True]*B,None),
          ('full_capacity_first_reject',ns,[True]*B,'FIRST'),
          ('actual_one_draft_each_capacity_still_5',[1]*B,[True]*B,None),
          ('actual_one_draft_each_first_reject',[1]*B,[True]*B,'FIRST'),
          ('mixed_first_reject',ns,[i%2==0 for i in range(B)],'FIRST'),
          ('zero_draft',[0]*B,[True]*B,None)]
        for label,n,greedy,reject in scenarios:
            Nvalid=sum(n);m=[(0 if reject=='FIRST'and nr else nr)for nr in n]
            ordinary=sum(min(j+1,nr)for nr,j,g in zip(n,m,greedy)if g)
            synthetic=sum(j<nr for nr,j,g in zip(n,m,greedy)if g)
            W=sum((min(j+1,nr)+(j>=nr))for nr,j,g in zip(n,m,greedy)if g)
            specialized=all(nr==1 for nr in n)and all(greedy)
            helper_stores=2*B if specialized else W
            cost.append(dict(B=B,T_capacity=T,V=V,S=S,scenario=label,actual_count_scenario=n,greedy_scenario=greedy,N_valid_scenario=Nvalid,
              scenario_input_N_matches_recorded_sampler_capacity=(Nvalid==T),
              changed_N_scenarios_are_unobserved_shape_branches=(Nvalid!=T),
              ordinary_demand_argmax_rows=ordinary,synthetic_demand_argmax_rows=synthetic,
              baseline_FLOAT_argmax_score_reference_bytes=4*Nvalid*V,
              ordinary_argmax_score_reference_bytes=4*ordinary*V,synthetic_argmax_score_reference_bytes=4*synthetic*V,
              guarded_Index_Cast_clone_argmax_22NV_reference_bytes=22*Nvalid*V,
              ordinary_only_argmax_pruned_18NV_plus_4AV_reference_bytes=(18*Nvalid+4*ordinary)*V,
              synthetic_only_argmax_pruned_18NV_plus_4AV_reference_bytes=(18*Nvalid+4*synthetic)*V,
              pure_ID_composition_greedy_subdomain_scenario=all(greedy),
              pure_ID_joint_ordinary_2AV_reference_bytes=2*ordinary*V if all(greedy)else None,
              pure_ID_joint_synthetic_2AV_reference_bytes=2*synthetic*V if all(greedy)else None,
              pure_ID_joint_ordinary_credit_reference_bytes=(22*Nvalid-2*ordinary)*V if all(greedy)else None,
              pure_ID_joint_synthetic_credit_reference_bytes=(22*Nvalid-2*synthetic)*V if all(greedy)else None,
              joint_pure_ID_reader_epoch_rounding_guards_proved=False,
              source_general_three_BS_value_planes_reference_bytes=17*B*S,
              source_general_global_idx_value_plane_reference_bytes=8*B*(S+1),
              these_metadata_plane_costs_apply_only_if_general_helper_reached=not specialized,
              specialized_actual_one_count_not_S1=specialized,
              source_specialized_untouched_suffix_cells=B*max(0,S-1)if specialized else None,
              original_caller_fill_store_reference_bytes=4*B*(S+1),
              original_helper_store_reference_bytes=4*helper_stores,
              original_fill_plus_helper_store_reference_bytes=4*(B*(S+1)+helper_stores),
              transformed_fullwriter_store_reference_bytes=4*B*(S+1),
              fullwrite_output_store_credit_bytes=4*helper_stores,
              current_actual_flags_counts_or_new_speedup_attested=False))
    write(O/'greedy-fallback-demand-cost-models-v90.json',dict(models=cost,models_count=len(cost),profile_capacity_geometries=3,
      Vector_UB_per_AIV=196608,FLOAT_full_vocab_row_bytes=4*129280,BF16_full_vocab_row_bytes=2*129280,
      capacity_reference='A3-local-capacity-and-ceiling-scope-v80.json',
      full_FLOAT_or_BF16_row_fits_single_AIV_UB=False,
      excluded_plan='OneAIVretainsentireuncompressedFLOATorBF16V129280rowinUBbeforeargmax. Streaming/chunks/multiAIV remaineligible;notfusionimpossibility.',
      all_counts_are_declared_scenarios_not_native_ragged_values=True,logical_planes_not_measured_allocations_or_HBM=True,
      actual_instruction_capacity_or_wallclock_speedup_proved=False,overall_complete=False))

    # Supplement the frozen finite graph catalog with the direct greedy-body primitives.
    templates=[]
    def template(tid,spec,edges):
        nodes=[dict(id=n,kind=k,resource=r,source_scope='greedy-fallback-source-contract-v90.json')for n,k,r in spec]
        es=[dict(producer=a,consumer=b,value=v,dependency_kind=kind,condition=cond,
          source='greedy-fallback-source-contract-v90.json',grade='PINNED_SOURCE_OR_DECLARED_EFFECT_ORDER')for a,b,v,kind,cond in edges]
        templates.append(dict(id=tid,nodes=nodes,edges=es,native_occurrence_proved=False,all_sampler_primitives_complete=False))
    template('SAMPLER_GREEDY_FALLBACK_ROUTE',[
      ('working_logits','FLOAT_WORKING_SCORE_BANK','GM'),('row_argmax','FULL_VOCAB_ROW_ARGMAX','VECTOR_REDUCTION'),
      ('metadata','COUNTS_CU_FLAGS_S_BONUS','META'),('placeholder_fill','INT32_FILL_BxSPLUS1','VECTOR_STORE'),
      ('guard','BACKEND_AND_ACTUAL_COUNT_GUARD','CONTROL'),('specialized','ACTUAL_N_ONE_GREEDY_BODY','VECTOR'),
      ('general','GENERAL_RAGGED_GREEDY_BODY','VECTOR'),('rng_old','RNG_STATE_OLD','STATE'),
      ('uniform_draws','EXISTING_SYNTHETIC_GREEDY_UNIFORMS','RNG'),('uniform_F32','F64_TO_F32_EXISTING_CONVERSION','VECTOR_CAST'),
      ('rng_after_greedy','RNG_STATE_AFTER_GREEDY_DRAWS','STATE'),('later_random_rng','DISTINCT_LATER_RANDOM_DRAWS_IF_REACHED','RNG'),
      ('output_after_greedy','OUTPUT_EPOCH_AFTER_GREEDY_WRITES','GM'),('later_random_writer','NONGREEDY_RANDOM_PHASE_NOT_IN_THIS_BODY','BOUNDARY'),
      ('output_final','OUTPUT_FINAL_EPOCH','GM')],
      [('working_logits','row_argmax','allNworkingFLOATrows','RAW','not all_random'),
       ('metadata','guard','actualcounts1/allgreedy/HAS_TRITON/S','CONTROL','not all_random AND not HAS_TRITON'),
       ('guard','specialized','allactualn1 ANDallgreedy;Sneednot1','CONTROL','specialized guard'),
       ('guard','general','other reachedgreedyfallback','CONTROL','not specialized'),
       ('row_argmax','specialized','NrowIDs','RAW','specialized guard'),('row_argmax','general','NrowIDs','RAW','general guard'),
       ('placeholder_fill','specialized','oldcol1andtails','STATE_EPOCH','specialized guard'),('placeholder_fill','general','oldtails/nongreedyrows','STATE_EPOCH','general guard'),
       ('rng_old','uniform_draws','state/drawnamespace','EFFECT_ORDER','synthetic ANDnot all_random'),
       ('uniform_draws','uniform_F32','default+seededuniformvalues','RAW','synthetic ANDnot all_random'),
       ('uniform_draws','rng_after_greedy','advancedgenerators','EFFECT_ORDER','existing draws'),
       ('rng_after_greedy','later_random_rng','samefuturegeneratorstate','EFFECT_ORDER','nongreedyrandomphase reached'),
       ('uniform_F32','specialized','sameUvalues','RAW','synthetic specialized'),('uniform_F32','general','sameUvalues','RAW','synthetic general'),
       ('specialized','output_after_greedy','cols0/1only;tailsold','STATE_EPOCH','specialized guard'),
       ('general','output_after_greedy','greedyprefixandbonus;tailsold','STATE_EPOCH','general guard'),
       ('output_after_greedy','output_final','earlyreturnbank','STATE_EPOCH','allgreedy'),
       ('output_after_greedy','later_random_writer','preservegreedyrows','STATE_EPOCH','not allgreedy'),
       ('later_random_rng','later_random_writer','distinct laterU/Q','EFFECT_ORDER','randomphase reached'),
       ('later_random_writer','output_final','randomwritesnongreedyrows','STATE_EPOCH','randomphase reached')])
    template('SAMPLER_GREEDY_GENERAL_BODY',[
      ('counts_cu','COUNTS_CU_N_S','META'),('device_counts','COUNT_TENSOR_AND_STARTS','VECTOR_META'),
      ('rowmap','REPEAT_INTERLEAVE_REQUEST_AND_TOKEN_POSITIONS','VECTOR_META'),('draft_ids','DRAFT_IDS','GM'),('target_argmax','EXACT_TARGET_IDS','GM'),
      ('uniform_rates','EXISTING_FLOAT32_U_AND_ORIGINAL_RATE_VALUES','GM'),('mismatch','NORMAL_COMPARE_OR_SYNTHETIC_ACCEPT_NOT','VECTOR_COMPARE'),
      ('pos_matrix','INT64_POSITION_SCATTER_BS','VECTOR_META'),('mismatch_matrix','BOOL_MISMATCH_SCATTER_BS','VECTOR_META'),
      ('sentinel_matrix','INT64_WHERE_MISMATCH_ELSE_2S','VECTOR_SELECT'),('first_mismatch','ROW_MIN_AND_NOFAIL_REPLACEMENT','VECTOR_REDUCTION'),
      ('zeroS','S0_FIRST_MISMATCH_ZERO','CONTROL'),('copylen','MIN_FIRSTPLUS1_N','VECTOR_META'),
      ('output_grid','ARANGE_EXPAND_AND_GLOBAL_IDX','VECTOR_META'),('greedy_mask','GREEDY_REQUEST_MASK','META'),
      ('copy_mask','COPYLEN_AND_GREEDY_MASK','VECTOR_COMPARE'),('copy_tokens','NORMAL_TARGET_OR_SYNTHETIC_DRAFT_TARGET_WHERE','VECTOR_SELECT'),
      ('old_output','OLD_OUTPUT_EPOCH','GM'),('prefix_write','MASKED_GATHER_CAST_STORE','VECTOR_STORE'),
      ('needs_bonus','GREEDY_AND_FIRST_GE_N','VECTOR_COMPARE'),('any_bonus','SOURCE_TENSOR_ANY_HOST_CONDITION','CONTROL'),
      ('bonus_write','BONUS_AT_COLUMN_N','VECTOR_STORE'),('new_output','NEW_OUTPUT_EPOCH_RETAIN_OTHER_CELLS','GM')],
      [('counts_cu','device_counts','n,cu','RAW','body reached'),('device_counts','rowmap','starts/counts','RAW','body reached'),
       ('draft_ids','mismatch','draftID','RAW','body reached'),('target_argmax','mismatch','targetID','RAW','not synthetic'),('uniform_rates','mismatch','U<rate(position)','RAW','synthetic'),
       ('rowmap','pos_matrix','tokenrequest/tokenposition','RAW','S>0'),
       ('rowmap','mismatch_matrix','tokenrequest/tokenposition','RAW','S>0'),('mismatch','mismatch_matrix','per-tokenmismatch','RAW','S>0'),
       ('pos_matrix','sentinel_matrix','candidatepositions','RAW','S>0'),('mismatch_matrix','sentinel_matrix','mismatchmask','RAW','S>0'),
       ('sentinel_matrix','first_mismatch','rowmin','RAW','S>0'),('device_counts','first_mismatch','nofailreplacementn','RAW','S>0'),
       ('counts_cu','zeroS','S0','CONTROL','S==0'),('zeroS','copylen','first0','RAW','S==0'),
       ('first_mismatch','copylen','first+1','RAW','S>0'),('device_counts','copylen','n','RAW','body reached'),
       ('device_counts','output_grid','starts','RAW','body reached'),('copylen','copy_mask','copylength','RAW','body reached'),
       ('output_grid','copy_mask','columns','RAW','body reached'),('greedy_mask','copy_mask','activegreedy','RAW','body reached'),
       ('draft_ids','copy_tokens','accepteddraftvalues','RAW','synthetic'),('target_argmax','copy_tokens','targetvalues','RAW','body reached'),('mismatch','copy_tokens','syntheticaccept','RAW','synthetic'),
       ('copy_mask','prefix_write','maskBEFOREindexgather','RAW','body reached'),('output_grid','prefix_write','maskedglobalIDs','RAW','body reached'),
       ('copy_tokens','prefix_write','selectedvalues/cast','RAW','body reached'),('old_output','prefix_write','oldepoch','STATE_EPOCH','body reached'),
       ('first_mismatch','needs_bonus','first>=n','RAW','S>0'),('zeroS','needs_bonus','0>=n','RAW','S==0'),
       ('device_counts','needs_bonus','n','RAW','body reached'),('greedy_mask','needs_bonus','isgreedy','RAW','body reached'),
       ('needs_bonus','any_bonus','torch.any','CONTROL','body reached'),('any_bonus','bonus_write','sourcebranch','CONTROL','anytrue'),
       ('prefix_write','bonus_write','orderedbankupdates','STATE_EPOCH','anytrue'),('device_counts','bonus_write','columnn/bonusplan','RAW','anytrue'),
       ('prefix_write','new_output','prefixwrites/othersretained','STATE_EPOCH','no bonus'),('bonus_write','new_output','finalwrites/othersretained','STATE_EPOCH','anytrue')])
    template('SAMPLER_GREEDY_ACTUAL_N1_BODY',[
      ('old_output','OLD_BxSPLUS1_BANK','GM'),('draft_ids','DRAFT_IDS_N_EQUALS_B','GM'),('target_argmax','EXACT_TARGET_IDS_N_EQUALS_B','GM'),('uniform_rates','EXISTING_U_AND_RATE0','GM'),
      ('accept','NORMAL_EQUAL_OR_SYNTHETIC_U_RATE_VALID','VECTOR_COMPARE'),('col0','TARGET_OR_ACCEPTED_DRAFT','VECTOR_SELECT'),
      ('bonus','BONUS_IDS','GM'),('col1','WHERE_ACCEPT_BONUS_OLD_COL1','VECTOR_SELECT'),('output','UPDATED_COLS0_1_RETAIN_SUFFIX','GM')],
      [('draft_ids','accept','draftID/validity','RAW','body reached'),('target_argmax','accept','targetID','RAW','not synthetic'),('uniform_rates','accept','U<rate0','RAW','synthetic'),
       ('draft_ids','col0','accepteddraft','RAW','synthetic'),('target_argmax','col0','targetwhenrequired','RAW','body reached'),('accept','col0','syntheticselection','RAW','synthetic'),
       ('old_output','col1','oldcol1','STATE_EPOCH','body reached'),('bonus','col1','bonus','RAW','body reached'),('accept','col1','acceptmask','RAW','body reached'),
       ('col0','output','allcol0writes','STATE_EPOCH','body reached'),('col1','output','allcol1writes','STATE_EPOCH','body reached'),
       ('old_output','output','cols2..Sretained','STATE_EPOCH','S>1')])
    graph=dict(id='SAMPLER_GREEDY_SOURCE_G6_V90',templates=templates,source_contract='greedy-fallback-source-contract-v90.json',
      scope='Two directgreedy fallback bodies and immediate producer/output/RNG boundaries. Random/recovery/processor bodies remain external or priorv49contracts.',
      nodes_are_not_unique_native_operators=True,complete=False,overall_complete=False)
    write(O/'operator-graph-G6-v90.json',graph)
    nodes=[];edges=[];counts=[]
    for t in templates:
        ids={n['id']:n for n in t['nodes']};sets=connected_sets(ids,t['edges']);nc=defaultdict(Counter);ec=defaultdict(Counter)
        for k,subs in sets.items():
            for sub in subs:
                for n in sub:nc[n][k]+=1
                for i,e in enumerate(t['edges']):
                    if e['producer']in sub and e['consumer']in sub:ec[i][k]+=1
        for n in t['nodes']:nodes.append(dict(graph='G6',template=t['id'],node_id=n['id'],kind=n['kind'],module='M17',
          connected_set_counts=dict(nc[n['id']]),candidate_source_node_associations=['F35'],
          structural_scope='ALL_CONNECTED_SETS_OF2_TO4_WITHIN_THIS_DECLARED_GRAPH',semantic_coverage='NOT_ESTABLISHED_BY_ENUMERATION',native_lowering_and_all_runtime_variants_closed=False))
        for i,e in enumerate(t['edges']):edges.append(dict(graph='G6',template=t['id'],edge_index=i,edge=e,connected_set_counts=dict(ec[i]),
          producer_candidate_associations=['F35'],consumer_candidate_associations=['F35'],semantic_coverage='NOT_ESTABLISHED_BY_ENUMERATION',dependencies_are_not_all_same_invocation_DAG_edges=True))
        counts.append(dict(template=t['id'],nodes=len(ids),edges=len(t['edges']),connected_sets={str(k):len(v)for k,v in sets.items()}))
    oldnodes=deepcopy(read(O/'operator-graph-node-coverage-v88.json'));oldedges=deepcopy(read(O/'operator-graph-edge-coverage-v88.json'))
    oldnodes['rows']+=nodes;oldnodes.update(row_count=len(oldnodes['rows']),parent='operator-graph-node-coverage-v88.json',overall_complete=False)
    oldedges['rows']+=edges;oldedges.update(row_count=len(oldedges['rows']),parent='operator-graph-edge-coverage-v88.json',overall_complete=False)
    write(O/'operator-graph-node-coverage-v90.json',oldnodes);write(O/'operator-graph-edge-coverage-v90.json',oldedges)
    bindings=deepcopy(read(O/'operator-graph-candidate-bindings-v88.json'))
    f=next(r for r in bindings['candidates']if r['id']=='F35');f['G6_direct_greedy_fallback_supplement']=dict(templates=[t['id']for t in templates],node_associations=len(nodes),
      source_body_review_first_version=90,all_F35_primitives_or_variants_mapped=False,native_occurrence_attested=False)
    bindings.update(parent='operator-graph-candidate-bindings-v88.json',new_pair_hints_added=0,overall_complete=False)
    write(O/'operator-graph-candidate-bindings-v90.json',bindings)
    catalog={k:O/n for k,n in [('G1','operator-graph-G1-v88.json'),('G5','operator-graph-G5-v88.json'),('G6','operator-graph-G6-v90.json')]}
    write(O/'operator-graph-coverage-audit-v90.json',dict(parent='operator-graph-coverage-audit-v88.json',graph_catalog={k:str(p)for k,p in catalog.items()},
      graph_sha256={k:sha(p)for k,p in catalog.items()},new_G6_templates=counts,
      node_rows=oldnodes['row_count'],edge_rows=oldedges['row_count'],new_G6_node_rows=len(nodes),new_G6_edge_rows=len(edges),
      finite_source_structural_rows_are_not_unique_operators_or_semantic_completion=True,
      F35_full_variant_mapping_complete=False,new_native_or_profile_regime_promotions=0,whole_modules_closed=0,overall_complete=False))
    scan=deepcopy(read(R/'fusion-scan.json'));write(O/'fusion-scan-before-v90.json',scan)
    scan['per_node_edge_coverage_ledger']=dict(summary=str(O/'operator-graph-coverage-audit-v90.json'),nodes=str(O/'operator-graph-node-coverage-v90.json'),
      edges=str(O/'operator-graph-edge-coverage-v90.json'),graph_catalog={k:str(p)for k,p in catalog.items()},structural_only=True,all_native_or_semantic_variants_complete=False)
    scan['operator_coverage_v90_scope']='G6supplements2directgreedyfallbackbodies,includingactualcount1vsS-capacity distinction. Random/recovery and other F35 primitive/branch mappings remainpartial.'
    write(O/'fusion-scan-proposed-v90.json',scan)
    print(dict(new_body_functions=2,local_ifs=6,local_arms=12,cost_models=len(cost),G6_nodes=len(nodes),G6_edges=len(edges),
      current_node_rows=oldnodes['row_count'],current_edge_rows=oldedges['row_count'],new_native_work=False,overall_complete=False))


if __name__=='__main__':main()
