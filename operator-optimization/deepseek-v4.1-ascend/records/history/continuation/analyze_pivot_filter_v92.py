"""Named monolithic filter operator,ordered fuzzy statistics and source-work envelopes."""
from pathlib import Path
from copy import deepcopy
from collections import Counter,defaultdict
import ast,hashlib,math
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from reconcile_operator_graph_coverage_v88 import connected_sets


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not(O/'REVIEW-CHECKPOINT-v92.json').exists()
    root=W/'frozen-source-snapshot';p=root/'vllm/vllm/v1/sample/ops/topk_topp_triton.py';manifest=root/'SOURCE-MANIFEST.json'
    entry=next(r for r in read(manifest)['files']if r['path']==str(p.relative_to(root)));assert sha(p)==entry['sha256']
    src=p.read_text();lines=src.splitlines();tree=ast.parse(src);wanted=['_update_min_larger_stats','_topk_topp_kernel','_topk_topp','apply_top_k_top_p_triton'];parts=[];guards=[];loops=[]
    table_lengths={x.targets[0].id:len(ast.literal_eval(x.value))for x in tree.body if isinstance(x,ast.Assign)and isinstance(x.targets[0],ast.Name)and x.targets[0].id in ['_NORMAL_CDF_TO_SIGMA_TABLE','_PERCENTILE_TO_STD_TABLE']}
    assert sorted(table_lengths.values())==[200,200]
    for f in tree.body:
        if not isinstance(f,ast.FunctionDef)or f.name not in wanted:continue
        parts.append(dict(name=f.name,first_line=f.lineno,last_line=f.end_lineno,source=lines[f.lineno-1:f.end_lineno]))
        for x in ast.walk(f):
            if isinstance(x,ast.If):guards.append(dict(function=f.name,line=x.lineno,condition=ast.unparse(x.test),arms=['BODY','ORELSE_OR_FALLTHROUGH']))
            if isinstance(x,(ast.For,ast.While)):
                cond=ast.unparse(x.test)if isinstance(x,ast.While)else ast.unparse(x.iter)
                text='\n'.join(lines[x.lineno-1:x.end_lineno]);loops.append(dict(function=f.name,line=x.lineno,last_line=x.end_lineno,type=type(x).__name__,condition=cond,has18_iteration_stop=('num_iters >= 18'in text)))
    assert len(parts)==4 and len(guards)==51 and sum(r['type']=='While'for r in loops)==5 and sum(r['type']=='For'for r in loops)==15
    assert all(r['has18_iteration_stop']for r in loops if r['type']=='While')
    platform=W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/platform.py';pl=platform.read_text().splitlines()
    platform_root=W/'sampler-source-supplement/frozen-source-snapshot';platform_manifest=platform_root/'SOURCE-MANIFEST.json'
    pp=next(r for r in read(platform_manifest)['files']if r['path']==str(platform.relative_to(platform_root)));assert sha(platform)==pp['sha256']
    route=dict(path=str(platform),sha256=sha(platform),manifest=str(platform_manifest),manifest_sha256=sha(platform_manifest),source_manifest_hash_match=True,parts=[dict(first_line=a,last_line=b,source=pl[a-1:b])for a,b in [(89,94),(1311,1312)]],
      finding='PinnedNPUPlatform declaresOOT/device_type=npu,andcommentsis_cuda_alikeFalse. Wrapperuse_splitrequiresdevice.type==cuda;theA3modelheredeclaresSPLIT_COVERS_PONLY=False.',
      current_loaded_platform_or_compiled_kernel_attested=False)
    write(O/'pivot-filter-source-contract-v92.json',dict(path=str(p),sha256=sha(p),manifest=str(manifest),manifest_sha256=sha(manifest),source_manifest_hash_match=True,
      functions=parts,source_if_cases=guards,source_if_sites=51,source_if_arms=102,loops=loops,while_loops=5,for_loops=15,NPU_route=route,
      facts=[
        'NPU-declaredmonolithicrouteuses8192elementfulltiles,4096truncatedtiles,8warpsparameter,programs=min(num_sm,B),oneworkspaceV-rowperpid. GPUwarpsarenotAscendAIVcount. CUDA-onlysplithelpers arenot expanded asA3execution.',
        'Alreadyfused:firstfullscanmax/min/finitecount/outliercompaction;twoKpivots shareonecandidate scan;eachpivotmass/min/duplicatecount sharesonepass;finalmaskiswithin samekernel. Onekernelisnotonevocabpass.',
        'KsearchchoosesoutlierBUFFERifO>k,elsewholeLOGITS. Fivewhileloops have18iteration/toleranceexit. Fallbackdoesnot guaranteeanexactsortedKthreshold or tiecardinality;invalid/>=maxfinalpivot suppressesmasking.',
        'DuplicatehelperusesexacttileMin<runningMin but approximateabsdifference<1e-9 forcounts. It isnot astandard commutative/associativemin-countmonoid;originaltileboundaries andorderedupdates matter.',
        'PostKtopPnormalizerselectsK/duplicatequota;bufferpaththirdpassExpusesmaskedselectedlogits,fourthpassExpusesoriginalbufferlogits. Theyarenot the sameExpinput. RegatherpathstoresselectedExpthenonlydividesnextpass.',
        'StandaloneP usesfirstsamplemean+10std asnormalization shift,notglobalmax. It computesglobalmaxduringfullscanbutExpusesheuristicmax_sample. SecondfullpassrepeatsExpofsameoriginalL/max_sample;fallbackrepopulatesfullP andaddsathirdExp/Divpass.',
        'FinalguardsetsnopforNaN orpivot>=max;otherwiseonlyactivepivot!=-inf invokesfinalwrite. Duplicatequota usesoriginalvocabordertilecumsums,includingnear-groupvaluesnoteligiblebystrictpivot.',
        'Mutableworkspacecachekeyisdevice/dtype/V,notstream/call;reuseisallocationreuse,notactivationreuse. Constantsread-onlyafterready;overlappingcallscannotsafelysharemutablepidrowwithoutownership/eventsornewbank.',
        'EachBUFFERreadrange isboundedbycurrentO/O2/V producedprefix. Anewisolatedscratchbankneednotcopyoldcontentsifthedeclaredwrite-before-readprefixandrowownershiphold. Fullallocatedextentisnottraffic.'
      ],
      scope='Four namedmonolithic operatorfunctions plusminimalNPUroutingclauses;notGPU-splitbody/compiler/backendexecutionproof.',
      actual_random_batch_invariant_or_native_compilation_attested=False,new_target_source_files=0,native_test=False,overall_complete=False))
    source_branches=[
      ('K_NOOP','Kabsentork>=V;Pabsentorp>=1','Parameter/cache/launchmayremain,butnoLOGITSsweepinmonolith.'),
      ('K_BUFFER','k<V,O>k','KsearchreadsOcompactedlogitsperiteration.'),('K_FULL','k<V,O<=k','KsearchreadsVLOGITSperiteration.'),
      ('K_FINITE_LE_K','finite_count<=k','Kfinalpivot-inf;standalonePmaystillberequired.'),
      ('POST_K_P','TOPP,finite_count>k,p<1','KselectednormalizerthenPmasssearch;mask/quotaexpressionsnotgenericsoftmaxCSE.'),
      ('STANDALONE_P_BUFFER','finalpivot-inf,run_standalone,p<1,mass_outliers>p','TwofullExp/Divpasses,searchO.'),
      ('STANDALONE_P_FULL','samebutmass_outliers<=p','AdditionalfullPbufferrebuildthenVsearch.'),
      ('FINAL_GUARD_QUOTA','activepivot<max andpivot!=-inf','Strictpivotandorderedfuzzyduplicatequota;otherwiseunchangedinputbank.')]
    write(O/'pivot-filter-branch-inventory-v92.json',dict(rows=[dict(id=i,condition=c,contract=t,state='PINNED_LOCAL_OPERATOR_BRANCH;NATIVE_OCCURRENCE_UNKNOWN')for i,c,t in source_branches],
      source_guard_navigation=[51,102],all_sampler_or_compiler_branches_closed=False,overall_complete=False))
    costs=[]
    V=129280;S=min(V,8192);T=math.ceil(V/8192)
    for B in [2,4]:
        for k in [1,64,1024]:
            for Osize in [0,64,1024,V]:
                for I in [1,18]:
                    Rsize=Osize if Osize>k else V
                    costs.append(dict(case='K_ONLY',B=B,V=V,k=k,O_scenario=Osize,iterations_scenario=I,search_source='BUFFER'if Osize>k else'LOGITS',
                      search_values_per_iteration=Rsize,sample_load_reference_bytes=4*B*S,initial_full_load_reference_bytes=4*B*V,
                      outlier_store_reference_bytes=4*B*Osize,search_load_reference_bytes=4*B*I*Rsize,
                      active_final_mask_readwrite_reference_bytes=8*B*V,
                      total_requested_reference_bytes_if_final_mask_active=4*B*(S+V+Osize+I*Rsize+2*V),
                      native_parameters_iterations_or_active_final_guard_attested=False))
        for Osize in [0,1024,V]:
            for fallback in [False,True]:
                if Osize==0 and not fallback:continue
                for I in [1,18]:
                    Rsize=V if fallback else Osize;exp=(3 if fallback else 2)*B*V;div=(2 if fallback else 1)*B*V
                    costs.append(dict(case='STANDALONE_P',B=B,V=V,O_scenario=Osize,full_buffer_fallback=fallback,iterations_scenario=I,
                      search_values_per_iteration=Rsize,
                      total_requested_reference_bytes_if_final_mask_active=4*B*(S+2*V+Osize+(2*V if fallback else 0)+I*Rsize+2*V),
                      valid_value_Exp_reference_elements=exp,valid_value_Div_reference_elements=div,
                      Exp_capacity_cycle_reference_48AIV=exp/(48*32),
                      Exp_capacity_cycle_reference_if_one_program_per_AIV_and_B_programs=exp/(B*32),
                      one_program_per_AIV_mapping_attested=False,
                      cache_Exp_additional_store_reference_bytes=4*B*V,cache_Exp_removed_reference_elements=B*V,
                      source_sample_tile_reuse_requested_read_credit_bytes=4*B*S,
                      source_load_CSE_by_actual_compiler_attested=False,source_counted_exp_isnot_allalgorithm_lowerbound=True))
    workspaces=[]
    for B in [2,4]:workspaces.append(dict(B=B,V=V,assuming_num_sm_ge_B=True,source_workspace_row_bytes=4*V,logical_live_rows=B,
      fresh_power2_B_allocation_bytes=4*B*V,second_isolated_bank_additional_bytes=4*B*V,
      logical_programs_per_row=1,logical_row_programs=B,
      fixed_original_8192_tiles_per_row=T,new_tile_scan_programs=B*T,
      new_K_two_pivot_partial_bytes_per_row_per_iteration=24*T,new_P_single_pivot_partial_bytes_per_row_per_iteration=12*T,
      metadata_write_and_read_required=True,
      prefix_bank_uint32_bytes_per_row=4*(T+1),
      prefix_metadata_virtualmask_credit_reference_bytes_per_row_for_h2_consumers=4*V-4*(T+1)*3,
      descriptor_only_virtualmask_removed_final_readwrite_reference_bytes_per_row=8*V,
      new_lifetime_launch_precision_or_bandwidth_cost_priced_as_zero=False))
    write(O/'pivot-filter-work-and-resource-models-v92.json',dict(models=costs,model_count=len(costs),workspace_models=workspaces,
      V=V,BLOCK_SIZE=8192,BLOCK_SIZE_TRUNC=4096,NUM_TILES=T,valid_values=V,padded_tile_lanes=T*8192,
      UB_per_AIV=196608,whole_workspace_row_fits_single_AIV_UB=False,
      readonly_lookup_table_lengths=table_lengths,readonly_lookup_tables_reference_bytes=sum(table_lengths.values())*4,
      fixed_SOURCE_search_iteration_cap=18,source_P_shift_finite_sufficient_envelope_maxdiff=math.log((2-2**-23)*2**127)-math.log(V),
      scenario_joint_native_reachability_unproved=True,valid_metadata_domain='V>0,1<=k<=V,0<=p<=1;O0cannotenterstrictmass_outliers>pbranch.',
      envelope_scope='Real exponential/positive sum sufficientbound;rounding margin/nativeExp supportstillrequired,not actualinputbound.',
      traffic_scope='Requested logical FLOATload/storevalues ofdeclaredsourcephases,not HBMtraffic orelapsedcycles. ConditionalSort32boundv91doesnotapplytothispivotkernel.',
      resource_scope='Exp32elements/AIVcycle publicA3case. Source logicalprogramcountisnotnativeAIVmapping;48AIVandone-program/AIVcasesseparate. No clockscaling/MFU.',
      active_final_guard_or_runnable_A3_Triton_kernel_proved=False,overall_complete=False))
    prefix=[('K_BUFFER','[0,O)','[0,O)'),('K_FULL','[0,O)','none:searchreadsLOGITS'),('KP_BUFFER','[0,O)','[0,O)'),('KP_REGATHER','[0,O2)','[0,O2)'),('P_BUFFER','[0,O)','[0,O)'),('P_FULL_REBUILD','[0,V)','[0,V)')]
    write(O/'pivot-filter-workspace-prefix-and-effects-v92.json',dict(rows=[dict(branch=a,produced_BUFFER_prefix=b,subsequent_BUFFER_read_range=c)for a,b,c in prefix],
      source_case_current_prefix_written_before_read=True,old_workspace_contents_are_not_reusable_activation_evidence=True,
      cache_key='(logits.device,logits.dtype,vocab_size)',missing_from_cache_key=['stream','call_generation','row_plan_epoch'],
      overlapping_calls_require='Exclusiveworkspaceownership/events ordistinctbanks;newbankscanbeuninitializedundertheprefixconditions,readonlyconstanttablesreusableonlyafterready.',
      standalone_Exp_cache='SourceE=exp(originalL-max_sample) isidenticalinfirstandsecondstandalonepasses. FirstscanmaystoreEfullrow;secondscanreadsE/normalizerandseriallycompactsPprefixinplace. Adds4BVstore,removesBVExp. FallbackmuststillrebuildfullPfromLafterprefixcompaction.',
      Exp_cache_not_generic_postK_CSE='PostKbufferthirdpassExpusesselectedmaskedL;fourthpassExpusesalloriginalbufferL. Do notreplacebothwithmaskedExpwithoutnewproof.',
      actual_concurrency_or_scratch_race_observed=False,overall_complete=False))

    # Expand the previously opaque upstream boundary inside G7,retainingdefaultfiltergraph.
    g=deepcopy(read(O/'operator-graph-G7-v91.json'));boundary=next(n for n in g['templates'][0]['nodes']if n['id']=='upstream_filter_boundary')
    boundary['source_resolution_v92']=dict(template='UPSTREAM_MONOLITHIC_PIVOT_FILTER',condition='DeclaredNPUroutewithSPLIT_COVERS_PONLYFalse;nativecompilation/entryflagsunbound')
    nodespec=[('L','UNCHANGED_INPUT_LOGITS_EPOCH'),('params','K_P_ROUTE_AND_ROW_PARAMETERS'),('sample','FIRST_TILE_STATISTICS'),('tables','READONLY_LOOKUP_TABLES_READY_EPOCH'),
      ('fullscan','FULL_STATS_AND_OUTLIER_PREFIX'),('Ksearch','TWO_PIVOT_COUNT_MIN_FUZZY_QUOTA_SCAN'),('Kquota','K_PIVOT_AND_DUPLICATE_STATE'),
      ('KPnormalizer','SELECTED_K_NORMALIZER_AND_PROB_BANK'),('Pnormalizer','STANDALONE_HEURISTIC_SHIFT_NORMALIZER'),('Pbank','COMPACT_OR_FULL_CURRENT_PROB_PREFIX'),
      ('Psearch','PIVOT_MASS_MIN_FUZZY_QUOTA_SCAN'),('descriptor','FINAL_PIVOT_QUOTA_ACTIVE_GUARD'),('finalmask','ORDERED_VOCAB_MASK_STORE'),('output','UPDATED_LOGITS_OR_ORIGINAL_NOP'),
      ('workspace_owner','MUTABLE_SCRATCH_BANK_GENERATION'),('publication','WORKSPACE_LAST_READ_AND_RELEASE')]
    es=[('L','sample','row sample'),('params','sample','activebranch'),('sample','fullscan','Kheuristic'),('tables','fullscan','percentiles'),('L','fullscan','fullrow'),
      ('fullscan','Ksearch','OorVrange'),('workspace_owner','fullscan','exclusivebank'),('Ksearch','Kquota','orderedhelperstate'),('params','Kquota','k'),
      ('Kquota','KPnormalizer','selectedK/quota'),('L','KPnormalizer','maskedExpinputs'),('fullscan','KPnormalizer','originaloutlierbuffer'),
      ('sample','Pnormalizer','heuristicshift'),('tables','Pnormalizer','pheuristic'),('L','Pnormalizer','fullrows'),
      ('KPnormalizer','Pbank','postKprefix'),('Pnormalizer','Pbank','outlierorfullprefix'),('Pbank','Psearch','psearchrange'),('params','Psearch','p'),
      ('Kquota','descriptor','Konly'),('Psearch','descriptor','Pcase'),('descriptor','finalmask','activepivot/quota'),('L','finalmask','originalrows'),
      ('finalmask','output','maskedwrites'),('L','output','guardednopunchanged'),('workspace_owner','Pbank','currentgeneration'),('Ksearch','publication','Klastread'),('Psearch','publication','Plastread')]
    template=dict(id='UPSTREAM_MONOLITHIC_PIVOT_FILTER',nodes=[dict(id=i,kind=k,resource='DECLARED_OPERATOR_OR_STATE')for i,k in nodespec],
      edges=[dict(producer=a,consumer=b,value=v,dependency_kind='CONDITIONAL_SOURCE_VALUE_STATE_OR_CONTROL',condition='Seev92branchinventory;notallpathscoexecute',source='pivot-filter-source-contract-v92.json')for a,b,v in es])
    g['templates'].append(template);g.update(id='SAMPLER_FILTER_RACE_G7_V92',parent='operator-graph-G7-v91.json',complete=False,overall_complete=False);write(O/'operator-graph-G7-v92.json',g)
    ids={n['id']:n for n in template['nodes']};sets=connected_sets(ids,template['edges']);nc=defaultdict(Counter);ec=defaultdict(Counter)
    for k,subs in sets.items():
        for sub in subs:
            for i in sub:nc[i][k]+=1
            for ei,e in enumerate(template['edges']):
                if e['producer']in sub and e['consumer']in sub:ec[ei][k]+=1
    newnodes=[dict(graph='G7',template=template['id'],node_id=n['id'],kind=n['kind'],module='M17',connected_set_counts=dict(nc[n['id']]),candidate_source_node_associations=['F35'],
      structural_scope='ALL_CONNECTED_SETS_OF2_TO4_WITHIN_THIS_DECLARED_GRAPH',semantic_coverage='NOT_ESTABLISHED_BY_ENUMERATION',native_lowering_and_all_runtime_variants_closed=False)for n in template['nodes']]
    newedges=[dict(graph='G7',template=template['id'],edge_index=i,edge=e,connected_set_counts=dict(ec[i]),producer_candidate_associations=['F35'],consumer_candidate_associations=['F35'],semantic_coverage='NOT_ESTABLISHED_BY_ENUMERATION',dependencies_are_not_all_same_invocation_DAG_edges=True)for i,e in enumerate(template['edges'])]
    for name,extra in [('node',newnodes),('edge',newedges)]:
        d=deepcopy(read(O/f'operator-graph-{name}-coverage-v91.json'));d['rows']+=extra;d.update(row_count=len(d['rows']),parent=f'operator-graph-{name}-coverage-v91.json',overall_complete=False);write(O/f'operator-graph-{name}-coverage-v92.json',d)
    binding=deepcopy(read(O/'operator-graph-candidate-bindings-v91.json'));next(r for r in binding['candidates']if r['id']=='F35')['G7_monolithic_source_supplement_v92']=dict(template=template['id'],nodes=16,edges=28,all_F35_primitives_or_variants_mapped=False)
    binding.update(parent='operator-graph-candidate-bindings-v91.json',overall_complete=False);write(O/'operator-graph-candidate-bindings-v92.json',binding)
    old=read(O/'operator-graph-coverage-audit-v91.json');catalog=old['graph_catalog']|{'G7':str(O/'operator-graph-G7-v92.json')}
    write(O/'operator-graph-coverage-audit-v92.json',dict(parent='operator-graph-coverage-audit-v91.json',graph_catalog=catalog,graph_sha256={k:sha(Path(p))for k,p in catalog.items()},
      new_G7_node_rows=16,new_G7_edge_rows=28,node_rows=724,edge_rows=1187,finite_source_structural_rows_are_not_unique_operators_or_semantic_completion=True,F35_full_variant_mapping_complete=False,whole_modules_closed=0,overall_complete=False))
    scan=deepcopy(read(R/'fusion-scan.json'));write(O/'fusion-scan-before-v92.json',scan)
    scan['per_node_edge_coverage_ledger']=dict(summary=str(O/'operator-graph-coverage-audit-v92.json'),nodes=str(O/'operator-graph-node-coverage-v92.json'),edges=str(O/'operator-graph-edge-coverage-v92.json'),graph_catalog=catalog,structural_only=True,all_native_or_semantic_variants_complete=False)
    scan['operator_coverage_v92_scope']='Namedupstreammonolithicpivotoperatorandorderedfuzzy/descriptor/workspacecontracts;notGPU-split/nativeA3executionorallF35closure.';write(O/'fusion-scan-proposed-v92.json',scan)
    print(dict(named_kernel_functions=4,if_sites=51,if_arms=102,while_loops=5,for_loops=15,work_models=len(costs),new_nodes=16,new_edges=28,current_nodes=724,current_edges=1187,overall_complete=False))


if __name__=='__main__':main()
