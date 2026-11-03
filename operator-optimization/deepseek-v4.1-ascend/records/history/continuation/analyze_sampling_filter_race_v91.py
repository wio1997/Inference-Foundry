"""Operator-only probability-cutoff,exponential-race and RNG-effect contracts."""
from pathlib import Path
from copy import deepcopy
from collections import Counter,defaultdict
import ast,hashlib,math
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from reconcile_operator_graph_coverage_v88 import connected_sets


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not(O/'REVIEW-CHECKPOINT-v91.json').exists()
    roots=[W/'sampler-source-supplement/frozen-source-snapshot',W/'frozen-source-snapshot']
    specifications=[(roots[0]/'vllm-ascend/vllm_ascend/sample/sampler.py',
      ['random_sample','AscendTopKTopPSampler.forward_native','_apply_top_k_top_p_pytorch','_apply_top_k_top_p_torch_npu']),
      (roots[1]/'vllm/vllm/v1/sample/ops/topk_topp_sampler.py',
      ['TopKTopPSampler.forward_native','apply_top_k_top_p','apply_top_k_top_p_pytorch','empty_exponential_noise_like','sample_with_exponential_noise','random_sample']),
      (roots[1]/'vllm/vllm/v1/sample/sampler.py',['Sampler.sample'])]
    files=[];guards=[]
    for path,wanted in specifications:
        root=next(root for root in roots if path.is_relative_to(root));manifest=root/'SOURCE-MANIFEST.json'
        receipt=next(r for r in read(manifest)['files']if r['path']==str(path.relative_to(root)));assert sha(path)==receipt['sha256']
        source=path.read_text();lines=source.splitlines();tree=ast.parse(source);functions=[]
        for n in tree.body:
            if isinstance(n,ast.FunctionDef):functions.append((n.name,n))
            elif isinstance(n,ast.ClassDef):functions +=[(n.name+'.'+f.name,f)for f in n.body if isinstance(f,ast.FunctionDef)]
        parts=[]
        for name,f in functions:
            if name not in wanted:continue
            parts.append(dict(name=name,first_line=f.lineno,last_line=f.end_lineno,source=lines[f.lineno-1:f.end_lineno]))
            for q in ast.walk(f):
                if isinstance(q,ast.If):guards.append(dict(path=str(path),function=name,line=q.lineno,test=ast.unparse(q.test),reviewed_arms=['BODY','ORELSE_OR_FALLTHROUGH']))
        assert {x['name']for x in parts}==set(wanted)
        files.append(dict(path=str(path),sha256=sha(path),manifest=str(manifest),manifest_sha256=sha(manifest),source_manifest_hash_match=True,parts=parts))
    write(O/'sampling-filter-race-source-contract-v91.json',dict(source_files=files,named_functions_pinned=sum(len(f['parts'])for f in files),local_if_sites=len(guards),local_if_arms=2*len(guards),local_guard_navigation=guards,
      scope='Default Ascend filtering/race and direct upstream comparison/delegation/mixed-row boundaries. No CUDA/CPU framework survey;Triton filter body remains a separately open lowering.',
      facts=[
        'DefaultAscendfilter takesoneinitialsoftmax,sortsitsprobabilitiesascending once,andcomputesbothk/p cutoffs fromthat unchangedprobability bank. Thefirstk maskmutateslogits,notprobs_sort/probs.',
        'TopKcutoffisprobability orderstatistic atV-k;strictP<cutoffkeepsallboundaryties. k==Vsets cutoff-inf. TopPusesoriginalCDF<=1-p,forceslastsortedmaskfalse,thencomparesunsortedP<sortedP[count];alltiesatcutoffretained.',
        'UpstreamPyTorchfilter sortslogits,appliestopKtolocal sortedlogitsfirst,thencomputessoftmax/CDF onthetruncatedbank andmaskssortedpositionsbefore scatter. This differs inreal arithmetic andtiecardinality;notaninterchangeablefilter.',
        'Ascend hardware-capability choice selectsPyTorchdirector _torch_npu wrapper,whichalsoonlycalls thesamePyTorchfilter inthissnapshot. Itdoesnotprovea fusednpu_top_k_top_p invocation. BATCH_INVARIANTdelegatesupstream;HAS_TRITONthenchoosesitsdistinctfilterentry;body/nativemode notclosedhere.',
        'DefaultAscendforward retainsprocessed_logits asfilteredlogitsalias,orcomputesprocessed_logprobs before aseparatesoftmax. Takinglogofroundedsoftmax isnot agenerallog_softmax replacement.',
        'Ascendrandom_sample drawsQon global_stream,waitsfromcurrentstream,recordsQoncurrentstream,andreturnsargmax ofprobs.div_(Q). Pismutated;Qzero/Inf/NaNisnotreplacedwithfallback-recovery sentinels.',
        'DefaultQdraw isomitted exactlywhenlen(generators)==B;otherwise itdrawsallB*V then overwrites seededrows in dictionaryorder. Allseededbehavior differsfromthev33rejection-recoverywrapper thatalwaysdrawsdefaultQ. Preservegeneratoridentity/alias andeffects,notjustseednumbers.',
        'Sampler.sampleallgreedyreturnsbeforethisrandompipeline. Mixedsampling runsrandompipelineforallB then selectswith temperature<1e-5;thatpredicate cannot be assumedidentical to rejectionhelpertemperature==0 withoutanupstreammetadata-domainproof.',
        'PrivatePfromforwardsoftmaxhasnolocalreturnedobserverbesidesrandom_sample;processedscoreoutputsareseparatefilteredlogits/log_softmaxbanks. Genericdirectrandom_sample(P) stillhasanobservableinplacePcontract;privatecallerorfusedwritebackmustbedeclared.'
      ],
      source_math_not_native_backend_attestation=True,current_probability_execution_or_flags_proved=False,
      full_sampler_branch_universe_or_readers_closed=False,new_target_source_files=0,implementation=False,native_test=False,overall_complete=False))
    write(O/'sampling-filter-race-algebra-v91.json',dict(
      mask_fusion='GivenidenticaloriginalP,cutoffK,cutoffP andsameoldlogits,twomasked_fill(-inf) writes equalonewriteunder (P<cutoffK)OR(P<cutoffP). Keep originalgathers/validationandnointermediateobserver. UnderorderednonNaNcutoffs,thisisP<max(cutoffK,cutoffP);NaNcasesmustretainORorprovedNaN-awarehandling.',
      CDF_search='ForanexistingnondecreasingnonNaNCDF andsameactualthresholdt=1-p,source count=sum(CDF<=t) withlastmaskforcedfalse equalsmin(upper_bound(CDF,t),V-1). ReusesexactexistingCDFvalues;doesnotchangeFP32scanorder. Binarysearchworstcase<=ceil(log2(V+1))probes;fusing/replacingtheCDFproducerneedsitsownnative-roundingproof.',
      top_k_only='Anexactorder-statisticselection ontheSAMEroundedprobabilities canreplacefullsortwhentopPabsent,retainingallties. Logitselection/exactkoutputs areseparateprecision/tie contracts. No-topKallkV,pNone canbe aconditionalno-opguardifmetadataanderrors preserved.',
      race='GivenidenticalP/Qanddivisionresults,streamingDiv→argmaxcan avoidratioGMwrite/read. GenerichelpermustretainmodifiedP unlessprivate;withwriteback keepPstore butavoidargmaxreread. PreserveallQ/RNG/stream/lifetime effects andsame comparator.',
      mixed_rows='WithinSampler.samplemixedpath,the randomIDisunusedonrows selectedgreedybytemperature<1e-5. AfterALLoriginalQdraws,privatePDiv/argmax mayonlycomputeactive randomrows;otherlogit/score/processorworkremains. Qshape/statecannotbecompactedwithoutanew RNGprotocol.',
      precision_fences=['Initialfiltersoftmax/sort alreadyshared acrossk/p;do notcountthat asnewfusion.',
        'Do notrenameAscendprobabilitycutofffilterasanupstreamsequentialtopK/topPfilter.',
        'Do notreplacefinalsoftmaxwithzeroedoriginalP solelybecausea positivecommonfactor cancelsinreal arithmetic;FP32probability/division rounding canchange winner.',
        'Do notcomputeprocessed_logprobs aslog(roundedP) underunderflow.',
        'p==1 isnot automaticallyno-opifinitialPcontainszero afterrounding;sourceprobabilitycutoff canmaskfiniteunderflowedlogits.',
        'No highlevelsort/CDF/softmax/argmaxcapacity isassignedfroma genericVectorAdd/Maxrate.'],
      theoretical_domains='FullSort32/MrgSortcase only:atleastoneSort32pass overallroundedvalues;selection/otheralgorithms outside. Exactunstructuredrowargmax needsallrelevantvalues withoutcertificate;thisisnotHBMbytes orphysicalMFU.',
      new_physical_speedup_or_native_precision_proved=False,overall_complete=False))
    costs=[]
    for B in [2,4]:
        V=129280;n=B*V
        for k_on,p_on in [(False,False),(True,False),(False,True),(True,True)]:
            active_filters=int(k_on)+int(p_on)
            for A in sorted(set([1,max(1,B//2),B])):
                mask_old=8*n*active_filters;mask_new=8*n*bool(active_filters)
                costs.append(dict(B=B,V=V,active_random_rows_scenario=A,topK_present=k_on,topP_present=p_on,
                  probability_execution_or_parameter_flags_attested=False,
                  original_mask_logit_readwrite_reference_bytes=mask_old,one_mask_logit_readwrite_reference_bytes=mask_new,
                  mask_logit_plane_reference_credit_bytes=mask_old-mask_new,
                  original_allrow_div_argmax_value_reference_bytes=16*n,
                  fused_privateP_active_row_value_reference_bytes=8*A*V,
                  fused_with_modifiedP_writeback_allrows_value_reference_bytes=12*n,
                  mask_and_private_race_disjoint_span_old_reference_bytes=mask_old+16*n,
                  mask_and_private_race_disjoint_span_new_reference_bytes=mask_new+8*A*V,
                  baseline_TWO_cutoff_FLOAT_Compare_elements=(2*n if active_filters==2 else n*bool(active_filters)),
                  combined_ordered_cutoff_FLOAT_Compare_elements=n*bool(active_filters),
                  additional_ordered_cutoff_Max_elements=B if active_filters==2 else 0,
                  cutoff_Max_and_Compare_joint_issue_capacity_proved=False,
                  ordered_cutoff_Compare_cycle_reference=(n*bool(active_filters))/(48*64),
                  active_race_Div_cycle_reference=A*V/(48*32),
                  conditional_full_Sort32_firstpass_elements=n if active_filters else 0,
                  conditional_full_Sort32_firstpass_cycle_reference=n/96 if active_filters else 0,
                  full_CDF_boolean_compare_elements=n if p_on else 0,
                  existing_CDF_binary_search_max_value_probes=B*math.ceil(math.log2(V+1))if p_on else 0,
                  source_single_FP32_plane_bytes=4*n,source_discarded_sort_index_plane_reference_bytes=8*n if active_filters else 0,
                  omitted_costs='Softmax/sort/merges/CDF,gather/parameterpreparation,returnedlogits/logprobs,Qgeneration/stores/streamwaits,newpacking/launch/nativeworkspaces.',
                  reference_planes_not_measured_allocation_or_HBM=True))
    rng=[]
    for B in [2,4]:
        for G in range(B+1):rng.append(dict(B=B,seeded_rows=G,V=129280,default_draw_elements=(0 if G==B else B*129280),seeded_overwrite_draw_elements=G*129280,
          all_original_calls_preserved=True,independent_generator_objects_or_nativecounter_mapping_proved=False))
    write(O/'sampling-filter-race-cost-models-v91.json',dict(models=costs,models_count=len(costs),B_reference_geometry=[2,4],V_reference_geometry=129280,
      rng_draw_cases=rng,source_native_flags_or_new_shapes_attested=False,
      resource_reference='typed-capacity-primary-contract-v89.json',Vector_AIV_reference=48,
      whole_FLOAT_P_or_Q_row_bytes=517120,UB_per_AIV=196608,oneAIVsimultaneous_whole_P_Q_fits=False,
      actual_software_instruction_lowering_or_physical_wallclock_upper_proved=False,overall_complete=False))

    # Primitive source supplement,not a mergedall-backend numerical graph.
    spec=[('input_logits','WORKING_FLOAT_LOGITS_OLD_EPOCH'),('filter_P','ORIGINAL_FP32_SOFTMAX'),('sort_P','ASCENDING_PROBABILITY_SORT_VALUES'),
      ('discarded_sort_idx','UNUSED_SORT_INDEX_OUTPUT'),('K','OPTIONAL_K_PARAMETER'),('Pparam','OPTIONAL_P_PARAMETER'),
      ('cutK','PROBABILITY_ORDER_STATISTIC_AND_KV_NOOP'),('cdf','ORIGINAL_PROBABILITY_CDF'),('cdf_mask','LE_ONE_MINUS_P_LAST_FALSE'),
      ('countP','COUNT_MASKED_SORTED_POSITIONS'),('cutP','SORTED_PROBABILITY_AT_COUNT'),('maskK','ORIGINAL_P_LT_CUTK'),('maskP','ORIGINAL_P_LT_CUTP'),
      ('logits_afterK','MASKED_LOGITS_K_EPOCH'),('filtered_logits','MASKED_LOGITS_P_EPOCH'),('returned_scores','OPTIONAL_FILTERED_LOGITS_OR_LOGSOFTMAX'),
      ('race_P','POST_FILTER_FP32_SOFTMAX'),('rng_old','OLD_GLOBAL_AND_REQUEST_GENERATOR_STATES'),('Qdefault','DEFAULT_BV_EXPONENTIAL_IF_G_NE_B'),
      ('Qseed','SEEDED_ROW_EXPONENTIAL_OVERWRITES_DICT_ORDER'),('rng_new','ADVANCED_GENERATOR_STATES'),('Qwait','CURRENT_WAIT_GLOBAL_AND_RECORD_STREAM'),
      ('ratio_P','INPLACE_P_DIV_Q'),('random_ID','FULL_VOCAB_RATIO_ARGMAX'),('greedy_ID','OTHER_GREEDY_ID_INPUT'),
      ('mixed_mask','TEMPERATURE_LT_1E_MINUS5'),('chosen_ID','MIXED_WHERE_OR_ALLRANDOM_ID'),('batch_invariant','UPSTREAM_DELEGATION_GUARD'),
      ('upstream_filter_boundary','UPSTREAM_TRITON_OR_PYTORCH_FILTER_DIFFERENT_CONTRACT')]
    edges=[('input_logits','filter_P','initialprobabilityvalues','anyfilter'),('filter_P','sort_P','onealreadysharedsort','anyfilter'),
      ('filter_P','discarded_sort_idx','sortindexsideproduct','anyfilter'),('sort_P','cutK','sameoriginalsortedP','kpresent'),('K','cutK','V-k,kVflag','kpresent'),
      ('sort_P','cdf','sameoriginalsortedP','ppresent'),('Pparam','cdf_mask','actual1-p','ppresent'),('cdf','cdf_mask','CDFvalues','ppresent'),
      ('cdf_mask','countP','boolsum','ppresent'),('countP','cutP','count','ppresent'),('sort_P','cutP','probabilitycutoff','ppresent'),
      ('filter_P','maskK','sameoriginalP','kpresent'),('cutK','maskK','cutoff','kpresent'),('filter_P','maskP','sameoriginalP','ppresent'),('cutP','maskP','cutoff','ppresent'),
      ('input_logits','logits_afterK','oldlogits','kpresent'),('maskK','logits_afterK','masked_fill-inf','kpresent'),('logits_afterK','filtered_logits','orderedlogitsbank','kandp'),
      ('input_logits','filtered_logits','oldlogits','notk'),('maskP','filtered_logits','masked_fill-inf','ppresent'),
      ('filtered_logits','returned_scores','samefilteredbankorlogsoftmax','processedmode'),('filtered_logits','race_P','newsoftmax','randompath'),
      ('rng_old','Qdefault','globalstate','G!=B'),('rng_old','Qseed','seededgeneratorstates','G>0'),('Qdefault','Qseed','defaultbankthenoverwrites','0<G<B'),
      ('Qdefault','rng_new','globalstateadvanced','G!=B'),('Qseed','rng_new','perrequeststatesadvanced','G>0'),
      ('Qdefault','Qwait','globalstreamQready','G!=B'),('Qseed','Qwait','globalstreamQready','G>0'),
      ('Qwait','ratio_P','sameQvaluesandlifetime','randompath'),('race_P','ratio_P','Pmutatedinplace','randompath'),('ratio_P','random_ID','sourcecomparator','randompath'),
      ('random_ID','chosen_ID','randomresult','randomormixed'),('greedy_ID','chosen_ID','greedyresult','mixed'),('mixed_mask','chosen_ID','rowselection','mixed'),
      ('batch_invariant','upstream_filter_boundary','notdefaultfilter','batchinvariant')]
    graph=dict(id='SAMPLER_FILTER_RACE_G7_V91',templates=[dict(id='ASCEND_DEFAULT_PROBABILITY_CUTOFF_AND_RACE',
      nodes=[dict(id=i,kind=k,resource='DECLARED_OPERATOR_OR_STATE')for i,k in spec],
      edges=[dict(producer=a,consumer=b,value=v,dependency_kind='CONDITIONAL_SOURCE_VALUE_OR_EFFECT',condition=c,source='sampling-filter-race-source-contract-v91.json')for a,b,v,c in edges])],
      scope='OnlydefaultAscendfilter/race andupstreamdelegationboundary. Otherbackends/processorprimitivesnotimplicitlymapped.',complete=False,overall_complete=False)
    # Explicit no-filter and k-only value paths,not merely order-adjacency.
    graph['templates'][0]['edges'] +=[dict(producer=a,consumer=b,value='branchvalue',dependency_kind='CONDITIONAL_SOURCE_VALUE',condition=c,
      source='sampling-filter-race-source-contract-v91.json')for a,b,c in [('input_logits','race_P','nokandnop'),('input_logits','returned_scores','nokandnopandprocessed'),('logits_afterK','race_P','kandnotp'),('logits_afterK','returned_scores','kandnotpandprocessed')]]
    write(O/'operator-graph-G7-v91.json',graph)
    nodes=[];ers=[]
    for t in graph['templates']:
        ids={n['id']:n for n in t['nodes']};sets=connected_sets(ids,t['edges']);nc=defaultdict(Counter);ec=defaultdict(Counter)
        for k,subs in sets.items():
            for sub in subs:
                for i in sub:nc[i][k]+=1
                for ei,e in enumerate(t['edges']):
                    if e['producer']in sub and e['consumer']in sub:ec[ei][k]+=1
        for n in t['nodes']:nodes.append(dict(graph='G7',template=t['id'],node_id=n['id'],kind=n['kind'],module='M17',connected_set_counts=dict(nc[n['id']]),candidate_source_node_associations=['F35'],
          structural_scope='ALL_CONNECTED_SETS_OF2_TO4_WITHIN_THIS_DECLARED_GRAPH',semantic_coverage='NOT_ESTABLISHED_BY_ENUMERATION',native_lowering_and_all_runtime_variants_closed=False))
        for i,e in enumerate(t['edges']):ers.append(dict(graph='G7',template=t['id'],edge_index=i,edge=e,connected_set_counts=dict(ec[i]),producer_candidate_associations=['F35'],consumer_candidate_associations=['F35'],semantic_coverage='NOT_ESTABLISHED_BY_ENUMERATION',dependencies_are_not_all_same_invocation_DAG_edges=True))
    for name,extra in [('node',nodes),('edge',ers)]:
        d=deepcopy(read(O/f'operator-graph-{name}-coverage-v90.json'));d['rows']+=extra;d.update(row_count=len(d['rows']),parent=f'operator-graph-{name}-coverage-v90.json',overall_complete=False);write(O/f'operator-graph-{name}-coverage-v91.json',d)
    binding=deepcopy(read(O/'operator-graph-candidate-bindings-v90.json'))
    next(r for r in binding['candidates']if r['id']=='F35')['G7_filter_race_source_supplement']=dict(graph='operator-graph-G7-v91.json',nodes=len(nodes),edges=len(ers),all_F35_primitives_or_variants_mapped=False)
    binding.update(parent='operator-graph-candidate-bindings-v90.json',overall_complete=False);write(O/'operator-graph-candidate-bindings-v91.json',binding)
    old=read(O/'operator-graph-coverage-audit-v90.json');catalog=old['graph_catalog']|{'G7':str(O/'operator-graph-G7-v91.json')}
    write(O/'operator-graph-coverage-audit-v91.json',dict(parent='operator-graph-coverage-audit-v90.json',graph_catalog=catalog,graph_sha256={k:sha(Path(p))for k,p in catalog.items()},
      new_G7_node_rows=len(nodes),new_G7_edge_rows=len(ers),node_rows=old['node_rows']+len(nodes),edge_rows=old['edge_rows']+len(ers),
      finite_source_structural_rows_are_not_unique_operators_or_semantic_completion=True,F35_full_variant_mapping_complete=False,whole_modules_closed=0,overall_complete=False))
    scan=deepcopy(read(R/'fusion-scan.json'));write(O/'fusion-scan-before-v91.json',scan)
    scan['per_node_edge_coverage_ledger']=dict(summary=str(O/'operator-graph-coverage-audit-v91.json'),nodes=str(O/'operator-graph-node-coverage-v91.json'),edges=str(O/'operator-graph-edge-coverage-v91.json'),graph_catalog=catalog,structural_only=True,all_native_or_semantic_variants_complete=False)
    scan['operator_coverage_v91_scope']='G7defaultAscend probabilitycutoff/race supplement;upstreamdifferentfilter andunreviewedTritonbody remainseparate. Noallbranch/nativeclosure.'
    write(O/'fusion-scan-proposed-v91.json',scan)
    print(dict(source_files=3,named_functions=11,if_sites=len(guards),cost_models=len(costs),RNG_cases=len(rng),G7_nodes=len(nodes),G7_edges=len(ers),node_rows=old['node_rows']+len(nodes),edge_rows=old['edge_rows']+len(ers),overall_complete=False))


if __name__=='__main__':main()
