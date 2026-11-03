"""Source-DAG/resource-model refinements and version-bound partial HC fork graph."""
from copy import deepcopy
from collections import defaultdict,Counter
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import O,R,read,write
from reconcile_operator_graph_coverage_v88 import connected_sets


def main():
    assert not(O/'REVIEW-CHECKPOINT-v93.json').exists()
    write(O/'HC-attention-deadline-and-precision-models-v93.json',dict(
      prior_split_DAG_first_review_version=58,existing_split_not_counted_new=True,
      ideal_equal_service_model=dict(symbols='P=currentcoefficientprojection;S=coefficientactivation/Sinkhorn;D=delayedY+norm;A=attentionbranch;H=Post. Unlimitedresources,sameservicecosts,zeroextraABI/launchcost.',
        old_monolithic_reference='max(P+S,D)+A+H',early_coefficient_fork='max(P+S,D+A)+H',defer_coefficient_normalization_into_Post='max(P,D+A)+S+H',
        late_minus_early='min(S,max(D+A-P,0))>=0',
        meaning='MovingnormalizationtoPostcanlosepreviousoverlap;iflaunch/memory/resourcecostschangethentheyrequirearepricedgraph. Thisisnotanactualoptimal-timeorhardwarethroughputbound.'),
      actual_profile_service_components_identified=False,
      source_coefficient_deadlines=['Currentpost/combreadybyactualattentionHcPostrowreads','Currentpre readybyfollowingFFNHcPre;updatedPostresidualalsorequired'],
      input_information_fence=dict(
        counterexample_domain='HC4,oldpremix[1,0,0,0],finiteBF16entries;chooseonecoefficientweightselectingchannel1feature0.',
        X0_firstfeatures=[1,0,0,0],X1_firstfeatures=[1,1,0,0],collapsed_Y_in_both=1,coefficient_dot_before_rstd=[0,1],
        both_normalized_Y_identical=True,HC_full_4H_coefficient_cannot_generically_factor_through_Y=True,
        actual_checkpoint_special_weight_structure_or_value_identity_proved=False),
      descriptor_outputs=dict(pre_FLOAT_values_per_row=4,post_FLOAT_values_per_row=4,comb_FLOAT_values_per_row=16,
        moving_normalization_into_HCPost_needs='ComputeonceperrowwithhandofforpriceperD-tileduplication;sourcePostDsplitand39normalizationroundsnotfree. KeeporiginalFP32/eps/rowFactor/BF16boundary.'),
      input_layout_fence='F01currentcoeffusesoriginal4H BF16→FLOAT,QAusesRMS(BF16Y)localrows,SWAglobalrows. Jointcompoundkernelsstillneedtwonumeric/valuebranches,notautomaticallyonewideliner.',
      new_native_or_precision_or_speedup_proved=False,overall_complete=False))
    variants=[
      dict(id='HC93-EARLY-Y',state='EXISTING_V58_DIRECTION_REPRICED_AND_PROFILE_BOUND',change='Exposeexternal-premixY/normattentionbranchbeforecurrentcoeffcompletion;keepcoeffdeadlineatPost/nextPre.',gates='NativeYvisibility/newABI,exactsourceprecision,allreaders/epochs,sharedresourcesandlaunchcosts. Oldobservedspannotfreeslack.'),
      dict(id='HC93-COMPOUND',state='CONDITIONAL_SAME_BLOCK_COMPOSITION',change='ComposeHCY/norm/query/KV/outputpathwhilecurrentcoeffforkisretainedonce.',gates='Distinct4H/Htypesandlocal/globalrowowners;oneunionweightservicecut;C2/C1/IndexerstateandPostrawresidualreadersremain.'),
      dict(id='HC93-COEFF-POST',state='CONDITIONAL_FUSION_WITH_POSSIBLE_LOST_OVERLAP',change='CombinecoefficientfinalnormalizationwithPostconsumerandavoidprivatecoefficientinterfaceswhereallowed.',gates='Idealunchanged-serviceDAGcannotoutperformearlycoeffforkbythismovalone. RepriceVectorcontention/launch/IO andperDtileduplicatednormalization;pre outputforFFNPrestilllive.'),
      dict(id='HC93-COLUMNS',state='CONDITIONAL_PRIORITY_SPLIT_NOT_FREE_COMPUTE',change='Prioritizepost/comb20columns versuspre4laterdeadline.',gates='Originalall24alreadyoneprojection. IndependentM/N16paddingreferenceurgent20costsamereferenceas24;split4addsservice. SharedinputRMS/weights/pre outputsandnativeHF32orderpreserved.'),
      dict(id='HC93-WEIGHT-SERVICE',state='CONDITIONAL_SOURCE_RESOURCE_CUT',change='AssessHCprojectionandattentionplanesinonesharedcapacitywindowwithoneinitialwarmgrant.',gates='UncompresseddistinctweightsanddeclaredL1→L0pathmandatory;otheralgorithms/GMdirect/compression/pruning areseparatedomains. Noactualmemorybottleneckorclockboundclaimed.')]
    write(O/'HC-attention-fork-variants-v93.json',dict(variants=variants,existing_split_and_prior_precision_facts_not_new_discovery=True,new_maincandidate=False,implementation=False,native_test=False,overall_complete=False))
    # New hierarchical source view only;native tensors/aliases are not inferred from graph adjacency.
    spec=[('HC_residual_old','ORIGINAL_LOCAL_HC4_RESIDUAL'),('external_pre_mix','OLD_PREMIX_FROM_PREVIOUS_BOUNDARY'),
      ('current_coeff','CURRENT_FULL4H_HF32_COEFFICIENT_PATH'),('delayed_Y','DELAYED_PREMIX_COLLAPSE_BF16'),('Y_norm','FULL_H_RMS_BF16'),
      ('owner_publish','NORMALIZED_Y_OWNER_GATHER_SLICE'),('QA','LOCAL_BF16_QA'),('Qnorm_DQ','QR_NORM_AND_QUANT_RETAIN_ROUNDING'),('QB','LOCAL_INT8_QB'),
      ('SWA','GLOBAL_BF16_SWA_KV'),('KVnorm_RoPE_store','KV_NORM_ROPE_AND_STATE_PUBLICATION'),('attention','QUERY_KV_STATE_ATTENTION'),
      ('output_A','INVERSE_ROPE_RESTORE_WOA'),('output_B','WOB'),('output_RS','OWNER_OUTPUT_REDUCE_SCATTER'),
      ('attention_Post','CURRENT_POST_COEFF_RESIDUAL_JOIN'),('FFN_Pre','UPDATED_RESIDUAL_NEXT_PREMIX_HCPRE'),('FFN_norm','RMSNORMCAST_FOR_FFN'),('coeff_deadline','CURRENT_POST_COMB_AND_NEXT_PRE_LIVE_OUTPUTS')]
    es=[('HC_residual_old','current_coeff','full4H/rstd'),('HC_residual_old','delayed_Y','sameinputdistinctwork'),('external_pre_mix','delayed_Y','oldpremix'),
      ('delayed_Y','Y_norm','BF16boundary'),('Y_norm','owner_publish','normalizedY'),('owner_publish','QA','localqueryrows'),('QA','Qnorm_DQ','BF16QR'),('Qnorm_DQ','QB','samequantizationcontract'),
      ('owner_publish','SWA','globalKVrows'),('SWA','KVnorm_RoPE_store','persistentstatefork'),('QB','attention','queryrounding'),('KVnorm_RoPE_store','attention','currentstate/readiness'),
      ('attention','output_A','attentionresults'),('output_A','output_B','twolevelprojectionboundary'),('output_B','output_RS','ownerpublication'),('output_RS','attention_Post','attentionresult'),
      ('current_coeff','coeff_deadline','currentpost/comb/pre'),('coeff_deadline','attention_Post','post/combdeadline'),('HC_residual_old','attention_Post','rawresidualstilllive'),
      ('attention_Post','FFN_Pre','updatedresidual'),('coeff_deadline','FFN_Pre','currentprecoeff'),('FFN_Pre','FFN_norm','nextBF16Y')]
    associations={'HC_residual_old':['F01','F02'],'external_pre_mix':['F01','F02'],'current_coeff':['F01','F02'],'delayed_Y':['F01','F02'],'Y_norm':['F02'],
      'owner_publish':['F02','F06'],'QA':['F06'],'Qnorm_DQ':['F05'],'QB':['F05'],'SWA':['F06','F08'],'KVnorm_RoPE_store':['F08'],
      'attention':['F14'],'output_A':['F14','F16'],'output_B':['F16'],'output_RS':['F15','F16'],'attention_Post':['F03'],
      'FFN_Pre':['F03'],'FFN_norm':['F04'],'coeff_deadline':['F01']}
    template=dict(id='MHC_ATTENTION_COEFFICIENT_FORK',nodes=[dict(id=i,kind=k,resource='DECLARED_OPERATOR_OR_VALUE_EPOCH')for i,k in spec],
      edges=[dict(producer=a,consumer=b,value=v,dependency_kind='SOURCE_OPERATOR_VALUE_OR_LIFETIME',condition='Sameblock/sourceepoch;newYearlyhandoffconditional',source='HC-attention-fork-source-contract-v93.json')for a,b,v in es])
    g=deepcopy(read(O/'operator-graph-G5-v88.json'));next(t for t in g['templates']if t['id']=='MHC_BLOCK_AND_CROSS_LAYER')['coefficient_fork_source_view_v93']=dict(template=template['id'],prior_split_source_first_version=58,native_handoff_proved=False)
    g['templates'].append(template);g.update(id='SOURCE_TEMPLATES_G5_V93',parent='operator-graph-G5-v88.json',whole_model_complete=False);write(O/'operator-graph-G5-v93.json',g)
    nodes={n['id']:n for n in template['nodes']};sets=connected_sets(nodes,template['edges']);nc=defaultdict(Counter);ec=defaultdict(Counter)
    for k,subs in sets.items():
        for sub in subs:
            for i in sub:nc[i][k]+=1
            for ei,e in enumerate(template['edges']):
                if e['producer']in sub and e['consumer']in sub:ec[ei][k]+=1
    newn=[dict(graph='G5',template=template['id'],node_id=n['id'],kind=n['kind'],module='M06/M07/M10/M11/M12',connected_set_counts=dict(nc[n['id']]),candidate_source_node_associations=associations[n['id']],
      structural_scope='ALL_CONNECTED_SETS_OF2_TO4_WITHIN_THIS_DECLARED_GRAPH',semantic_coverage='NOT_ESTABLISHED_BY_ENUMERATION',native_lowering_and_all_runtime_variants_closed=False)for n in template['nodes']]
    newe=[dict(graph='G5',template=template['id'],edge_index=i,edge=e,connected_set_counts=dict(ec[i]),producer_candidate_associations=associations[e['producer']],consumer_candidate_associations=associations[e['consumer']],
      semantic_coverage='NOT_ESTABLISHED_BY_ENUMERATION',dependencies_are_not_all_same_invocation_DAG_edges=True)for i,e in enumerate(template['edges'])]
    for name,extra in [('node',newn),('edge',newe)]:
        d=deepcopy(read(O/f'operator-graph-{name}-coverage-v92.json'));d['rows']+=extra;d.update(row_count=len(d['rows']),parent=f'operator-graph-{name}-coverage-v92.json',overall_complete=False);write(O/f'operator-graph-{name}-coverage-v93.json',d)
    bindings=deepcopy(read(O/'operator-graph-candidate-bindings-v92.json'))
    for row in bindings['candidates']:
        ids=[i for i,cs in associations.items()if row['id']in cs]
        if ids:row['HC_fork_source_supplement_v93']=dict(template=template['id'],nodes=ids,all_variants_mapped=False,native_handoff_or_alias_proved=False)
    # Rebuild source-navigation hints from current partial node/edge ledgers,including newer supplements.
    nr=read(O/'operator-graph-node-coverage-v93.json')['rows'];er=read(O/'operator-graph-edge-coverage-v93.json')['rows'];lookup={(r['graph'],r['template'],r['node_id']):r['candidate_source_node_associations']for r in nr}
    hints=defaultdict(list);children=defaultdict(set)
    for r in er:
        gr,ti=r['graph'],r['template'];e=r['edge'];a,b=e['producer'],e['consumer'];children[(gr,ti,a)].add(b)
        for ca in lookup[(gr,ti,a)]:
            for cb in lookup[(gr,ti,b)]:
                if ca!=cb:hints[tuple(sorted([ca,cb]))].append(dict(graph=gr,template=ti,edge_index=r['edge_index'],kind='DECLARED_EDGE_ENDPOINT_ASSOCIATIONS'))
    from itertools import combinations
    for (gr,ti,a),cs in children.items():
        for l,r in combinations(sorted(cs),2):
            for ca in lookup[(gr,ti,l)]:
                for cb in lookup[(gr,ti,r)]:
                    if ca!=cb:hints[tuple(sorted([ca,cb]))].append(dict(graph=gr,template=ti,producer=a,children=[l,r],kind='COMMON_INPUT_SIBLING_ASSOCIATIONS'))
    allclauses=read(O/'candidate-relation-clauses-v92.json')['clauses']+read(O/'HC-attention-composition-clauses-v93.json')['clauses']
    bindings['structural_pair_hints']=[dict(a=a,b=b,locations=loc,existing_conditional_clause_ids=[c['id']for c in allclauses if{c['a'],c['b']}=={a,b}],structure_does_not_prove_same_call_or_joint_legality=True)for (a,b),loc in sorted(hints.items())]
    bindings.update(parent='operator-graph-candidate-bindings-v92.json',current_partial_graph_hints_rebuilt=True,overall_complete=False);write(O/'operator-graph-candidate-bindings-v93.json',bindings)
    old=read(O/'operator-graph-coverage-audit-v92.json');catalog=old['graph_catalog']|{'G5':str(O/'operator-graph-G5-v93.json')}
    write(O/'operator-graph-coverage-audit-v93.json',dict(parent='operator-graph-coverage-audit-v92.json',graph_catalog=catalog,graph_sha256={k:hashlib.sha256(Path(p).read_bytes()).hexdigest()for k,p in catalog.items()},
      new_source_node_rows=len(newn),new_source_edge_rows=len(newe),node_rows=724+len(newn),edge_rows=1187+len(newe),current_structural_pair_hints=len(hints),
      prior_split_source_answer_first_version=58,finite_source_structural_rows_are_not_unique_operators_or_semantic_completion=True,whole_modules_closed=0,overall_complete=False))
    scan=deepcopy(read(R/'fusion-scan.json'));write(O/'fusion-scan-before-v93.json',scan)
    scan['per_node_edge_coverage_ledger']=dict(summary=str(O/'operator-graph-coverage-audit-v93.json'),nodes=str(O/'operator-graph-node-coverage-v93.json'),edges=str(O/'operator-graph-edge-coverage-v93.json'),graph_catalog=catalog,structural_only=True,all_native_or_semantic_variants_complete=False)
    scan['operator_coverage_v93_scope']='HCattentionfork/deadlineviewandnative3200frontierlink;splitfactpredatesv93,nativecomponenttimes/alias/precisionremainopen.';write(O/'fusion-scan-proposed-v93.json',scan)
    print(dict(variants=5,new_source_nodes=len(newn),new_source_edges=len(newe),node_rows=724+len(newn),edge_rows=1187+len(newe),current_structural_pair_hints=len(hints),overall_complete=False))


if __name__=='__main__':main()
