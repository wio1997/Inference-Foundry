"""Version-bind finite structural coverage; do not promote enumeration to semantic proof."""
from collections import defaultdict,Counter
from copy import deepcopy
from pathlib import Path
from itertools import combinations
import hashlib,zipfile
from finalize_origin_coverage_v9 import O,R,W,read,write

def receipt(path):
    b=path.read_bytes();return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def connected_sets(nodes,edges):
    neighbors={n:set()for n in nodes}
    for e in edges:
        a,b=e['producer'],e['consumer'];assert a in nodes and b in nodes
        if a!=b:neighbors[a].add(b);neighbors[b].add(a)
    current={(n,)for n in nodes};result={}
    for size in[2,3,4]:
        nxt=set()
        for subset in current:
            available=set().union(*(neighbors[n]for n in subset))-set(subset)
            for n in available:nxt.add(tuple(sorted(subset+(n,))))
        result[size]=nxt;current=nxt
    return result

def main():
    assert not(O/'operator-graph-coverage-audit-v88.json').exists()
    base1=R/'graph-revision-aux-context-v2/G1.json';base5=R/'graph-revision-aux-context-v2/G5.json'
    g1=read(base1);g5=deepcopy(read(base5));legacy=read(R/'coverage/summary.json')
    mismatch={k:dict(legacy=legacy['graph_sha256'][k],current=receipt(p)['sha256'])for k,p in [('G1',base1),('G5',base5)]}
    assert all(v['legacy']!=v['current']for v in mismatch.values())
    # Existing v36 source answer is now represented in the stale coarse graph.
    t=next(t for t in g5['templates']if t['id']=='DSPARK_BOUNDARIES');nodes={n['id']:n for n in t['nodes']}
    nodes['last_hidden_dynamic_input'].update(kind='RAW_DSPARK_LAST_HIDDEN_ARGUMENT',source_resolution='v36 source answer;pre-head-selection last_hidden_states,not normalized/sample_hidden_states')
    nodes['confidence'].update(kind='FP32_CONFIDENCE_WITH_MARKOV_INPUT_EMBEDDING_AND_SIGMOID',source_resolution='v36 DSparkConfidenceHead;raw H plus ordered seed/previous-input embedding')
    nodes['dynamic_scheduler'].update(kind='CONDITIONAL_FP32_SURVIVAL_SHARED_BUDGET_GLOBAL_TOPK',source_resolution='v36 DynamicSpecScheduler shared policy;actual runtime flags/counter values/native precision remain unknown')
    removed=[e for e in t['edges']if(e['producer'],e['consumer'])in[('last_hidden_dynamic_input','dynamic_scheduler'),('draft_token_output','dynamic_scheduler')]]
    assert len(removed)==2
    t['edges']=[e for e in t['edges']if e not in removed]
    extra_nodes=[('ordered_markov_inputs','SEED_AND_PREVIOUS_PREDICTION_INPUT_SEQUENCE'),
      ('dynamic_policy_state_old','EXTERNAL_POLICY_STATE_OLD_EPOCH'),('dynamic_policy_state_new','EXTERNAL_POLICY_STATE_NEW_EPOCH'),
      ('dynamic_keep_lengths','PER_REQUEST_DEVICE_KEEP_LENGTHS'),('retrieved_trimmed_draft_ids','CONDITIONAL_ORDERED_HOST_PREFIX_OUTPUT')]
    for name,kind in extra_nodes:
        assert name not in nodes;t['nodes'].append(dict(id=name,kind=kind,resource='MIXED_OR_METADATA'))
    added=[('sp_gather_slice','last_hidden_dynamic_input','raw last_hidden_states;before head selection'),
      ('last_hidden_dynamic_input','confidence','raw hidden.reshape(B*D,H)'),('seed','ordered_markov_inputs','T[:,0]'),
      *[(f'sample_{i}','ordered_markov_inputs',f'T[:,{i+1}]')for i in range(4)],
      ('ordered_markov_inputs','confidence','T[:B,:D],D5 template;excludes final prediction sample4'),
      ('confidence','dynamic_scheduler','copied/clamped FP32 token_probs'),
      ('dynamic_policy_state_old','dynamic_scheduler','previous budget/counter/config'),
      ('dynamic_scheduler','dynamic_policy_state_new','new budget/counter epoch'),
      ('dynamic_scheduler','dynamic_keep_lengths','FP32 cumprod/global TopK/scatter keep counts'),
      ('dynamic_keep_lengths','retrieved_trimmed_draft_ids','per-request clamp+prefix selection'),
      ('draft_token_output','retrieved_trimmed_draft_ids','untrimmed predicted output sequence')]
    assert len(added)==14
    for a,b,value in added:t['edges'].append(dict(producer=a,consumer=b,value=value,dependency_kind='SOURCE_DATA_OR_STATE_EPOCH',
      condition='DSpark dynamic scheduler reached;D5 frozen structural template. Other branches/flags remain explicit.',
      source='M20-Markov-confidence-and-verification-contract-v36.json;F30-dynamic-verify-budget-and-prefix-bounds-v36.json',
      grade='PRIOR_DOCUMENTED_SOURCE_CONTRACT_RECONCILED_NOT_NEW_RUNTIME_EVIDENCE'))
    t.update(complete_primitive_lowering=False,source_reconciliation_v88=dict(prior_answer_version=36,new_semantic_discovery=False,
      removed_stale_edges=removed,added_node_ids=[x[0]for x in extra_nodes],added_edges=len(added),
      no_all_native_consumer_or_loaded_binary_proof=True))
    g5.update(id='SOURCE_TEMPLATES_G5_OPERATOR_RECONCILED_V88',parent=str(base5),parent_sha256=receipt(base5)['sha256'],whole_model_complete=False)
    g1=deepcopy(g1);g1.update(id='G1_OPERATOR_SCOPE_AUX_V88',parent_graph_id=read(base1)['id'],parent_graph_sha256=receipt(base1)['sha256'],complete=False,
      hierarchical_scope_note='G1coarse model modules and G5partial source templates overlap. dspark.head includes the reconciled DSPARK_BOUNDARIES confidence/policy branch;no implicit primitive/native equality or new numerical graph claim.')
    write(O/'operator-graph-G1-v88.json',g1);write(O/'operator-graph-G5-v88.json',g5)
    mappings=defaultdict(set);candidate_records=[]
    fps=read(O/'candidate-operator-footprints-v87.json')['candidates'];templates={t['id']:t for t in g5['templates']}
    for c in fps:
        tid=c.get('original_template_graph_id');bound=[];bad=[]
        if tid in templates:
            ids={n['id']for n in templates[tid]['nodes']}
            for a,b in(c.get('original_node_mapping')or{}).items():
                if a in ids:mappings[(tid,a)].add(c['id']);bound.append(a)
                else:bad.append(dict(key=a,value=b))
        candidate_records.append(dict(id=c['id'],original_template=tid,original_template_node_associations=bound,
          invalid_keys=bad,source_question_or_native_semantics_proved_by_binding=False,all_variants_mapped=False))
    # Explicit current source associations,not blanket completion of the metadata candidate.
    supplement={'F30':['last_hidden_dynamic_input','ordered_markov_inputs','confidence','dynamic_scheduler','dynamic_keep_lengths','retrieved_trimmed_draft_ids'],
      'F31':['dynamic_policy_state_old','dynamic_scheduler','dynamic_policy_state_new','dynamic_keep_lengths','retrieved_trimmed_draft_ids']}
    for cid,ids in supplement.items():
        for nid in ids:mappings[('DSPARK_BOUNDARIES',nid)].add(cid)
        next(c for c in candidate_records if c['id']==cid)['prior_v36_source_supplement']=dict(template='DSPARK_BOUNDARIES',nodes=ids,scope='Only this dynamic-policy source path;not full candidate primitive graph.')
    graphs=[('G1','',g1)]+[('G5',t['id'],t)for t in g5['templates']]
    nrows=[];erows=[];counts=[];relations=read(O/'candidate-relation-clauses-v87.json')['clauses'];pair_hints=defaultdict(list)
    for graph,tid,g in graphs:
        ids={n['id']:n for n in g['nodes']};sets=connected_sets(ids,g['edges']);nc=defaultdict(Counter);ec=defaultdict(Counter)
        edgekey={(min(e['producer'],e['consumer']),max(e['producer'],e['consumer'])):[]for e in g['edges']}
        for i,e in enumerate(g['edges']):edgekey[(min(e['producer'],e['consumer']),max(e['producer'],e['consumer']))].append(i)
        for size,items in sets.items():
            for sub in items:
                for n in sub:nc[n][size]+=1
                for a,b in combinations(sub,2):
                    for i in edgekey.get((a,b),[]):ec[i][size]+=1
        for n in g['nodes']:
            nrows.append(dict(graph=graph,template=tid,node_id=n['id'],kind=n['kind'],module=n.get('module',tid),
              connected_set_counts=dict(nc[n['id']]),candidate_source_node_associations=sorted(mappings[(tid,n['id'])])if graph=='G5'else[],
              structural_scope='ALL_CONNECTED_SETS_OF2_TO4_WITHIN_THIS_DECLARED_GRAPH',semantic_coverage='NOT_ESTABLISHED_BY_ENUMERATION',
              native_lowering_and_all_runtime_variants_closed=False))
        for i,e in enumerate(g['edges']):
            aa=mappings[(tid,e['producer'])]if graph=='G5'else set();bb=mappings[(tid,e['consumer'])]if graph=='G5'else set()
            erows.append(dict(graph=graph,template=tid,edge_index=i,edge=deepcopy(e),connected_set_counts=dict(ec[i]),
              producer_candidate_associations=sorted(aa),consumer_candidate_associations=sorted(bb),semantic_coverage='NOT_ESTABLISHED_BY_ENUMERATION',
              dependencies_are_not_all_same_invocation_DAG_edges=True))
            for a in aa:
                for b in bb:
                    if a!=b:pair_hints[tuple(sorted((a,b)))].append(dict(template=tid,edge_index=i,kind='DECLARED_EDGE_ENDPOINT_ASSOCIATIONS'))
        # Common-input sibling fusion cannot be represented only by directed paths.
        children=defaultdict(set)
        for e in g['edges']:children[e['producer']].add(e['consumer'])
        if graph=='G5':
            for producer,cs in children.items():
                for left,right in combinations(sorted(cs),2):
                    for a in mappings[(tid,left)]:
                        for b in mappings[(tid,right)]:
                            if a!=b:pair_hints[tuple(sorted((a,b)))].append(dict(template=tid,producer=producer,children=[left,right],kind='COMMON_INPUT_SIBLING_ASSOCIATIONS'))
        counts.append(dict(graph=graph,template=tid,nodes=len(ids),edges=len(g['edges']),connected_subsets={str(k):len(v)for k,v in sets.items()},
          includes_forks_and_state_adjacency=True,all_semantic_combinations_analyzed=False))
    hints=[]
    for pair,locations in sorted(pair_hints.items()):
        clauses=[c['id']for c in relations if{c['a'],c['b']}==set(pair)]
        hints.append(dict(a=pair[0],b=pair[1],locations=locations,existing_conditional_clause_ids=clauses,
          structure_does_not_prove_same_call_or_joint_legality=True))
    write(O/'operator-graph-node-coverage-v88.json',dict(rows=nrows,row_count=len(nrows),overall_complete=False))
    write(O/'operator-graph-edge-coverage-v88.json',dict(rows=erows,row_count=len(erows),overall_complete=False))
    write(O/'operator-graph-candidate-bindings-v88.json',dict(candidates=candidate_records,structural_pair_hints=hints,
      supplemental_source_answers_first_established_version=36,not_native_semantic_or_all_variant_coverage=True,overall_complete=False))
    modules=read(O/'module-phase-branch-matrix-v87.json')['modules'];ops=read(O/'operator-analysis-coverage-ledger-v87.json')
    write(O/'operator-graph-coverage-audit-v88.json',dict(status='FINITE_GRAPH_REVISION_AND_COVERAGE_RECONCILED_GLOBAL_SCOPE_OPEN',
      input_receipts=[receipt(p)for p in[base1,base5,R/'coverage/summary.json',R/'coverage/node-structural-coverage.csv',R/'coverage/edge-structural-coverage.csv',O/'candidate-operator-footprints-v87.json',O/'M20-Markov-confidence-and-verification-contract-v36.json']],
      legacy_counts=dict(nodes=legacy['node_rows'],edges=legacy['edge_rows']),legacy_graph_hash_mismatches=mismatch,
      current_counts=dict(G1_nodes=len(g1['nodes']),G1_edges=len(g1['edges']),G5_templates=len(g5['templates']),
        G5_nodes=sum(len(t['nodes'])for t in g5['templates']),G5_edges=sum(len(t['edges'])for t in g5['templates']),node_rows=len(nrows),edge_rows=len(erows)),
      enumeration_counts=counts,source_confidence_graph_reconciliation_is_prior_answer_not_new_candidate=True,
      requirements=[
        dict(id='MODEL_AND_LOAD_UNIVERSE',state='PARTIAL',evidence='Pinned model/config/serve and native profiles define recorded scope;G1/G5 explicitly incomplete,rare/applicable variants still open.'),
        dict(id='ALL_MODULE_BRANCHES',state='PARTIAL',modules=len(modules),semantic_modules=sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in modules),records=sum(len(m['branches'])for m in modules),chronological_only=[m['id']for m in modules if m.get('branch_schema')!='SEMANTIC_BRANCHES_V1']),
        dict(id='FINITE_GRAPH_NODE_EDGE_LEDGER',state='VERSION_BOUND_STRUCTURAL_LEDGER_NOW_RECONCILED',evidence='Current G1aux/G5source revision plus v36 confidence/policy binding;structural rows not semantics.'),
        dict(id='CROSS_OPERATOR_LAYER_CARD_ITERATION',state='PARTIAL',model_candidates=35,conditional_pair_clauses=112,higher_order_groups=23,remaining_central_pair_navigation=483,evidence='Pair/cartesian counts are not an all-opportunity coverage percentage.'),
        dict(id='ALL_NATIVE_OPERATOR_REGIMES',state='PARTIAL',reviewed=[2073,285598],pending=[9,1352],evidence='ZerosLike2/MemSet7 unresolved;review is conditional algorithm/resource scope.'),
        dict(id='THEORETICAL_NUMERICAL_CEILINGS',state='INCOMPLETE',evidence='Actual FLOAT/HF32/Vector modes/frequency/bandwidth uppers and mixed capacity unbound;published peak fractions are references.'),
        dict(id='KEY_FUSION_APPLICABILITY_EVIDENCE',state='INCOMPLETE',evidence='All relevant row/slot/observer/value/parameter epochs,precision,aliases and capacity conditions not closed. Implementation/benchmark is not the analysis deliverable.'),
        dict(id='MODULE_MINIMUM_UTILIZATION',state='INCOMPLETE',evidence='Hierarchical whole-module denominators and matching hardware capacity not established;do not rank zero-assignment modules zero.'),
        dict(id='GLOBAL_COMPLETION',state='NOT_PROVED',evidence='Unanalyzed scopes/critical evidence remain;report/checkpoint/finite enumeration cannot complete Goal.')],
      unchanged_global_uncertainty=ops['unchanged_global_uncertainty'],new_device_or_remote_work=False,whole_modules_closed=0,overall_complete=False))
    print(dict(graph_counts=[len(g1['nodes']),len(g1['edges']),sum(len(t['nodes'])for t in g5['templates']),sum(len(t['edges'])for t in g5['templates'])],
      ledger_rows=[len(nrows),len(erows)],pair_hints=len(hints),legacy_graphs_stale=True,confidence_fact_new=False,overall_complete=False))

if __name__=='__main__':main()
