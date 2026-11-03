"""Verify bounded F30-F31 composition evidence, not full runtime legality."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,T,L,read,write
from query_verify_metadata_profile_v129 import frame_roles
from verify_metadata_relation_models_v129 import publish_prefixes,prefix_layout,next_count,topk_counts,enumerate_ties


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(name,v):
        assert v,name
        checks.append(name)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'verify-metadata-inputs-v129.json')['inputs']:ck('input:'+r['path'],pinned(r))
    source=read(O/'verify-metadata-source-contract-v129.json')
    for p in source['parts']:ck('source:'+str((p['path'],p['first_line'])),pinned(p)and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    prof=read(O/'verify-metadata-profile-scope-v129.json');bydb=defaultdict(list);counts=Counter();keys=set()
    for a in prof['selected_APIs']:
        key=(a['database_index'],a['api_rowid']);assert key not in keys;keys.add(key)
        roles=sorted({r for f in a['matched_frames']for r in frame_roles(f['frame'])})
        assert roles==a['roles']and a['endNs']>=a['startNs']
        counts.update(roles);bydb[a['database_index']].append(a)
    ck('183813profile_APIs',len(keys)==183813 and dict(counts)==prof['API_role_counts'])
    originals={r['database_index']:r for r in read(O/'host-launch-original-API-coverage-v126.json')['databases']}
    for db in prof['databases']:
        di=db['database_index'];ck('rawDBpin:'+str(di),pinned(db)and db['sha256']==originals[di]['sha256']and db['stat_unchanged'])
        role_count=Counter();tasksets=defaultdict(set);sample={}
        for a in bydb[di]:
            for role in a['roles']:
                role_count[role]+=1;tasksets[role].update(a['native_TASK_rowids'])
                sample.setdefault((role,'first'),a);sample[(role,'last')]=a
        ck('rolecounts:'+str(di),dict(role_count)==db['API_role_counts']and {r:len(s)for r,s in tasksets.items()}==db['TASK_role_counts'])
        # Independent first/last positive raw-row samples for every recorded role.
        # Complete-source selectors are preserved; this is not negative flag proof.
        p=Path(db['path']);con=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row
        for (role,side),a in sample.items():
            raw=con.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone()
            ck('raw_API_sample:'+str((di,role,side)),all((int(raw[k])if k in ['startNs','endNs']else raw[k])==a[k]for k in raw.keys()))
        con.close()
    for role in ['DYNAMIC_SCHEDULER_SOURCE_SURFACE','KEEP_LENGTH_RETRIEVAL_SOURCE_SURFACE','CONFIDENCE_WRAPPER_SOURCE_SURFACE','CONFIDENCE_HEAD_SOURCE_SURFACE']:
        ck('absence_not_flag_proof:'+role,prof['API_role_counts'].get(role,0)==0 and not prof['dynamic_feature_or_metadata_values_attested'])
    ck('native_seed_publication_counts',prof['TASK_role_counts']['SEED_PUBLICATION']==104 and prof['TASK_role_counts']['ID_PUBLICATION']==520 and prof['TASK_role_counts']['NEXT_ID_COUNT']==1232)
    kinds=Counter()
    for db in prof['databases']:kinds.update(db['TASK_types_by_role'].get('NEXT_ID_COUNT',{}))
    ck('nextcount1144typed88undefined',sum(kinds.values())==1232 and kinds['NO_COMPUTE']==88 and kinds['LogicalAnd']==88 and kinds['SelectV2']==88)
    layouts=read(O/'verify-metadata-layout-models-v129.json');ck('2834layouts',layouts['case_count']==len(layouts['cases'])==2834 and not layouts['actual_current_device_k_to_qsl_kernel_or_read_cost_established'])
    seen=set()
    for c in layouts['cases']:
        B,D,ell=c['B'],c['D'],c['lengths'];key=(B,D,tuple(ell));assert key not in seen;seen.add(key)
        expected=[sum(ell[:i])+i for i in range(B+1)];owner=[i for i,n in enumerate(ell)for _ in range(n+1)]
        ck('prefixlayout:'+str(key),c['qsl']==expected and c['owner']==owner and c['Q']==sum(ell)+B and c['Q']<2**31)
        ck('content_not_HBM:'+str(key),c['new_ABI_logical_content_bytes']==4*B+4*(B+1)+4*c['Q']and not c['actual_baseline_HBM_or_latency_saving'])
    witness=read(O/'verify-metadata-value-and-epoch-witnesses-v129.json')
    for row in witness['proposal_keep_vs_target_acceptance']:
        k=row['proposal_keep_length'];ck('keep_not_accept:'+str(k),row['possible_target_valid_counts']==list(range(k+2))and row['target_query_length']==k+1)
        for out in row['outcomes']:ck('outcome:'+str((k,out['accepted_drafts'])),next_count(out['sampled'],99,100)==(out['next_ID'],out['valid_count'])and out['valid_count']==out['accepted_drafts']+1)
    ck('tie_positions_not_prefix',topk_counts([[4,4,4,4]],1,1,[2])==[2]and publish_prefixes([[10,11,12,13]],[2],4)==[[10,11]])
    c=witness['same_total_different_row_layout'];ck('total_not_owner',len(c['owner'])==len(c['uniform_owner'])==6 and c['qsl']!=c['uniform_qsl']and c['owner']!=c['uniform_owner']and c['reordered_qsl']!=c['qsl'])
    ck('cap_actual_width',publish_prefixes([[1,2],[3,4,5]],[5,0],5)==[[1,2],[]])
    ck('source_zip_not_fabricated_rows',publish_prefixes([[1,2],[3,4]],[1],5)==[[1]])
    ck('literal_holes_and_negative',next_count([7,-1,9],99,100)==(-1,2)and next_count([-2,5,-1],99,100)==(5,2))
    tie=read(O/'verify-metadata-tie-reference-validation-v129.json');again=enumerate_ties()
    ck('tie_reference_domain',tie['candidate_subsets']==again['candidate_subsets']==65028 and tie['valid_TopK_subsets']==14182 and tie['nonprefix_selected_position_cases']==8938 and not tie['native_tie_policy_or_runtime_values_attested'])
    graph=read(O/'verify-metadata-generation-graph-v129.json');nodes={r['id']for r in graph['nodes']};edges=defaultdict(set)
    for e in graph['edges']:ck('edge:'+e['producer']+e['consumer'],e['producer']in nodes and e['consumer']in nodes);edges[e['producer']].add(e['consumer'])
    def reachable(a,b):
        pending=list(edges[a]);seen=set()
        while pending:
            x=pending.pop()
            if x==b:return True
            if x not in seen:seen.add(x);pending.extend(edges[x])
        return False
    ck('graph_acyclic',not any(reachable(x,x)for x in nodes))
    ck('no_retroactive_prune',reachable('M0','K0')and not reachable('K0','M0')and not reachable('K0','S0')and reachable('K0','S1'))
    ck('local_not_native_graph',graph['native_graph_nodes_added']==0 and 'Nonparallel autoregressive continuation remains a separate branch'in graph['domain'])
    contract=read(O/'verify-metadata-joint-contract-v129.json');ck('gated_relation',contract['pair']==['F30','F31']and contract['new_pair_clauses']==1 and contract['new_higher_order_groups']==0 and len(contract['reuse_conditions'])==5 and not contract['new_device_work'])
    if args.preflight:
        old=read(O/'candidate-relation-coverage-v128.json');p=next(p for p in old['pairs']if {p['a'],p['b']}=={'F30','F31'})
        ck('actually_pending_before',p['scope_state']!='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL')
        manifest=read(O/'checkpoint-manifest-v128.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v129.json');ck('open_progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v129.json');pairs={tuple(sorted((r['a'],r['b'])))for r in rel['clauses']}
        ck('136pairs136clauses26groups',len(rel['clauses'])==len(pairs)==136 and len(rel['higher_order_groups'])==26 and rel['clause_count']==rel['selected_pair_count']==136)
        p=next(r for r in rel['clauses']if r['id']=='REL129-01');ck('relation_conditional',p['a']=='F30'and p['b']=='F31'and not p['all_runtime_gates_proved']and p['critical_path_gain_us']is None)
        cov=read(O/'candidate-relation-coverage-v129.json');ck('136reviewed459pending',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(136,26,459)and len(cov['pairs'])==595)
        p=next(r for r in cov['pairs']if {r['a'],r['b']}=={'F30','F31'});ck('pair_index_reconciled',p['clauses']==['REL129-01']and p['scope_state']=='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL'and not p['whole_pair_all_variants_closed'])
        nav=read(O/'candidate-pair-navigation-v129.json');p=next(r for r in nav['pairs']if {r['a'],r['b']}=={'F30','F31'});ck('navigation_reconciled',p['conditional_clause_ids']==['REL129-01']and p['semantic_pair_reconciliation_done_in_stated_scope']and nav['scoped_conditional_pairs']==136 and nav['remaining_central_pair_reconciliation']==459)
        matrix=read(O/'module-phase-branch-matrix-v129.json');ck('same358_26',len(matrix['modules'])==26 and sum(len(x['branches'])for x in matrix['modules'])==358)
        old=read(O/'operator-analysis-coverage-ledger-v128.json');new=read(O/'operator-analysis-coverage-ledger-v129.json');ck('native_uncertainty_unchanged',all(old[k]==new[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v129.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v129.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    out=dict(status='PASS',checks=len(checks),preflight=args.preflight,profile_APIs_checked=183813,raw_positive_API_samples=True,
        layout_cases=2834,tie_candidate_subsets=65028,new_scoped_pair=1,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'verify-metadata-preflight-v129.json',out)
    print(json.dumps(out))


if __name__=='__main__':main()
