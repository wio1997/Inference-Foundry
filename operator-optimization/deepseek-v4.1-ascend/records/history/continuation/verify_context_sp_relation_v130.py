"""Verify conditional context/SP scope, positive dummy branch and cost bounds."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,T,L,read,write
from context_sp_models_v130 import carrier_case,recipient_cases,writer_witness


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(name,v):
        assert v,name
        checks.append(name)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'context-sp-inputs-v130.json')['inputs']:ck('input:'+r['path'],pinned(r))
    source=read(O/'context-sp-source-contract-v130.json')
    for p in source['parts']:ck('source:'+str((p['path'],p['first_line'])),pinned(p)and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    prof=read(O/'context-sp-profile-frontiers-v130.json');ck('104_312_264_48_88',(prof['frontiers'],prof['draft_SP_collectives'],prof['query_KV_typed_TASKs'],prof['positive_None_metadata_zero_output_TASKs'],prof['retained_main_feature_gathers'])==(104,312,264,48,88))
    rawrefs={r['database_index']:r for r in read(O/'host-launch-original-API-coverage-v126.json')['databases']};bydb=defaultdict(list)
    for r in prof['records']:bydb[r['database_index']].append(r)
    used_comm=set();used_q=set();used_zero=set();used_main=set()
    for ref in prof['databases']:
        di=ref['database_index'];ck('cached_native_json:'+str(di),pinned(ref));raw=rawrefs[di];ck('raw_DBpin:'+str(di),pinned(raw)and raw['sha256']==ref['raw_database_sha256'])
        db=sqlite3.connect(Path(raw['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};cidmap=defaultdict(set)
        for row in db.execute('SELECT *FROM CONNECTION_IDS'):cidmap[row['id']].add(row['connectionId'])
        zerochains={r[0]for r in db.execute("SELECT DISTINCT c.id FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE s.value LIKE '%/attention/dsa_v41.py(698): forward'")}
        for f in bydb[di]:
            C,Q=f['context_rows'],f['query_KV_rows'];ck('3layers:'+str((di,f['source_order_ordinal'])),len(f['layers'])==3 and not f['exact_tensor_request_cache_weight_identity_bound'])
            for j,l in enumerate(f['layers']):
                g=l['draft_SP_allGather'];k=(di,g['COMM_OP_rowid']);assert k not in used_comm;used_comm.add(k)
                rr=db.execute('SELECT *FROM COMMUNICATION_OP WHERE rowid=?',(g['COMM_OP_rowid'],)).fetchone()
                ck('raw_SP:'+str(k),all(rr[x]==g[x]for x in ['count','rankSize','startNs','endNs','connectionId'])and g['source_grade']=='DIRECT_NATIVE_FULLCHAIN'and g['dtype']=='BFP16'and g['count']==5120*l['per_sender_hidden_rows_from_count'])
                c=l['context_cycle'];ck('context_before_repeat_before_SP:'+str(k),c['endNs']<=f['draft_repeat']['startNs']and f['draft_repeat']['endNs']<=g['startNs'])
                if l['query_KV']:
                    q=l['query_KV'];t=q['TASK'];d=q['definition'];key=(di,t['task_record_rowid']);assert key not in used_q;used_q.add(key)
                    rawt=db.execute('SELECT *FROM TASK WHERE rowid=?',(t['task_record_rowid'],)).fetchone()
                    rawd=db.execute('SELECT *FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
                    ck('raw_query_KV:'+str(key),all(rawt[x]==t[x]for x in rawt.keys())and strings[rawd['inputShapes']]==d['inputShapes_text']==f'"{Q},5120;512,5120"'and strings[rawd['inputDataTypes']]==d['inputDataTypes_text']=='DT_BF16;DT_BF16')
                    end=f['layers'][j+1]['draft_SP_allGather']['startNs']if j<2 else f['seed_TASK']['startNs']
                    ck('query_order_window:'+str(key),g['endNs']<=t['startNs']<=t['endNs']<=end and f['source_context']!='CALLER_DUMMY'and l['dummy_zero_output_witness']is None)
                else:
                    z=l['dummy_zero_output_witness'];t=z['TASK'];key=(di,t['task_record_rowid']);assert key not in used_zero;used_zero.add(key)
                    ck('dummy_positive_zero:'+str(key),f['source_context']=='CALLER_DUMMY'and z['definition']['opType_text']=='ZerosLike'and z['definition']['inputShapes_text']=='"20,5120"')
                    for aid in z['API_ids']:
                        a=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(aid,)).fetchone()
                        ck('positive_None_metadata_source:'+str((di,aid)),a['callchainId']in zerochains and t['connectionId']in cidmap[a['connectionId']])
            mg=f['retained_main_feature_allGather']
            if mg:
                k=(di,mg['COMM_OP_rowid']);assert k not in used_main;used_main.add(k)
                rr=db.execute('SELECT *FROM COMMUNICATION_OP WHERE rowid=?',(mg['COMM_OP_rowid'],)).fetchone();m=f['preceding_main_projection']
                ck('retained_raw_main:'+str(k),rr['count']==mg['count']==640*C and rr['rankSize']==8 and m['M']==C and m['GEMM']['endNs']<=mg['startNs']<=mg['endNs']<=m['gather_transpose']['startNs'])
        db.close()
    ck('exact_unique_populations',(len(used_comm),len(used_q),len(used_zero),len(used_main))==(312,264,48,88))
    costs=read(O/'context-sp-hybrid-owner-models-v130.json');total=0;frontcount=0
    for r in costs['regimes']:
        C,Q,mq=r['context_rows'],r['query_KV_rows'],r['query_sender_rows'];main=1280*C;total+=C+1
        threshold=min(C,72*mq);ck('threshold:'+str(C),r['max_distributed_context_rows_without_exceeding_old_query_carrier_content']==threshold)
        frontier=r['model_Pareto_cases'];frontcount+=len(frontier)
        previous_b=-1;previous_w=None
        for f in frontier:
            s=f['distributed_context_rows'];owner=(s+7)//8
            ck('Pareto_rawformula:'+str((C,s)),f==carrier_case(C,Q,mq,s,main_bytes=main)and f['new_KV_carrier_sender_content_bytes_per_layer']==1024*(mq+owner)and f['new_projection_rows_per_card_per_layer']==C-s+owner+mq)
            b=f['new_KV_carrier_sender_content_bytes_per_layer'];w=f['new_projection_rows_per_card_per_layer']
            ck('Pareto_strict:'+str((C,s)),b>previous_b and(previous_w is None or w<previous_w));previous_b=b;previous_w=w
        for s in range(C+1):
            o=(s+7)//8;b=1024*(mq+o);w=C-s+o+mq
            # Analytic bucket representative, with possible equal-work final
            # padding bucket dominated by its predecessor.
            best_s=min(o*8,C);best_o=(best_s+7)//8;best_w=C-best_s+best_o+mq
            assert best_w<=w
            ck('carrier_budget_cut:'+str((C,s)),(b<=10240*mq)==(s<=72*mq))
        for c in r['boundary_cases']:ck('boundary:'+str((C,c['distributed_context_rows'])),c==carrier_case(C,Q,mq,c['distributed_context_rows'],main_bytes=main))
    ck('19637options2460frontier',total==costs['all_s_options_checked']==19637 and frontcount==2460)
    c=carrier_case(24,20,3,24,main_bytes=30720);ck('small_specific',c['baseline_query_gather_sender_content_bytes_per_layer']==30720 and c['new_KV_carrier_sender_content_bytes_per_layer']==6144 and c['baseline_combined_sender_content_bytes']==122880 and c['new_combined_sender_content_bytes']==49152 and c['new_projection_rows_per_card_per_layer']==6)
    c=carrier_case(6400,10,2,6400,main_bytes=8192000);ck('long_specific',c['new_KV_carrier_sender_content_bytes_per_layer']==821248 and c['baseline_query_gather_sender_content_bytes_per_layer']==20480 and c['baseline_combined_sender_content_bytes']==8253440 and c['new_combined_sender_content_bytes']==10655744)
    c=carrier_case(6400,10,2,144,main_bytes=8192000);ck('hybrid_specific',c['new_KV_carrier_sender_content_bytes_per_layer']==20480 and c['new_projection_rows_per_card_per_layer']==6276)
    rec=read(O/'context-sp-recipient-owner-models-v130.json');again=recipient_cases();ck('9330recipientcases',rec['case_count']==len(again)==9330 and rec['cases']==again and not rec['actual_cache_recipient_or_full_reader_sets_attested'])
    for c in rec['cases']:assert c['endpoint_KV_row_deliveries']==len(c['recipient_ranks'])-len(c['compute_ranks'])and c['endpoint_value_content_bytes']==1024*c['endpoint_KV_row_deliveries']
    order=read(O/'context-sp-order-and-sharing-witnesses-v130.json');ck('writerorder_witness',order['writer_order']==json.loads(json.dumps(writer_witness()))and order['no_actual_slot_or_native_race_or_fusion_parity_proved'])
    graph=read(O/'context-sp-joint-graph-and-conditions-v130.json');ck('conditional_local_graph',graph['pair']==['F29','F32']and graph['native_graph_nodes_added']==0 and not graph['new_device_work'])
    ed=defaultdict(set)
    for e in graph['edges']:ed[e['producer']].add(e['consumer'])
    def reach(a,b):
        todo=list(ed[a]);seen=set()
        while todo:
            x=todo.pop()
            if x==b:return True
            if x not in seen:seen.add(x);todo.extend(ed[x])
        return False
    ck('acyclic_and_WAW',not any(reach(n['id'],n['id'])for n in graph['nodes'])and reach('CW','QW')and reach('QH','QRAW')and reach('MAIN','PROJ'))
    err=read(O/'context-sp-query-assumption-error-v130a.json');ck('failed_domain_assumption_retained',err['status']=='PROFILE_SELECTOR_DOMAIN_ASSUMPTION_REJECTED'and 'dummy'in err['reason'])
    if args.preflight:
        old=read(O/'candidate-relation-coverage-v129.json');r=next(r for r in old['pairs']if {r['a'],r['b']}=={'F29','F32'});ck('actually_pending',r['scope_state']!='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL')
        manifest=read(O/'checkpoint-manifest-v129.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v130.json');ck('open_progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v130.json');pairs={tuple(sorted((r['a'],r['b'])))for r in rel['clauses']};ck('137pairs26groups',len(rel['clauses'])==len(pairs)==rel['clause_count']==rel['selected_pair_count']==137 and len(rel['higher_order_groups'])==26)
        r=next(r for r in rel['clauses']if r['id']=='REL130-01');ck('new_relation_gated',{r['a'],r['b']}=={'F29','F32'}and not r['all_runtime_gates_proved']and r['critical_path_gain_us']is None)
        g=next(g for g in rel['higher_order_groups']if g['id']=='HG85-VERIFY-CONTEXT-DRAFT');ck('group_explicitly_includes_SP','F32'in g['members']and not g['all_subsets_analyzed']and not g['legal_joint_fusion_proved'])
        cov=read(O/'candidate-relation-coverage-v130.json');ck('137_26_458',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(137,26,458)and len(cov['pairs'])==595)
        nav=read(O/'candidate-pair-navigation-v130.json');r=next(r for r in nav['pairs']if {r['a'],r['b']}=={'F29','F32'});ck('navigation_reconciled',r['conditional_clause_ids']==['REL130-01']and r['semantic_pair_reconciliation_done_in_stated_scope']and nav['remaining_central_pair_reconciliation']==458)
        m=read(O/'module-phase-branch-matrix-v130.json');ck('358_26',len(m['modules'])==26 and sum(len(x['branches'])for x in m['modules'])==358)
        old=read(O/'operator-analysis-coverage-ledger-v129.json');new=read(O/'operator-analysis-coverage-ledger-v130.json');ck('native_uncertainty_unchanged',all(old[k]==new[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v130.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v130.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    out=dict(status='PASS',checks=len(checks),preflight=args.preflight,source_order_frontiers=104,proposal_KV_TASKs=264,positive_dummy_zero_TASKs=48,retained_main_feature_gathers=88,
        owner_options=19637,recipient_cases=9330,new_scoped_pair=1,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'context-sp-preflight-v130.json',out)
    print(json.dumps(out))


if __name__=='__main__':main()
