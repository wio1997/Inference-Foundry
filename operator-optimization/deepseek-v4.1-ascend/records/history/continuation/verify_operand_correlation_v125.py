"""Independently verify runtime/group ID joins and retained operand uncertainty."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse,hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(name,v):
        assert v,name
        checks.append(name)
    def pinned(r):
        b=Path(r['path']).read_bytes()
        return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'operand-correlation-inputs-v125.json')['inputs']:ck('input:'+r['path'],pinned(r))
    audit=read(O/'operand-correlation-native-records-v125.json');rows=audit['records'];bydb=defaultdict(list)
    original=read(O/'remaining-operand-evidence-status-v98.json')['execution_bindings']
    ck('1352_exact_original_keys',{(r['database_index'],r['TASK']['task_record_rowid'])for r in rows}=={(r['database_index'],r['task_record_rowid'])for r in original}and len(rows)==1352)
    for r in rows:bydb[r['database_index']].append(r)
    cpu=read(O/'operand-correlation-CPU-stacks-v125.json');bool_bydb=defaultdict(list)
    for r in cpu['BOOL24_zeros_like_calls']:bool_bydb[r['database_index']].append(r)
    for db in audit['databases']:
        ck('database_SHA:'+str(db['database_index']),pinned(db))
        con=sqlite3.connect(Path(db['path']).as_uri()+'?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row;con.execute('PRAGMA temp_store=MEMORY')
        expected=bydb[db['database_index']];ids=sorted({r['TASK']['connectionId']for r in expected})
        placeholder=','.join('?'for _ in ids)
        joined=list(con.execute('SELECT p.rowid AS p_row,c.id AS group_id,c.connectionId AS runtime_id,s.value AS name FROM CONNECTION_IDS c JOIN PYTORCH_API p ON c.id=p.connectionId JOIN STRING_IDS s ON p.name=s.id WHERE c.connectionId IN ('+placeholder+')',ids))
        by_runtime=defaultdict(list)
        for r in joined:by_runtime[r['runtime_id']].append(r)
        for r in expected:
            key=str((db['database_index'],r['TASK']['task_record_rowid']));task=con.execute('SELECT *FROM TASK WHERE rowid=?',(r['TASK']['task_record_rowid'],)).fetchone()
            ck('TASK_identity:'+key,all(task[k]==r['TASK'][k]for k in task.keys()))
            links=by_runtime[task['connectionId']]
            ck('exact_join:'+key,{(x['p_row'],x['group_id'],x['runtime_id'])for x in links}=={(x['pytorch_api_rowid'],x['connectionId'],x['runtime_connection_id'])for x in r['exact_connection_CPU_APIs']})
            if r['operator']=='MemSet':ck('eager_parent_only:'+key,task['modelId']==0xffffffff and len(links)==1 and links[0]['name']=='aclnnConstantPadNd')
            else:ck('captured_no_CPU_link:'+key,task['modelId']!=0xffffffff and not links)
            ck('operands_still_unknown:'+key,r['destination_address']is None and r['native_write_length']is None and r['BOOL_semantic_field_or_all_readers']is None)
        for b in bool_bydb[db['database_index']]:
            root=b['root'];ck('BOOL_root_geometry:'+str(root['pytorch_api_rowid']),root['inputDtypes_text'].split(';',1)[0].strip()=='bool'and root['inputShapes_text'].split(';',1)[0].strip()=='24')
            for link in b['exact_native_links']:
                t=con.execute('SELECT * FROM TASK WHERE rowid=?',(link['TASK']['task_record_rowid'],)).fetchone()
                group=[r[0]for r in con.execute('SELECT connectionId FROM CONNECTION_IDS WHERE id=?',(link['connection_group_id'],))]
                ck('eager_BOOL_link_not_pending:'+str((db['database_index'],t['globalTaskId'])),group==[t['connectionId']]and t['modelId']==0xffffffff and not link['is_one_of_original_pending_tasks']and all(r['TASK']['task_record_rowid']!=link['TASK']['task_record_rowid']for r in expected))
        con.close()
    c=read(O/'operand-correlation-consequences-v125.json')
    ck('actual_capture_eager_split',(c['captured_pending_tasks'],c['eager_pending_tasks'],c['pending_with_exact_CPU_API_link'],c['pending_without_exact_CPU_API_link'])==(80,1272,1272,80))
    ck('16_actual_same_shape_counterexamples',(c['CPU_BOOL24_zeros_like_calls'],c['CPU_BOOL24_exact_linked_tasks'],c['CPU_BOOL24_links_to_original_pending'])==(16,16,0))
    ck('actual_GetParams_device_packet_fact',c['getter_probe_fact']['isHostArgs_counts']=={'0':5}and not c['additional_probe_or_model_capture_authorized'])
    witness=read(O/'operand-correlation-identity-witnesses-v125.json')
    for r in witness['inputs']:ck('witness_input:'+r['path'],pinned(r))
    ck('actual_modelId_collision',witness['cross_process_model_id_collision']['own_probe_model_id']==49 and witness['cross_process_model_id_collision']['original_pending_tasks_with_same_numeric_model_id']>0)
    # Numeric equality in the wrong namespace is insufficient even in a small independent fixture.
    groups={10:[7],7:[99]};py=[dict(group=10,field='correct'),dict(group=7,field='wrong_numeric_match')]
    matches=[r['field']for r in py if 7 in groups[r['group']]]
    ck('group_runtime_namespace_counterexample',matches==['correct']and [r['field']for r in py if r['group']==7]!=matches)
    source=read(O/'operand-correlation-source-and-case-witness-v125.json')['source_part'];ck('source_SHA',pinned(source))
    ck('source_lines',Path(source['path']).read_text().splitlines()[source['first_line']-1:source['last_line']]==source['source'])
    if args.preflight:manifest=read(O/'checkpoint-manifest-v124.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v125.json');ck('progress_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        cov=read(O/'candidate-relation-coverage-v125.json');ck('same_relations',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(135,26,460))
        matrix=read(O/'module-phase-branch-matrix-v125.json');ck('same_semantic_inventory',len(matrix['modules'])==26 and sum(len(m['branches'])for m in matrix['modules'])==358)
        old=read(O/'operator-analysis-coverage-ledger-v124.json');new=read(O/'operator-analysis-coverage-ledger-v125.json')
        ck('all_native_uncertainty_retained',all(old[k]==new[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v125.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('Task_ACTIVE_Loop_EVALUATING',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v125.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,databases=32,pending_executions=1352,
        eager_MemSet_parent_links=1272,captured_BOOL_without_CPU_producer=80,same_shape_eager_BOOL_counterexamples=16,
        native_regime_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'operand-correlation-preflight-v125.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
