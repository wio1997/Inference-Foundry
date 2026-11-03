"""Check first-init native/source scope and conditional owner/cache models."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,hashlib,json,sqlite3,math
from finalize_origin_coverage_v9 import N,O,T,L,read,write
from aux_sp_models_v131 import cycle


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pinned(p):
        b=Path(p['path']).read_bytes();return len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256']
    for p in read(O/'init-sp-inputs-v132.json')['inputs']:ck('input:'+p['path'],pinned(p))
    for p in read(O/'init-sp-source-contract-v132.json')['parts']:ck('source:'+str((p['path'],p['first_line'])),pinned(p) and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    prof=read(O/'init-sp-native-frontiers-v132.json');bydb=defaultdict(list);seen=defaultdict(set);domains=Counter()
    for r in prof['records']:bydb[r['database_index']].append(r)
    inits={(r['database_index'],r['stages'][0]['TASK_rowid']):r for r in read(O/'embedding-initialization-native-chains-v67.json')['draft_initializations']}
    oldfront={(r['database_index'],r['source_order_ordinal']):r for r in read(O/'context-sp-profile-frontiers-v130.json')['records']}
    rawrefs={r['database_index']:r for r in read(O/'host-launch-original-API-coverage-v126.json')['databases']}
    for ref in prof['databases']:
        di=ref['database_index'];ck('cached:'+str(di),pinned(ref));raw=ref['raw_database'];ck('raw:'+str(di),pinned(raw) and raw['sha256']==rawrefs[di]['sha256'])
        db=sqlite3.connect(Path(raw['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};cidmap=defaultdict(set)
        for r in db.execute('SELECT *FROM CONNECTION_IDS'):cidmap[r['id']].add(r['connectionId'])
        def source_frames(api,frames):
            cc={a['callchainId']for a in api if a['callchainId']is not None}
            actual=[dict(x,frame=strings[x['stack']])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS WHERE id IN ('+','.join('?'for _ in cc)+')',list(cc))]if cc else[]
            return sorted(actual,key=lambda x:(x['id'],x['stackDepth'],x['stack']))==sorted(frames,key=lambda x:(x['id'],x['stackDepth'],x['stack']))
        def api_check(a):
            rr=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone()
            ck('API:'+str((di,a['api_rowid'])),all((int(rr[k])==int(a[k])if k in ['startNs','endNs']else rr[k]==a[k])for k in rr.keys()if k in a))
        def native(x,kind):
            t,d=x['TASK'],x['definition'];key=(di,t['task_record_rowid']);assert key not in seen[kind];seen[kind].add(key)
            rr=db.execute('SELECT *FROM TASK WHERE rowid=?',(key[1],)).fetchone();dd=db.execute('SELECT *FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
            ck('TASK:'+str(key),all(rr[k]==t[k]for k in rr.keys()) and all(dd[k]==d[k]for k in dd.keys()) and strings[dd['opType']]==d['opType_text'] and strings[dd['inputShapes']]==d['inputShapes_text'] and strings[dd['outputShapes']]==d['outputShapes_text'])
            for a in x['APIs']:
                api_check(a);ck('CID:'+str((key,a['api_rowid'])),t['connectionId']in cidmap[a['connectionId']] and a['api_rowid']in t['native_linked_API_rowids'])
            ck('callchain:'+str(key),source_frames(x['APIs'],x['frames']))
        def comm(g,kind):
            key=(di,g['COMM_OP_rowid']);assert key not in seen[kind];seen[kind].add(key)
            rr=db.execute('SELECT *FROM COMMUNICATION_OP WHERE rowid=?',(key[1],)).fetchone()
            ck('COMM:'+str(key),all(rr[k]==g[k]for k in ['opId','count','rankSize','connectionId','startNs','endNs']) and strings[rr['groupName']]==g['group_name'] and rr['dataType']==11 and g['dtype']=='BFP16')
        for r in bydb[di]:
            f=oldfront[di,r['source_order_ordinal']];init=r['initialization'];tile=init['stages'][0];g=r['first_SP_allGather'];ar=r['embedding_allReduce'];hc=r['first_HcPre'];norm=r['first_RmsNorm'];n=init['local_rows'];N=init['N'];domains[r['branch']]+=1
            ck('exact_original_frontier:'+str((di,tile['TASK_rowid'])),init==inits[di,tile['TASK_rowid']] and g==f['layers'][0]['draft_SP_allGather'] and tile['TASK_rowid']==f['draft_repeat']['TASK_rowid'] and len(init['stages'])==15 and sum(x['operator']=='MemSet'for x in init['stages'])==3)
            for s in init['stages']:
                t=db.execute('SELECT *FROM TASK WHERE rowid=?',(s['TASK_rowid'],)).fetchone();dd=db.execute('SELECT *FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
                ck('init_stage:'+str((di,s['TASK_rowid'])),all(t[k]==s[k]for k in ['startNs','endNs','streamId']) and strings[dd['opType']]==s['operator'])
            native(hc,'HC');native(norm,'NORM');comm(ar,'AR');comm(g,'SP')
            for a in r['embedding_native_APIs']:
                api_check(a);ck('AR_CID:'+str((di,a['api_rowid'])),ar['connectionId']in cidmap[a['connectionId']])
            ck('AR_source:'+str((di,ar['COMM_OP_rowid'])),source_frames(r['embedding_native_APIs'],r['embedding_native_frames']) and any('/dspark.py(249): forward'in x['frame']for x in r['embedding_native_frames']))
            ck('first_order_geometry:'+str((di,tile['TASK_rowid'])),ar['count']==5120*N and g['count']==5120*n and ar['rankSize']==g['rankSize']==8 and n==(N+7)//8 and r['N_valid_from_Tile']==N and r['A_gather_padded']==8*n and ar['endNs']<=tile['startNs'] and init['stages'][-1]['endNs']<=hc['TASK']['startNs']<=hc['TASK']['endNs']<=norm['TASK']['startNs']<=norm['TASK']['endNs']<=g['startNs'])
            ck('typed_HC_norm:'+str((di,tile['TASK_rowid'])),hc['definition']['opType_text']=='HcPre' and hc['definition']['inputShapes_text']==f'"{n},4,5120;24,20480;3;24;{n},4"' and norm['definition']['opType_text']=='RmsNorm' and norm['definition']['inputShapes_text']==f'"{n},5120;5120"')
            if r['first_query_QA']:
                qa=r['first_query_QA'];native(qa,'QA');ck('positive_QA',qa['definition']['opType_text']=='MatMulV2' and qa['definition']['inputDataTypes_text']=='DT_BF16;DT_BF16' and qa['definition']['outputDataTypes_text']=='DT_BF16' and g['endNs']<=qa['TASK']['startNs']<=qa['TASK']['endNs']<=r['first_attention_window_end'])
            if r['first_global_KV']:
                kv=r['first_global_KV'];native(kv,'KV');ck('global_KV',kv['definition']['inputShapes_text']==f'"{N},5120;512,5120"')
            if r['branch']=='POSITIVE_LOCAL_QUERY_EMPTY_GLOBAL_KV':ck('positive_cacheonly_guard',r['first_query_QA']is None and any('/dsa_v41.py(354): _preprocess'in x['frame']for x in r['first_global_KV']['frames']))
            if r['branch']=='POSITIVE_METADATA_NONE_ZERO':
                z=r['positive_dummy_zero'];t=z['TASK'];dd=z['definition'];rr=db.execute('SELECT *FROM TASK WHERE rowid=?',(t['task_record_rowid'],)).fetchone();ck('positive_dummy_zero',all(rr[k]==t[k]for k in rr.keys()) and dd['opType_text']=='ZerosLike' and dd['inputShapes_text']=='"20,5120"' and r['first_global_KV']is None)
            ck('runtime_identity_not_invented',not r['actual_SP_CP_row_map_tensor_alias_parameter_or_request_identity_bound'])
        db.close()
    ck('104_69_19_16',len(prof['records'])==104 and dict(domains)==prof['branch_counts']=={'POSITIVE_METADATA_NONE_ZERO':16,'POSITIVE_QUERY_QA_AND_GLOBAL_KV':69,'POSITIVE_LOCAL_QUERY_EMPTY_GLOBAL_KV':19} and {k:len(seen[k])for k in ['HC','NORM','AR','SP','QA','KV']}=={'HC':104,'NORM':104,'AR':104,'SP':104,'QA':69,'KV':88})
    copies=read(O/'init-sp-copy-and-mask-models-v132.json');ck('3816',len(copies['cases'])==copies['case_count']==3816);bits=[0,0x8000,0x7f80,0xff80,0x7fc1,0x3f80]
    for r in copies['cases']:
        N,P,rank=r['N'],r['P'],r['rank'];n=(N+P-1)//P;rows=[];m=[]
        for i in range(rank*n,(rank+1)*n):
            e=[bits[(i+j)%6]for j in range(3)]if i<N else [0,0,0];rows.append([e]*4);m.append((False if r['mask']is None else r['mask'][i])if i<N else True)
        ck('bitcopy_mask:'+str(len(checks)),r['A']==n*P and r['local_HC_bits']==rows and r['local_mask']==m)
    routes=read(O/'init-sp-QA-owner-routing-models-v132.json');ck('864',len(routes['cases'])==routes['case_count']==864)
    for r in routes['cases']:
        N,P=r['N'],r['P'];n=(N+P-1)//P;missing=[]
        ck('queries_cover_once',sorted(sum(r['query_rows'],[]))==list(range(N)))
        for p,ids in enumerate(r['query_rows']):
            ck('QA_order',r['ordered_QA_values'][p]==[5*(3*i+1)+7 for i in ids]);missing += [[i//n,p,i]for i in ids if i//n!=p]
        ck('missing_route:'+str(len(checks)),r['missing_owner_recipient_rows']==missing and r['raw_endpoint_value_bytes']==10240*len(missing) and r['QA_endpoint_value_bytes']==2560*len(missing))
    costs=read(O/'init-sp-joint-work-content-v132.json')['geometries']
    for r in costs:
        N,n,A=r['N'],r['n'],r['A'];ck('logical_cost:'+str(N),A==8*n and r['old_embedding_AR_input_and_output_content_each']==10240*N and r['new_embedding_RS_padded_input_content']==10240*A and r['new_embedding_RS_output_content']==10240*n and r['old_first_SP_sender_content']==10240*n and r['new_KV_only_sender_content']==1024*n and r['new_KV_QA_sender_content']==3584*n and r['current_global_KV_dense_rows_across_P']==8*N and r['new_uniform_owner_KV_dense_rows']==A and r['new_uniform_owner_QA_dense_rows']==A and r['current_disjoint_cover_QA_dense_rows']==N and not r['physical_wire_HBM_latency_or_unconditional_necessary_work'])
        ck('joint_padding_cost',r['new_uniform_owner_QA_dense_rows']>r['current_disjoint_cover_QA_dense_rows'] and r['new_embedding_RS_padded_input_content']>r['old_embedding_AR_input_and_output_content_each'])
    cache=read(O/'init-sp-first-layer-token-reuse-v132.json');ck('config',pinned(cache['config']));v=cache['value_content'];ck('table_content',cache['V']==129280 and v['full_replicated_pre_RoPE_KV_bytes']==132382720 and v['full_replicated_KV_and_raw_QA_bytes']==463339520 and v['KV_full_vocabulary_dense_MAC_reference']==129280*5120*512 and not cache['actual_token_duplicates_hits_epochs_native_equivalence_or_benefit_established'])
    w=cache['finite_epoch_witnesses'];ck('cache_counterexamples',w['stale_token_only_key']['old']['KV']!=w['stale_token_only_key']['new']['KV'] and w['padding_is_not_token_zero']['padding_raw_embedding']!=w['padding_is_not_token_zero']['token_zero_embedding'] and w['pre_RoPE_cache_cannot_cache_final_positioned_value']['position_A_rotated']!=w['pre_RoPE_cache_cannot_cache_final_positioned_value']['position_B_rotated'])
    graph=read(O/'init-sp-selected-graph-and-guards-v132.json');ck('graph_coeff_fork',not cycle(graph['edges']) and ['HC_COEFF','Y_BF16']not in graph['edges'] and ['HC_COEFF','HC_POST']in graph['edges'] and ['INIT_VIRTUAL_HC','HC_POST']in graph['edges'] and graph['new_native_nodes']==0)
    ck('literal_onehot_not_unconditional_identity',math.isnan(1*math.inf+0*math.inf+0*math.inf+0*math.inf) and graph['numeric_witness']['checked'])
    if args.preflight:
        old=read(O/'candidate-relation-coverage-v131.json');r=next(r for r in old['pairs']if {r['a'],r['b']}=={'F28','F32'});ck('actually_pending',r['scope_state']!='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL');manifest=read(O/'checkpoint-manifest-v131.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v132.json');ck('open_progress',cp['status']=='PROGRESS' and not cp['overall_complete'] and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v132.json');ck('139_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==139 and len({tuple(sorted((r['a'],r['b'])))for r in rel['clauses']})==139 and len(rel['higher_order_groups'])==26)
        c=next(r for r in rel['clauses']if r['id']=='REL132-01');ck('conditional_only',{c['a'],c['b']}=={'F28','F32'} and not c['all_runtime_gates_proved'] and c['critical_path_gain_us']is None)
        cov=read(O/'candidate-relation-coverage-v132.json');nav=read(O/'candidate-pair-navigation-v132.json');ck('139_456',cov['scoped_conditional_pairs']==139 and cov['remaining_pair_relations']==nav['remaining_central_pair_reconciliation']==456)
        r=next(r for r in nav['pairs']if {r['a'],r['b']}=={'F28','F32'});ck('navigation',r['conditional_clause_ids']==['REL132-01'] and r['semantic_pair_reconciliation_done_in_stated_scope'])
        m=read(O/'module-phase-branch-matrix-v132.json');ck('358_26',len(m['modules'])==26 and sum(len(r['branches'])for r in m['modules'])==358)
        a=read(O/'operator-analysis-coverage-ledger-v131.json');b=read(O/'operator-analysis-coverage-ledger-v132.json');ck('uncertainty_retained',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        g10=next(r for r in read(O/'coverage-current-obligations-v132.json')['obligations']if r['id']=='G10');ck('G10_stale_counts_corrected','139'in g10['remaining'] and '456'in g10['remaining'] and '135 conditional'not in g10['remaining'])
        for r in read(O/'candidate-record-index-v132.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE' and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v132.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    out=dict(status='PASS',checks=len(checks),preflight=args.preflight,first_init_frontiers=104,positive_QA=69,positive_cache_only=19,positive_dummy=16,copy_models=3816,owner_route_models=864,new_scoped_pair=1,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'init-sp-preflight-v132.json',out)
    print(json.dumps(out))


if __name__=='__main__':main()
