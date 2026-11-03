"""Verify raw Query/SWA joins, reused old-latent chains and versioned schedules."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,hashlib,sqlite3,math,statistics
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pin(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for p in read(O/'hc-rope-latent-inputs-v140.json')['inputs']:ck('input:'+p['path'],pin(p))
    for p in read(O/'hc-rope-latent-source-contract-v140.json')['parts']:ck('source:'+str((p['path'],p['first_line'])),pin(p)and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    prof=read(O/'hc-rope-latent-native-frontiers-v140.json');old={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sparse-eager-native-frontiers-v138.json')['records']};index={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-index-query-weight-native-producers-v139.json')['records']};qgroups=defaultdict(list);lgroups=defaultdict(list);seen=set();roles=Counter();geoms=Counter();latentgeoms=Counter();windows=defaultdict(list);ceilings=defaultdict(list);qbefore=positive=0
    prior={}
    for n in ['hc-c1-native-consumers-v135.json','hc-c2-native-consumers-v136.json']:
        for r in read(O/n)['records']:prior[n,r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']]=r
    for r in prof['query_records']:qgroups[r['database_index']].append(r)
    for r in prof['latent_records']:lgroups[r['database_index']].append(r)
    for ref in prof['databases']:
        di=ref['database_index'];ck('rawpin:'+str(di),pin(ref['raw_database']));ck('cachepin:'+str(di),pin(ref['cached_native']))
        db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};frames={(r['id'],r['stackDepth'],r['stack'])for r in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};cids=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cids[x['id']].add(x['connectionId'])
        def stage(role,s,parent):
            a,t,d=s['API'],s['TASK'],s['definition'];tk=di,t['task_record_rowid'];ck('unique:'+str(tk),tk not in seen);seen.add(tk);roles[role]+=1
            aa=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone();tt=db.execute('SELECT *FROM TASK WHERE rowid=?',(tk[1],)).fetchone();dd=db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
            ck('API_TASK_CID:'+str(tk),all(aa[k]==a[k]for k in aa.keys())and all(tt[k]==t[k]for k in tt.keys())and ss[aa['name']]==a['name_text']and t['connectionId']in cids[a['connectionId']]and a['api_rowid']in t['native_linked_API_rowids'])
            ck('COMPUTE:'+str(tk),all((ss[dd[k[:-5]]]if k.endswith('_text')else dd[k])==v for k,v in d.items()))
            fs=s['source_frames'];fk={(f['id'],f['stackDepth'],f['stack'])for f in fs};ck('frames:'+str(tk),bool(fs)and len(fk)==len(fs)and fk<=frames and all(f['id']==a['callchainId']and f['value']==ss[f['stack']]for f in fs))
            ag,rs=parent['SP_allGather']['API'],parent['output_reduceScatter']['API'];ck('scope:'+str(tk),a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(rs['startNs']))
        for r in qgroups[di]:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];p=r['eager_HC_sparse_parent'];ck('parent:'+str(key),p==old[key]and not r['actual_CS_alias_or_new_published_row_native_ABI_or_counterfactual_gain_bound']);parent=p['eager_HC_SP_parent'];s=r['stages']
            for role,x in s.items():stage(role,x,parent)
            m=int(s['QB']['definition']['outputShapes_text'].strip('"').split(',')[0]);g=int(s['SWA_NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[m,g]+=1
            ck('typed_QB_and_Query',s['QB']['definition']['inputDataTypes_text']=='INT8;INT8;DT_BF16;FLOAT'and s['QB']['definition']['outputShapes_text']==f'"{m},32768"'and s['QB']['definition']['outputDataTypes_text']=='DT_BF16'and s['QUERY_ROPE']['definition']['outputShapes_text']==f'"{m},1,64,512"'and s['QUERY_ROPE']['definition']['outputDataTypes_text']=='DT_BF16')
            ck('globalSWA',s['SWA_NORM']['definition']['inputShapes_text']==f'"{g},512;512"'and s['SWA_STORE']['definition']['inputShapes_text']==f'"17306,128,512;{g},2;{g},512"')
            qb,sw,qr=[s[k]['TASK']for k in ['QB','SWA_STORE','QUERY_ROPE']];window=max(0,sw['endNs']-qb['endNs']);positive+=window>0;windows[m,g].append(window);ceilings[m,g].append(min(window,qr['endNs']-qr['startNs']))
            ck('source_join_observed',r['recorded_SWA_store_end_minus_QB_end_ns']==sw['endNs']-qb['endNs']and r['recorded_positive_SWA_wait_window_ns']==window and r['query_RoPE_TASK_starts_after_QB_and_SWA_store']and qr['startNs']>=max(qb['endNs'],sw['endNs']))
            if key in index:
                ck('already_before_index',r['query_RoPE_before_IndexQ_TASK']and qr['endNs']<=index[key]['producers']['INDEX_Q']['TASK']['startNs']);qbefore+=1
            else:ck('no invented index',r['query_RoPE_before_IndexQ_TASK']is None)
        for r in lgroups[di]:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];orig=prior[(r['reused_artifact'],)+key];p=old[key]['eager_HC_SP_parent'];norm='C1_NORM'if r['mode']=='C1'else'POOL_NORM'
            for role,s in r['consumers'].items():
                ck('reuse:'+str((key,role)),s==orig['consumers'][norm if role=='LATENT_NORM'else role]);stage(role,s,p)
            s=r['consumers'];rows=int(s['LATENT_NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]);latentgeoms[r['mode'],rows]+=1
            ck('two_norm_domains',s['LATENT_NORM']['definition']['inputShapes_text']==f'"{rows},512;512"'and s['INDEX_WK']['definition']['inputShapes_text']==f'"{rows},512;128,512"'and s['INDEX_NORM']['definition']['inputShapes_text']==f'"{rows},128;128"')
            ck('key_quant_scale_types',s['INDEX_QUANT']['definition']['outputDataTypes_text']=='INT8;FLOAT'and s['INDEX_SCALE_CAST']['definition']['inputDataTypes_text']=='FLOAT'and s['INDEX_SCALE_CAST']['definition']['outputDataTypes_text']==s['INDEX_SCALE_STORE']['definition']['outputDataTypes_text']=='FLOAT16'and s['LONG_STORE']['definition']['outputDataTypes_text']=='DT_BF16')
            wk,lr,ss0,ks,ls=[s[k]['TASK']for k in ['INDEX_WK','LONG_ROPE','INDEX_SCALE_STORE','INDEX_KEY_STORE','LONG_STORE']]
            ck('current_oldvalue_control',r['IndexWK_end_before_LongRoPE']and wk['endNs']<=lr['startNs']and r['recorded_LongRoPE_start_minus_IndexWK_end_ns']==lr['startNs']-wk['endNs']and r['recorded_LongRoPE_after_both_index_stores']and max(ss0['endNs'],ks['endNs'])<=lr['startNs']and ls['endNs']<=old[key]['stages']['QLI']['TASK']['startNs'])
        db.close()
    ck('5760bindings',len(seen)==5760 and len(prof['query_records'])==960 and len(prof['latent_records'])==96 and all(roles[k]==960 for k in ['QB','QUERY_ROPE','SWA_NORM','SWA_ROPE','SWA_STORE'])and all(roles[k]==96 for k in ['LATENT_NORM','INDEX_WK','INDEX_NORM','INDEX_ROPE','INDEX_QUANT','INDEX_KEY_STORE','INDEX_SCALE_CAST','INDEX_SCALE_STORE','LONG_ROPE','LONG_STORE']))
    ck('880windows',positive==prof['positive_SWA_wait_windows']==880 and qbefore==192 and geoms=={(800,6400):320,(816,6528):320,(830,6668):40,(834,6668):280})
    obs=read(O/'hc-rope-latent-observed-frontiers-v140.json');ck('sourceorder_falsifies_false_gap',obs['Query_RoPE_before_IndexQ_frontiers']==192 and obs['all_query_RoPE_after_QB_and_SWA'])
    for r in obs['query_geometry']:
        key=r['M'],r['global_SWA_rows'];v=windows[key];d=ceilings[key];ck('geometry_intervals',r['frontiers']==len(v)and r['positive_SWA_wait_intervals']==sum(x>0 for x in v)and r['median_ns']==statistics.median(v)and r['minimum_ns']==min(v)and r['maximum_ns']==max(v)and r['fixed_service_join_hiding_reference_median_ns']==statistics.median(d)and r['fixed_service_join_hiding_reference_max_ns']==max(d))
    for i,r in enumerate(obs['fixed_service_join_diagnostics']):
        q,s,d=r['QB_end_ns'],r['SWA_end_ns'],r['RoPE_recorded_duration_ns'];ck('conditional_join_ceiling:'+str(i),max(q,s)+d-max(q+d,s)==r['fixed_service_join_hiding_reference_ceiling_ns']==min(d,max(s-q,0)))
    def query_dag(r,fork,hoist,price=False):
        nd={'y':(r['y'],[]),'c':(r['c'],[]),'u':(r['n'],['y']if fork else['y','c']),'qb':(r['qb'],['u']),'sw':(r['sw']+(r['ds']if price else 0),['u']),'coeff':(r['dc']if price else 0,['c'])}
        if hoist:nd.update(qr=(r['rope']+(r['dr']if price else 0),['qb']),join=(0,['qr','sw']))
        else:nd.update(j=(0,['qb','sw']),join=(r['rope'],['j']))
        nd.update(tail=(r['tail'],['join']),post=(0,['tail','coeff']));memo={}
        def end(k):
            if k not in memo:memo[k]=nd[k][0]+max([end(p)for p in nd[k][1]]or[0])
            return memo[k]
        return end('post')
    tm=read(O/'hc-rope-latent-readiness-DAGs-v140.json');qc=Counter();lc=Counter()
    for i,r in enumerate(tm['query_cases']):
        ck('query_DAG:'+str(i),all(query_dag(r,f,h,p)==r[k]for k,f,h,p in [('baseline',False,False,False),('HC_fork_only',True,False,False),('Query_hoist_only',False,True,False),('joint_ideal',True,True,False),('joint_repriced',True,True,True)]))
        qc['JOINT_FASTER'if r['joint_repriced']<r['baseline']else'JOINT_SLOWER'if r['joint_repriced']>r['baseline']else'TIED']+=1
    def latent_dag(r,mode):
        nd={'l':(r['l'],[]),'qi':(r['qi'],[]),'q':(r['q'],[]),'c':(r['c'],[])}
        if mode=='saved_old_tail_segmented_WK':nd['copy']=(r['extra'],['l']);wkbase='copy'
        else:wkbase='l'
        nd.update(wk=(r['w'],[wkbase]),idx=(r['i'],['wk']))
        if mode in ['baseline','QLI_before_long_join']:nd['long']=(r['g'],['idx'])
        elif mode=='long_after_WK_read':nd['long']=(r['g']+r['extra'],['wk'])
        elif mode=='readonly_long_direct':nd['long']=(r['g']+r['extra'],['l'])
        else:nd['long']=(r['g'],['copy'])
        nd.update(qli=(r['k'],['qi','long']if mode=='baseline'else['qi','idx']),out=(r['out'],['q','long','qli']),post=(0,['c','out']));memo={}
        def end(k):
            if k not in memo:memo[k]=nd[k][0]+max([end(p)for p in nd[k][1]]or[0])
            return memo[k]
        return end('post')
    for i,r in enumerate(tm['latent_cases']):
        ck('latent_DAG:'+str(i),all(latent_dag(r,k)==r[k]for k in ['baseline','QLI_before_long_join','long_after_WK_read','readonly_long_direct','saved_old_tail_segmented_WK']))
        lc['READONLY_FASTER'if r['readonly_long_direct']<r['baseline']else'READONLY_SLOWER'if r['readonly_long_direct']>r['baseline']else'TIED']+=1
    ck('DAG_scopes',tm['query_case_count']==len(tm['query_cases'])==2304 and tm['latent_case_count']==len(tm['latent_cases'])==1536 and dict(qc)==tm['query_classes']and dict(lc)==tm['latent_classes']and qc['JOINT_SLOWER']>0 and lc['READONLY_SLOWER']>0)
    wit=read(O/'hc-rope-latent-version-and-CS-witnesses-v140.json');r=wit['prefix_tail'];oldrows=r['old_rows'];rot=r['full_rotated_rows'];tail=r['tail_only'];prefix=r['D']-r['R']
    ck('segmented_tail_full_prefix',[a[:prefix]+b for a,b in zip(oldrows,tail)]==rot and [sum(a*b for a,b in zip(row,r['IndexWk']))for row in oldrows]==r['old_index']and [sum(a*b for a,b in zip(row,r['IndexWk']))for row in rot]==r['wrong_after_inplace']and rot[0]!=r['wrong_tail_as_contiguous_full_row'])
    ck('saved_old_tail_recovers_index',[a[:prefix]+b[prefix:]for a,b in zip(rot,oldrows)]==oldrows)
    r=wit['source_CS_lifetime'];ck('CS_last_reader',r['old_index_RoPE_read_done']<r['reuse_time']<r['queued_long_RoPE_read']and r['correct_long']!=r['wrong_long'])
    r=wit['query_key_quantizer_distinction'];ck('key_query_scale_round_not_same',round(r['BF16_value']/r['FP32_scale'])==r['quant_with_FP32_scale']==95 and round(r['BF16_value']/r['rounded_FP16_scale'])==r['quant_with_FP16_scale']==96)
    work=read(O/'hc-rope-latent-version-work-and-lifetimes-v140.json');ck('6variant_contracts',len(work['variants'])==6)
    for r in work['query_models']:
        m,g=r['M'],r['global_SWA_rows'];ck('query_interface',r['frontiers']==geoms[m,g]and r['full_BF16_query_output']==65536*m and r['QB_RoPE_tail_interface_reference']==16384*m and r['separate_early_RoPE_interface_elision']==0 and r['no_postQB_RMS512'])
    for r in work['latent_models']:
        rows=r['N'];ck('version_work',r['frontiers']==latentgeoms[r['mode'],rows]and r['full_outofplace_rotated_latent_content']==1024*rows and r['tail_only_rotated_content']==128*rows and r['full_outofplace_unchanged_prefix_extra_value_IO']==1792*rows and r['direct_long_cache_RoPE_tail_interface_elision']==r['saved_old_tail_copy_value_IO']==256*rows)
        ck('key_interface',r['IndexNorm_RoPE_Quant_postnorm_interface_reference']==768*rows and r['IndexQuant_INT8_temporary_interface_reference']==256*rows and r['FloatScale_cast_interface_reference']==8*rows and r['FP16Scale_scatter_interface_reference']==4*rows and r['one_shared_source_CS_pair_value_content']==512*rows and not r['actual_unique_index_or_long_dirty_rows_known'])
    comp=read(O/'hc-rope-latent-composition-v140.json');ck('three_scoped_pairs',comp['new_pairs']==[['F01','F07'],['F01','F11'],['F01','F19']]and comp['new_pair_clauses']==3 and not comp['all_runtime_gates_proved']and not comp['whole_pair_or_all_subsets_closed'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v139.json')
        for pair in comp['new_pairs']:ck('pending:'+str(pair),not any({r['a'],r['b']}==set(pair)for r in rel['clauses']))
        manifest=read(O/'checkpoint-manifest-v139.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v140.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v140.json');ck('153_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==153 and len({tuple(sorted((r['a'],r['b'])))for r in rel['clauses']})==153 and len(rel['higher_order_groups'])==26)
        for c in comp['clauses']:
            x=next(r for r in rel['clauses']if r['id']==c['id']);ck('conditional:'+c['id'],{x['a'],x['b']}=={c['a'],c['b']}and not x['all_runtime_gates_proved']and x['critical_path_gain_us']is None)
        cv=read(O/'candidate-relation-coverage-v140.json');nv=read(O/'candidate-pair-navigation-v140.json');ck('153_442',cv['scoped_conditional_pairs']==153 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==442)
        for gid,member in comp['group_extensions']:
            gg=next(r for r in rel['higher_order_groups']if r['id']==gid);ck('group_open:'+member,member in gg['members']and not gg['all_subsets_analyzed']and not gg['legal_joint_fusion_proved'])
        a=read(O/'operator-analysis-coverage-ledger-v139.json');b=read(O/'operator-analysis-coverage-ledger-v140.json');ck('native_global_gaps',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        m=read(O/'module-phase-branch-matrix-v140.json');ck('358_26',len(m['modules'])==26 and sum(len(x['branches'])for x in m['modules'])==358)
        for r in read(O/'candidate-record-index-v140.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v140.json')
    for p in manifest['artifacts']:ck('manifest:'+p['path'],pin(p))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,query_frontiers=960,query_TASK_bindings=4800,reused_latent_frontiers=96,reused_latent_TASK_bindings=960,positive_SWA_wait_windows=880,query_DAGs=2304,latent_DAGs=1536,new_scoped_pairs=3,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-rope-latent-preflight-v140.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
