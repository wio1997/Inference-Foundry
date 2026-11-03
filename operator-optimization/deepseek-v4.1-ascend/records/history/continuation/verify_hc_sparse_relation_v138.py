"""Independent raw HC/sparse joins, weighted DAGs and partial information checks."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,hashlib,sqlite3,math
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pin(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for p in read(O/'hc-sparse-inputs-v138.json')['inputs']:ck('input:'+p['path'],pin(p))
    for p in read(O/'hc-sparse-source-and-join-contract-v138.json')['parts']:ck('source:'+str((p['path'],p['first_line'])),pin(p)and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    prof=read(O/'hc-sparse-eager-native-frontiers-v138.json');views=read(O/'hc-sparse-shared-topk-views-v138.json');viewgroups=defaultdict(list)
    for r in views['records']:viewgroups[r['database_index']].append(r)
    nativekeys={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in prof['records']}
    parents={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sp-eager-native-frontiers-v134.json')['records']};groups=defaultdict(list);roles=Counter();families=Counter();modes=Counter();seen=set();geoms=Counter();copyapis=0
    for r in prof['records']:groups[r['database_index']].append(r)
    for ref in prof['databases']:
        di=ref['database_index'];ck('rawpin:'+str(di),pin(ref['raw_database']));ck('cachepin:'+str(di),pin(ref['cached_native']))
        db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};cid=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cid[x['id']].add(x['connectionId'])
        frames={(x['id'],x['stackDepth'],x['stack'])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')}
        for r in groups[di]:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];p=r['eager_HC_SP_parent'];block=key[2];ck('parent:'+str(key),p==parents[key]and not r['selected_ID_payload_or_crossrank_episode_or_new_partial_state_ABI_bound']);families[r['family']]+=1;modes[r['index_mode']]+=1
            expected_index=block in [2,8,14,20,24,28,32,36]
            for name,s in r['stages'].items():
                if name=='TOPK_COPY_API':
                    ck('copy_API_count',len(s)==int(expected_index));copyapis+=len(s)
                    for v in s:
                        a=v['API'];aa=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone();ck('copy_API_exact',all(a[k]==aa[k]for k in aa.keys())and ss[aa['name']]=='aclnnInplaceCopy')
                        rawids={x[0]for x in db.execute('SELECT rowid FROM TASK WHERE connectionId IN ('+','.join('?'for _ in cid[a['connectionId']])+')',tuple(cid[a['connectionId']]))}if cid[a['connectionId']]else set()
                        ck('copy_exact_CID_TASK_set',rawids=={t['task_record_rowid']for t in v['exact_linked_TASKs']})
                        ck('copy_source_frames',all((f['id'],f['stackDepth'],f['stack'])in frames and f['id']==a['callchainId']and f['value']==ss[f['stack']]for f in v['source_frames']))
                    continue
                expected=True if name in ['SMLA','INVERSE_ROPE','SEND_LAYOUT']else block==20 if name=='CANDIDATE_COPY'else expected_index
                ck('role_presence:'+str((key,name)),bool(s)==expected)
                if not s:continue
                a,t,d=s['API'],s['TASK'],s['definition'];tk=di,t['task_record_rowid'];ck('uniqueTASK:'+str(tk),tk not in seen);seen.add(tk);roles[name]+=1
                aa=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone();tt=db.execute('SELECT *FROM TASK WHERE rowid=?',(tk[1],)).fetchone();dd=db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
                ck('API_TASK_CID:'+str(tk),all(aa[k]==a[k]for k in aa.keys())and all(tt[k]==t[k]for k in tt.keys())and ss[aa['name']]==a['name_text']and t['connectionId']in cid[a['connectionId']]and a['api_rowid']in t['native_linked_API_rowids'])
                ck('COMPUTE:'+str(tk),all((ss[dd[k[:-5]]]if k.endswith('_text')else dd[k])==v for k,v in d.items()))
                fs=s['source_frames'];fkeys={(f['id'],f['stackDepth'],f['stack'])for f in fs};ck('frames:'+str(tk),bool(fs)and len(fkeys)==len(fs)and fkeys<=frames and all(f['id']==a['callchainId']and f['value']==ss[f['stack']]for f in fs))
                ag,rs=p['SP_allGather']['API'],p['output_reduceScatter']['API'];ck('same_forward:'+str(tk),a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(rs['startNs']))
            st=r['stages'];m=int(st['SMLA']['definition']['outputShapes_text'].strip('"').split(',')[0]);geoms[m]+=1
            ck('SMLA_output_64x512_emptyLSE',st['SMLA']['definition']['outputShapes_text']==f'"{m},64,512;"'and st['SMLA']['definition']['outputDataTypes_text']=='DT_BF16;FLOAT')
            ck('current_output_chain_order',st['SMLA']['TASK']['endNs']<=st['INVERSE_ROPE']['TASK']['startNs']and st['INVERSE_ROPE']['TASK']['endNs']<=st['SEND_LAYOUT']['TASK']['startNs'])
            if expected_index:ck('current_sparse_chain_order',st['QUERY_QUANT']['TASK']['endNs']<=st['QLI']['TASK']['startNs']and st['QLI']['TASK']['endNs']<=st['PREPARE']['TASK']['startNs']and st['PREPARE']['TASK']['endNs']<=st['SMLA']['TASK']['startNs'])
            if block>=2:ck('shared_topk_role',r['source_ordinal_shared_topk_producer']==max(x for x in [2,8,14,20,24,28,32,36]if x<=block))
        for r in viewgroups[di]:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];a=r['shared_view_API'];par=parents[key];nat=nativekeys[key];aa=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone()
            ck('shared_view_API:'+str(key),all(a[k]==aa[k]for k in aa.keys())and ss[aa['name']]==a['name_text']=='aten::slice'and nat['index_mode']=='SHARED_TOPK')
            ck('shared_view_frames:'+str(key),bool(r['source_frames'])and all((f['id'],f['stackDepth'],f['stack'])in frames and f['id']==a['callchainId']and f['value']==ss[f['stack']]and '/attention/context_parallel/dsa_v41_cp.py(206):'in f['value']for f in r['source_frames']))
            ag,rs=par['SP_allGather']['API'],par['output_reduceScatter']['API'];ck('shared_view_scope:'+str(key),a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(nat['stages']['SMLA']['API']['startNs'])<=int(rs['startNs'])and r['SMLA_TASK']==nat['stages']['SMLA']['TASK']and r['source_topk_producer_block']==nat['source_ordinal_shared_topk_producer']and not r['actual_bank_pointer_or_ID_values_or_read_completion_attested'])
        db.close()
    ck('3480TASKs',len(prof['records'])==960 and len(seen)==prof['source_bound_TASKs']==3480 and roles==prof['stage_counts']=={'SMLA':960,'INVERSE_ROPE':960,'SEND_LAYOUT':960,'QLI':192,'PREPARE':192,'QUERY_QUANT':192,'CANDIDATE_COPY':24}and copyapis==prof['exact_TOPK_copy_APIs']==192)
    ck('families',families==prof['families']=={'SWA_ONLY':48,'CSA_C2':432,'CSA_C1':480}and modes==prof['index_modes']=={'NONE':48,'MODE3':72,'MODE1':24,'MODE2':96,'SHARED_TOPK':720})
    ck('querygeometries',geoms=={800:320,816:320,834:280,830:40})
    ck('720shared_view_branches',views['source_positive_view_APIs']==len(views['records'])==720 and views['producer_groups']==len(views['source_topk_groups'])==192 and not views['actual_pointer_payload_epoch_or_native_fusion_proved'])
    for g in views['source_topk_groups']:
        key=g['database_index'],g['forward_API_rowid'],g['producer_block'];src=nativekeys[key];seq=[2,8,14,20,24,28,32,36,40];stop=seq[seq.index(g['producer_block'])+1]
        ck('source_topk_reader_group:'+str(key),g['QLI_TASK']==src['stages']['QLI']['TASK']and g['PREPARE_TASK']==src['stages']['PREPARE']['TASK']and [r['block']for r in g['later_source_positive_readers']]==list(range(g['producer_block']+1,stop)))
        for reader in g['later_source_positive_readers']:
            r=nativekeys[key[:2]+(reader['block'],)];ck('later_SMLA_reader',reader['SMLA_TASK_rowid']==r['stages']['SMLA']['TASK']['task_record_rowid'])
    obs=0
    for f in prof['forward_candidate_publish_frontiers']:
        key=f['database_index'],f['forward_API_rowid'];src=next(r for r in prof['records']if (r['database_index'],r['forward_API_rowid'],r['source_block_ordinal'])==key+(20,))
        ck('candidate_source',f['candidate_publish']==src['stages']['CANDIDATE_COPY']and [r['block']for r in f['consumers']]==[24,28,32,36])
        for c in f['consumers']:
            correct=f['candidate_publish']['TASK']['endNs']<=c['HC']['TASK']['startNs'];ck('publish_observation',correct==c['candidate_publish_TASK_ends_before_consumer_HC_TASK_starts']);obs+=correct
    def weighted(r,fork,split,extra=0,delay=0):
        nodes={'y':(r['y'],[]),'c':(r['c'],[]),'k':(r['k'],[]),'release':(0,['y']if fork else['y','c']),'s':(r['ds'],['release']),'ipath':(r['di'],['release']),'i':(0,['ipath','k']),'cd':(delay,['c'])}
        if split:nodes.update(swa=(r['a'],['s']),cmp=(r['b'],['swa','i']),partial=(extra,['cmp']));tail='partial'
        else:nodes.update(start=(0,['s','i']),all=(r['a']+r['b'],['start']));tail='all'
        nodes.update(cont=(r['o'],[tail]),postready=(0,['cont','cd']));memo={}
        def end(k):
            if k not in memo:memo[k]=nodes[k][0]+max([end(p)for p in nodes[k][1]]or[0])
            return memo[k]
        return end('postready')
    tm=read(O/'hc-sparse-fork-and-join-DAG-models-v138.json');classes=Counter()
    for i,r in enumerate(tm['ideal_cases']):
        vs={k:weighted(r,f,s)for k,f,s in [('baseline',False,False),('HC_fork_only',True,False),('attention_split_only',False,True),('joint',True,True)]}
        ck('ideal_DAG:'+str(i),all(r[k]==v for k,v in vs.items())and r['HC_gain']==r['baseline']-r['HC_fork_only']and r['split_gain']==r['baseline']-r['attention_split_only']and r['joint_gain']==r['baseline']-r['joint'])
        cl='SYNERGY'if r['joint_gain']>r['HC_gain']+r['split_gain']else'OVERLAPPING_CREDIT'if r['joint_gain']<r['HC_gain']+r['split_gain']else'ADDITIVE_IN_THIS_CASE';classes[cl]+=1
    ck('ideal_nonadditivity',len(tm['ideal_cases'])==tm['ideal_case_count']==1296 and dict(classes)==tm['nonadditivity_classes']and classes['SYNERGY']>0 and classes['OVERLAPPING_CREDIT']>0)
    priced=Counter()
    for i,r in enumerate(tm['repriced_cases']):
        end=weighted(r['inputs'],True,True,r['extra_partial_state_launch_merge_service'],r['HC_coefficient_contention_delay']);base=weighted(r['inputs'],False,False)
        tag='JOINT_FASTER'if end<base else'JOINT_SLOWER'if end>base else'TIED';ck('repriced_DAG:'+str(i),end==r['joint']and base==r['baseline']and r['classification']==tag);priced[tag]+=1
    ck('priced_variants_not_guaranteed_gain',tm['repriced_case_count']==len(tm['repriced_cases'])==7776 and dict(priced)==tm['repriced_classes']and priced['JOINT_SLOWER']>0 and priced['JOINT_FASTER']>0)
    wit=read(O/'hc-sparse-partial-state-information-witnesses-v138.json')
    for i,r in enumerate(wit['real_reference_cases']):
        ss=r['scoresA']+r['scoresB']+([0]if r['sink_once']else[]);vv=r['valuesA']+r['valuesB']+([0]if r['sink_once']else[])
        for plane in ['A','B']:
            grouped=defaultdict(set)
            for sid,s,v in zip(r['logical_occurrences'+plane],r['scores'+plane],r['values'+plane]):grouped[sid].add((s,v))
            ck('duplicate_consistency',all(len(g)==1 for g in grouped.values()))
        expected=sum(math.exp(s)*v for s,v in zip(ss,vv))/sum(math.exp(s)for s in ss)if ss else None
        ck('real_merge:'+str(i),(expected is None and r['output']is None)or expected is not None and abs(expected-r['output'])<1e-12)
    ck('real_reference_scope',wit['case_count']==len(wit['real_reference_cases'])==96 and not wit['native_precision_or_new_kernel_proof'])
    h=wit['head_mixing'];alpha=[a/(a+b)for a,b in zip(h['partition_A_mass'],h['partition_B_mass'])]
    ck('head_information_loss',sum(h['case1_A_normalized'])==sum(h['case2_A_normalized'])==h['both_A_projected']==1 and sum(w*a for w,a in zip(alpha,h['case1_A_normalized']))==h['correct_final_projected_case1']==.5 and sum(w*a for w,a in zip(alpha,h['case2_A_normalized']))==h['correct_final_projected_case2']==.75)
    r=wit['sink_once'];ck('sink_once',r['CMP_numerator']/(r['SWA_mass']+r['CMP_mass']+r['sink_mass'])==r['correct']and r['CMP_numerator']/(r['SWA_mass']+r['CMP_mass']+2*r['sink_mass'])==r['wrong_sink_each_partition'])
    r=wit['duplicate_occurrence'];ck('occurrence_multiplicity',sum(r['values'])/3==r['original']and sum(dict(zip(r['logical_occurrences'],r['values'])).values())/2==r['wrong_set_deduplicated']and r['original']!=r['wrong_set_deduplicated'])
    edges=wit['liveness']['wait_edges'];adj=defaultdict(list)
    for a,b in edges:adj[a].append(b)
    def cycle(k,path):
        if k in path:return True
        return any(cycle(v,path|{k})for v in adj[k])
    ck('resource_wait_cycle',any(cycle(k,set())for k in list(adj))and not wit['liveness']['finite_progress_proved'])
    work=read(O/'hc-sparse-publication-and-work-cuts-v138.json');ck('candidate_hoist_observations',len(prof['forward_candidate_publish_frontiers'])==work['candidate_publish_frontiers']==24 and work['source20_to_later_mode2_frontiers']==96 and work['observed_candidate_publish_TASK_ends_before_consumer_HC_TASK_starts']==obs and work['source_topk_groups']==192 and work['later_source_positive_TopK_view_APIs']==720)
    for r in work['models']:
        m=r['M'];ck('state_value_content',r['frontiers']==geoms[m]and r['current_BF16_output_value_content']==m*64*512*2 and r['one_declared_FP32_m_l_a_state_value_content']==m*64*514*4 and r['two_state_value_contents']==r['one_state_private_write_plus_read_value_IO']==m*64*514*8 and r['BF16_normalized_plus_FLOAT_LSE_value_content']==m*64*1028 and r['existing_returned_LSE_value_content']==0 and r['reference_format_not_mandatory_extra_allocation_or_HBM'])
    comp=read(O/'hc-sparse-composition-clauses-v138.json');ck('new_scope',comp['new_pairs']==[['F01','F12'],['F01','F13']]and comp['new_pair_clauses']==2 and not comp['all_runtime_gates_proved']and not comp['whole_pair_or_all_subsets_closed'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v137.json')
        for pair in comp['new_pairs']:ck('actually_pending:'+str(pair),not any({r['a'],r['b']}==set(pair)for r in rel['clauses']))
        manifest=read(O/'checkpoint-manifest-v137.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v138.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v138.json');ck('148_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==148 and len({tuple(sorted((r['a'],r['b'])))for r in rel['clauses']})==148 and len(rel['higher_order_groups'])==26)
        for c in comp['clauses']:
            r=next(r for r in rel['clauses']if r['id']==c['id']);ck('conditional_pair:'+c['id'],{r['a'],r['b']}=={c['a'],c['b']}and not r['all_runtime_gates_proved']and r['critical_path_gain_us']is None)
        cv=read(O/'candidate-relation-coverage-v138.json');nv=read(O/'candidate-pair-navigation-v138.json');ck('148_447',cv['scoped_conditional_pairs']==148 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==447)
        for gid,member in comp['group_extensions']:
            r=next(r for r in rel['higher_order_groups']if r['id']==gid);ck('open_group:'+member,member in r['members']and not r['all_subsets_analyzed']and not r['legal_joint_fusion_proved'])
        a=read(O/'operator-analysis-coverage-ledger-v137.json');b=read(O/'operator-analysis-coverage-ledger-v138.json');ck('native_global_gaps_preserved',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        m=read(O/'module-phase-branch-matrix-v138.json');ck('358_26',len(m['modules'])==26 and sum(len(r['branches'])for r in m['modules'])==358)
        for r in read(O/'candidate-record-index-v138.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v138.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pin(r))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,eager_HC_sparse_frontiers=960,source_bound_TASKs=3480,QLI_prepare_chains=192,candidate_publish=24,source_positive_shared_TopK_views=720,ideal_DAG_cases=1296,repriced_DAG_cases=7776,real_partial_state_cases=96,new_scoped_pairs=2,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-sparse-preflight-v138.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
