"""Independently check saved native joins, original pair/state domains and DAGs."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,sqlite3,json,hashlib
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pin(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'hc-c2-inputs-v136.json')['inputs']:ck('input:'+r['path'],pin(r))
    for r in read(O/'hc-c2-source-and-state-contract-v136.json')['parts']:ck('source:'+str((r['path'],r['first_line'])),pin(r)and Path(r['path']).read_text().splitlines()[r['first_line']-1:r['last_line']]==r['source'])
    prof=read(O/'hc-c2-native-consumers-v136.json');old={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sp-eager-native-frontiers-v134.json')['records']};groups=defaultdict(list);seen=set();geoms=Counter();roles=Counter()
    for r in prof['records']:groups[r['database_index']].append(r)
    for ref in prof['databases']:
        di=ref['database_index'];ck('raw:'+str(di),pin(ref['raw_database']));ck('cache:'+str(di),pin(ref['cached_native']))
        db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};frames={(r['id'],r['stackDepth'],r['stack'])for r in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};cids=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cids[x['id']].add(x['connectionId'])
        for r in groups[di]:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];p=r['eager_HC_SP_parent'];ck('parent:'+str(key),p==old[key]and key[2]in [2,8,14]and not r['metadata_values_page_generation_actual_aliases_or_new_owner_numeric_parity_bound'])
            m=p['SP_allGather']['COMM']['count']//5120;n=int(p['C2_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);q=int(p['QA']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[m,n,q]+=1
            for name,s in r['consumers'].items():
                a,t,d=s['API'],s['TASK'],s['definition'];tk=di,t['task_record_rowid'];ck('uniqueTASK:'+str(tk),tk not in seen);seen.add(tk);roles[name]+=1
                aa=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone();tt=db.execute('SELECT *FROM TASK WHERE rowid=?',(tk[1],)).fetchone();dd=db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
                ck('API_TASK_CID:'+str(tk),all(a[k]==aa[k]for k in aa.keys())and all(t[k]==tt[k]for k in tt.keys())and strings[aa['name']]==a['name_text']and t['connectionId']in cids[a['connectionId']]and a['api_rowid']in t['native_linked_API_rowids'])
                ck('compute:'+str(tk),all((strings[dd[k[:-5]]]if k.endswith('_text')else dd[k])==v for k,v in d.items()))
                fs=s['source_frames'];fk={(f['id'],f['stackDepth'],f['stack'])for f in fs};ck('frames:'+str(tk),bool(fs)and len(fk)==len(fs)and fk<=frames and all(f['id']==a['callchainId']and f['value']==strings[f['stack']]for f in fs))
                ag,rs=p['SP_allGather']['API'],p['output_reduceScatter']['API'];ck('forward_scope:'+str(tk),a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(rs['startNs']))
            c=r['consumers'];ck('current_order:'+str(key),r['current_pool_update_norm_TASK_order']and c['POOL']['TASK']['endNs']<=c['RING_UPDATE']['TASK']['startNs']and c['RING_UPDATE']['TASK']['endNs']<=c['POOL_NORM']['TASK']['startNs']and r['current_index_old_read_before_long_write']and c['INDEX_WK']['TASK']['endNs']<=c['LONG_ROPE']['TASK']['startNs'])
            ck('twoFLOAT_projections:'+str(key),all(p[x]['definition']['inputDataTypes_text']=='FLOAT;FLOAT'and p[x]['definition']['outputDataTypes_text']=='FLOAT'for x in ['C2_KV','C2_SCORE']))
            ck('full_token_pool_norm:'+str(key),c['POOL_ZERO']['definition']['inputShapes_text']==f'"{n},512"'and c['POOL']['definition']['outputShapes_text']==f'"{n},512"'and c['POOL']['definition']['outputDataTypes_text']=='DT_BF16'and c['POOL_NORM']['definition']['inputShapes_text']==f'"{n},512;512"'and c['POOL_NORM']['definition']['outputDataTypes_text']=='DT_BF16;FLOAT')
            ck('ring_full_descriptor:'+str(key),c['RING_UPDATE']['definition']['outputShapes_text']=='"17306,32,1024"'and c['RING_UPDATE']['definition']['outputDataTypes_text']=='FLOAT')
            ck('index_types:'+str(key),c['INDEX_QUANT']['definition']['outputDataTypes_text']=='INT8;FLOAT'and c['INDEX_SCALE_CAST']['definition']['outputDataTypes_text']==c['INDEX_SCALE_STORE']['definition']['outputDataTypes_text']=='FLOAT16'and c['INDEX_KEY_STORE']['definition']['outputDataTypes_text']=='INT8')
        db.close()
    ck('72x16',len(prof['records'])==72 and prof['frontier_count']==72 and prof['consumer_roles_per_frontier']==16 and len(seen)==1152 and len(roles)==16 and all(n==72 for n in roles.values()))
    ck('geometry',geoms=={(800,6400,800):24,(816,6528,816):24,(834,6668,834):21,(834,6668,830):3})
    wc=read(O/'hc-c2-pair-owner-and-state-witnesses-v136.json');changed=extra=0
    for i,r in enumerate(wc['cases']):
        s,n,cut,base,valid=[r[k]for k in ['s','n','cut','base','valid']];ends=[t for t in range(s,s+n)if t%2]if valid else[]
        original_old=sum(t-1<s for t in ends);cross=sum(t-1>=s and (t-1-s<cut)!=(t-s<cut)for t in ends)
        rows=[base+t-s for t in ends];extra_writes=(min(cut,32)+min(n-cut,32)-min(n,32))if valid else 0
        ck('pair_case:'+str(i),r['E']==len(ends)==((s+n)//2-s//2 if valid else 0)and r['B']==original_old and r['U']==(min(n,32)if valid else 0)and r['original_completion_rows']==rows and r['cross_owner_current_pair_halos']==r['naive_changed_current_vs_old_pairs']==cross and r['naive_old_reads']==original_old+cross and r['naive_tail_extra_writes']==extra_writes)
        changed+=cross>0;extra+=extra_writes>0
    ck('finite_boundary_and_tail_counterexamples',wc['case_count']==len(wc['cases'])and changed>0 and extra>0)
    r=wc['stale_ring_scalar'];ck('stale_ring_changes_pool',sum(r['current_K'])/2==r['correct_pooled']==2 and (r['stale_ring_even_K']+r['current_K'][1])/2==r['wrong_shard_history_pooled']==6)
    r=wc['endpoint_only_loses_future_state'];ck('noninvertible_current_endpoint',sum(r['current_K_A'])/2==sum(r['current_K_B'])/2==r['current_pooled_both']==2 and (r['current_K_A'][0]+3)/2==r['future_pooled_A']==2 and (r['current_K_B'][0]+3)/2==r['future_pooled_B']==1.5)
    work=read(O/'hc-c2-delivery-cuts-and-work-v136.json')
    for r in work['profile_geometry_models']:
        m,n,q=r['sender_rows'],r['global_rows'],r['QA_rows'];ck('typed_planes',r['frontiers']==geoms[m,n,q]and r['original_hidden_sender_content']==10240*m and r['original_typed_SWA_and_C2_sender_content']==(2*512+2*4*512)*m and r['with_QA_typed_sender_content']==7680*m)
        ck('work_and_content',r['C2_pair_current_dense_ops']==4*n*5120*512 and r['C2_pair_owner_dense_ops']==4*m*5120*512 and r['BF16_to_FLOAT_cast_logical_value_IO']==6*n*5120 and r['union_HC_SWA_C2_weight_contents']==r['C2_FLOAT_weights']+r['SWA_BF16_weights']+r['HC_FLOAT_coeff_weights']==28180480 and r['with_QA_union_weight_contents']==41287680 and not r['metadata_E_B_U_and_actual_wire_HBM_or_new_native_mathmode_bound'])
    ref=work['reference'];e=[0]*8;u=[0]*8;ue=[0]*8;X=0
    for seg in ref['segments']:
        s,n,base=seg['s'],seg['n'],seg['base']
        for t in range(s,s+n):
            row=base+t-s;owner=row//800
            if t%2:
                e[owner]+=1
                if t-1>=s:X+=(row-1)//800!=owner
            if t>=s+n-32:u[owner]+=1;ue[owner]+=t%2==0
    ck('reference_owner_counts',e==ref['per_sender_completed']==[400]*8 and u==ref['per_sender_tail']==[0,0,0,32,0,0,0,32]and ue==ref['per_sender_even_tail']==[0,0,0,16,0,0,0,16]and X==ref['cross_owner_current_pair_halos']==0)
    ck('reference_payloads',ref['E']==3200 and ref['U']==64 and ref['aggregate_C2_value_content']==3538944 and ref['fixed_count_C2_content_per_sender']==540672 and ref['fixed_count_C2_even_only_content_per_sender']==475136 and ref['with_SWA_fixed_per_sender']==1359872 and ref['with_SWA_QA_fixed_per_sender']==3407872 and ref['wrong_mean_tail_cap']==32768<ref['full_tail_cap']==131072 and not ref['actual_profile_metadata_or_even_observer_closure_proved'])
    def weighted(r,variant):
        nodes={'s':(r['s'],[]),'k':(r['k'],[]),'h':(r['h'],[]),'c':(r['c'],[]),'pool':(r['p'],['k','h']),'snapshot':(r['t'],['k']),'packet':(0,['s','pool']),'q':(2,[])}
        if variant=='joint':nodes.update(arr=(r['cj'],['packet','snapshot']));pr=tr='arr'
        else:nodes.update(pa=(r['cp'],['packet']),ta=(r['ct'],['snapshot','pa']));pr,tr='pa','ta'
        nodes.update(pub=(4,[pr]),out=(2,['pub','q']),post=(0,['out','c']),ring=(r['u'],[tr,'pool']),both=(0,['post','ring']))
        memo={}
        def end(k):
            if k not in memo:memo[k]=nodes[k][0]+max([end(d)for d in nodes[k][1]]or[0])
            return memo[k]
        return end('post'),end('ring'),end('both')
    tm=read(O/'hc-c2-current-versus-future-DAG-models-v136.json');classes=Counter()
    for i,r in enumerate(tm['cases']):
        ck('DAG:'+str(i),all(weighted(r,v)==tuple(r[v+'_'+k]for k in ['current','ring','current_and_future'])for v in ['joint','split']))
        tag='JOINT_CURRENT_FASTER'if r['joint_current']<r['split_current']else'SPLIT_CURRENT_FASTER'if r['split_current']<r['joint_current']else'TIED_CURRENT';classes[tag]+=1
    ck('DAG_scope',tm['case_count']==len(tm['cases'])==1536 and dict(classes)==tm['classes']and classes['JOINT_CURRENT_FASTER']>0 and classes['SPLIT_CURRENT_FASTER']>0)
    g=read(O/'hc-c2-joint-graph-and-clauses-v136.json');ck('value_graph',g['pair']==['F01','F17']and ['OVERLAPPING_RING_UPDATE','NORM512_BF16']not in g['required_value_edges']and ['OVERLAPPING_RING_UPDATE','FUTURE_OLD_RING_READS']in g['required_value_edges']and ['CURRENT_PRE4','NEXT_FFN_PRE']in g['next_edges']and not g['native_joint_fusion_or_all_subsets_proved'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v135.json');ck('pending_pair',not any({r['a'],r['b']}=={'F01','F17'}for r in rel['clauses']));manifest=read(O/'checkpoint-manifest-v135.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v136.json');ck('open_progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v136.json');ck('144_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==144 and len({tuple(sorted((r['a'],r['b'])))for r in rel['clauses']})==144 and len(rel['higher_order_groups'])==26)
        r=next(r for r in rel['clauses']if r['id']=='REL136-01');ck('conditional_pair',{r['a'],r['b']}=={'F01','F17'}and not r['all_runtime_gates_proved']and r['critical_path_gain_us']is None)
        cv=read(O/'candidate-relation-coverage-v136.json');nv=read(O/'candidate-pair-navigation-v136.json');ck('144_451',cv['scoped_conditional_pairs']==144 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==451)
        for gid,member in g['groups_to_extend']:
            gg=next(x for x in rel['higher_order_groups']if x['id']==gid);ck('open_group:'+gid,member in gg['members']and not gg['all_subsets_analyzed']and not gg['legal_joint_fusion_proved'])
        a=read(O/'operator-analysis-coverage-ledger-v135.json');b=read(O/'operator-analysis-coverage-ledger-v136.json');ck('native_gaps',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        m=read(O/'module-phase-branch-matrix-v136.json');ck('358_26',len(m['modules'])==26 and sum(len(x['branches'])for x in m['modules'])==358)
        for r in read(O/'candidate-record-index-v136.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v136.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pin(r))
    out=dict(status='PASS',checks=len(checks),preflight=args.preflight,C2_frontiers=72,consumer_roles=16,source_bound_native_consumer_TASKs=1152,pair_cases=wc['case_count'],DAG_cases=1536,new_scoped_pairs=1,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-c2-preflight-v136.json',out)
    print(json.dumps(out))


if __name__=='__main__':main()
