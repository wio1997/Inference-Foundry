"""Verify native C1 consumers and conditional HC/C1 packet-cut models."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,hashlib,json,sqlite3,math
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pin(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for p in read(O/'hc-c1-inputs-v135.json')['inputs']:ck('input:'+p['path'],pin(p))
    for p in read(O/'hc-c1-source-and-precision-contract-v135.json')['parts']:ck('source:'+str((p['path'],p['first_line'])),pin(p)and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    prof=read(O/'hc-c1-native-consumers-v135.json');old={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sp-eager-native-frontiers-v134.json')['records']};bydb=defaultdict(list);counts=Counter();geoms=Counter();seen=set()
    for r in prof['records']:bydb[r['database_index']].append(r)
    for ref in prof['databases']:
        di=ref['database_index'];ck('rawpin:'+str(di),pin(ref['raw_database']));ck('cachepin:'+str(di),pin(ref['cached_native']))
        db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};frames={(r['id'],r['stackDepth'],r['stack'])for r in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};cid=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cid[x['id']].add(x['connectionId'])
        for r in bydb[di]:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];parent=r['eager_HC_SP_parent'];ck('parent:'+str(key),key[2]==20 and parent==old[key]and not r['input_parameter_pointer_or_new_owner_numeric_parity_bound'])
            m=parent['SP_allGather']['COMM']['count']//5120;n=int(parent['SWA_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);q=int(parent['QA']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[m,n,q]+=1
            for name,s in r['consumers'].items():
                t,d,a=s['TASK'],s['definition'],s['API'];tk=di,t['task_record_rowid'];assert tk not in seen;seen.add(tk);counts[name]+=1
                tt=db.execute('SELECT *FROM TASK WHERE rowid=?',(tk[1],)).fetchone();dd=db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone();aa=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone()
                ck('TASK_API_CID:'+str(tk),all(tt[k]==t[k]for k in tt.keys())and all(aa[k]==a[k]for k in aa.keys())and strings[aa['name']]==a['name_text']and t['connectionId']in cid[a['connectionId']]and a['api_rowid']in t['native_linked_API_rowids'])
                for k,v in d.items():ck('COMPUTE:'+str((tk,k)),(strings[dd[k[:-5]]]if k.endswith('_text')else dd[k])==v)
                fkeys={(f['id'],f['stackDepth'],f['stack'])for f in s['source_frames']};ck('source_frames:'+str(tk),len(fkeys)==len(s['source_frames'])and bool(fkeys)and fkeys<=frames and all(f['id']==a['callchainId']and strings[f['stack']]==f['value']for f in s['source_frames']))
                ag,rs=parent['SP_allGather']['API'],parent['output_reduceScatter']['API'];ck('same_forward_attention_scope',a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(rs['startNs']))
            c=r['consumers']
            for name in ['SWA_NORM','C1_NORM']:ck('two512norms',c[name]['definition']['inputShapes_text']==f'"{n},512;512"'and c[name]['definition']['outputDataTypes_text']=='DT_BF16;FLOAT')
            ck('index_WK128',c['INDEX_WK']['definition']['inputShapes_text']==f'"{n},512;128,512"'and c['INDEX_WK']['definition']['outputDataTypes_text']=='DT_BF16'and c['INDEX_NORM']['definition']['inputShapes_text']==f'"{n},128;128"')
            ck('quant_scale_and_stores',c['INDEX_QUANT']['definition']['outputDataTypes_text']=='INT8;FLOAT'and c['INDEX_SCALE_CAST']['definition']['outputDataTypes_text']==c['INDEX_SCALE_STORE']['definition']['outputDataTypes_text']=='FLOAT16'and c['INDEX_KEY_STORE']['definition']['outputDataTypes_text']=='INT8'and c['LONG_STORE']['definition']['outputDataTypes_text']=='DT_BF16')
            ck('observed_old_read_before_mutation',c['INDEX_WK']['TASK']['endNs']<=c['LONG_ROPE']['TASK']['startNs']and r['current_native_index_read_end_before_long_RoPE_start'])
        db.close()
    ck('24x12',len(prof['records'])==24 and len(seen)==288 and len(counts)==12 and all(x==24 for x in counts.values()))
    ck('4geometries',geoms=={(800,6400,800):8,(816,6528,816):8,(834,6668,834):7,(834,6668,830):1})
    work=read(O/'hc-c1-delivery-cuts-and-work-v135.json');ck('delivery_cut_sizes',[r['bytes_per_sender_row']for r in work['cuts']]==[2048,2048,2304,2178])
    for r in work['models']:
        m,n,q=r['sender_rows'],r['global_rows'],r['QA_rows'];ck('profile_geometry',r['frontiers']==geoms[m,n,q])
        ck('typed_payload',r['latent_cut_sender_content']==2048*m and r['index_norm_cut_sender_content']==2304*m and r['encoded_endpoint_cut_sender_content']==(2*512+2*512+128+2)*m and r['encoded_extra_vs_latent_content']==130*m and r['index_norm_extra_vs_latent_content']==256*m)
        ck('QA_payload',r['with_QA_latent_content']==4608*m and r['with_QA_index_norm_content']==4864*m and r['with_QA_encoded_endpoint_content']==4738*m)
        ck('dense_value_references',r['current_C1_index_WK_dense_ops']==2*n*512*128 and r['new_owner_index_WK_dense_ops']==2*m*512*128 and r['optional_packed_to_two_contiguous_projection_value_IO']==8*n*512 and r['current_two_projection_input_value_reads']==4*n*5120 and r['joint_input_value_read_reference']==2*n*5120)
        ck('matrix_union',r['two_BF16_projection_weight_value_content']==10485760 and r['HC_FLOAT_coeff_weight_value_content']==1966080 and r['union_HC_two_projection_weight_value_content']==12451840 and r['with_QA_union_weight_value_content']==25559040 and r['with_index_union_matrix_value_content']==12582912 and r['with_QA_and_index_union_matrix_value_content']==25690112 and not r['actual_HBM_wire_allocated_bytes_or_necessary_all_algorithm_work'])
    def finish(r,mode):
        # Construct an independently evaluated weighted precedence graph.
        nodes={'s':(r['s'],[]),'b':(r['b'],[]),'coeff':(r['c'],[]),'query':(3,[])}
        if mode=='ordered_split':nodes.update(gs=(r['g'],['s']),gb=(r['g'],['b','gs']));swa,c1='gs','gb'
        elif mode=='joint':nodes['g']=(r['gp'],['s','b']);swa=c1='g'
        else:
            nodes.update(ow=(r['wo'],['b']),oe=(1,['ow']),g=(r['gp']+r['extra'],['s','oe']),pub=(1,['g']),out=(2,['pub','query']),post=(0,['out','coeff']))
            swa=c1=None
        if swa:
            nodes.update(swapub=(1,[swa]),wk=(r['w'],[c1]),ipub=(r['i'],['wk']),lpub=(r['l'],['wk']),out=(2,['swapub','ipub','lpub','query']),post=(0,['out','coeff']))
        memo={}
        def end(k):
            if k not in memo:memo[k]=nodes[k][0]+max([end(x)for x in nodes[k][1]]or[0])
            return memo[k]
        return end('post')
    tm=read(O/'hc-c1-packet-readiness-models-v135.json');classes=Counter();ck('3456cases',tm['case_count']==len(tm['cases'])==3456)
    for j,r in enumerate(tm['cases']):
        ck('DAGcompletion:'+str(j),all(finish(r,k)==r[k]for k in ['joint','ordered_split','owner_encoded']))
        classes['JOINT_FASTER'if r['joint']<r['ordered_split']else'SPLIT_FASTER'if r['joint']>r['ordered_split']else'TIED']+=1
    ck('either_protocol_can_win',dict(classes)==tm['classes']and classes['SPLIT_FASTER']>0 and classes['JOINT_FASTER']>0)
    wit=read(O/'hc-c1-layout-and-old-value-witnesses-v135.json');p=wit['packed_view'];ck('stride_witness',p['correct_C1_indices']==[2,3,6,7]and set(p['wrong_contiguous_C1_indices'])&set(p['SWA_row1_indices'])=={4,5})
    w=wit['old_latent'];ck('old_value_WAR',sum(a*b for a,b in zip(w['latent'],w['Wk']))==w['old_index_dot']==1 and sum(a*b for a,b in zip(w['long_rotation'],w['Wk']))==w['wrong_after_rotation_dot']==0)
    g=read(O/'hc-c1-joint-graph-and-clauses-v135.json');ck('graph_join',g['pair']==['F01','F33']and g['new_pairs']==1 and ['INDEX_WK_OLD_READ','LONG_ROPE']in g['edges']and ['INDEX_PUBLICATION','LONG_ROPE']not in g['edges']and ['CURRENT_PRE4','NEXT_FFN_PRE']in g['next_FFN_edges']and not g['all_runtime_legal_fusion_or_all_subsets_closed'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v134.json');ck('actually_pending',not any({r['a'],r['b']}=={'F01','F33'}for r in rel['clauses']));manifest=read(O/'checkpoint-manifest-v134.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v135.json');ck('open_progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v135.json');ck('143_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==143 and len({tuple(sorted((r['a'],r['b'])))for r in rel['clauses']})==143 and len(rel['higher_order_groups'])==26)
        c=next(r for r in rel['clauses']if r['id']=='REL135-01');ck('new_pair_conditional',{c['a'],c['b']}=={'F01','F33'}and not c['all_runtime_gates_proved']and c['critical_path_gain_us']is None)
        cov=read(O/'candidate-relation-coverage-v135.json');nav=read(O/'candidate-pair-navigation-v135.json');ck('143_452',cov['scoped_conditional_pairs']==143 and cov['remaining_pair_relations']==nav['remaining_central_pair_reconciliation']==452)
        for gid,member in [('HG93-HC-ATTENTION-FORK-DEADLINE','F33'),('HG74-C1','F01')]:
            gg=next(x for x in rel['higher_order_groups']if x['id']==gid);ck('group_open:'+gid,member in gg['members']and not gg['all_subsets_analyzed']and not gg['legal_joint_fusion_proved'])
        m=read(O/'module-phase-branch-matrix-v135.json');ck('358_26',len(m['modules'])==26 and sum(len(x['branches'])for x in m['modules'])==358)
        a=read(O/'operator-analysis-coverage-ledger-v134.json');b=read(O/'operator-analysis-coverage-ledger-v135.json');ck('native_gaps_retained',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v135.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v135.json')
    for p in manifest['artifacts']:ck('manifest:'+p['path'],pin(p))
    out=dict(status='PASS',checks=len(checks),preflight=args.preflight,C1_frontiers=24,consumer_roles=12,packet_cases=3456,new_scoped_pairs=1,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-c1-preflight-v135.json',out)
    print(json.dumps(out))


if __name__=='__main__':main()
