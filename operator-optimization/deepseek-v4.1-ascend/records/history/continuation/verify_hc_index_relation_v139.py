"""Independent source/CID, ordered-stream, FP16 and HC join checks."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,hashlib,sqlite3,math,struct
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pin(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for p in read(O/'hc-index-inputs-v139.json')['inputs']:ck('input:'+p['path'],pin(p))
    for p in read(O/'hc-index-typed-input-and-round-contract-v139.json')['parts']:ck('source:'+str((p['path'],p['first_line'])),pin(p)and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    prof=read(O/'hc-index-query-weight-native-producers-v139.json');parents={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sparse-eager-native-frontiers-v138.json')['records']};groups=defaultdict(list);seen=set();roles=Counter();stats=Counter();geoms=Counter()
    for r in prof['records']:groups[r['database_index']].append(r)
    for ref in prof['databases']:
        di=ref['database_index'];ck('rawpin:'+str(di),pin(ref['raw_database']));ck('cachepin:'+str(di),pin(ref['cached_native']))
        db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};frames={(r['id'],r['stackDepth'],r['stack'])for r in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};cids=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cids[x['id']].add(x['connectionId'])
        for r in groups[di]:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];parent=r['eager_HC_sparse_parent'];ck('parent:'+str(key),parent==parents[key]and bool(parent['stages']['QLI'])and not r['actual_QR_quantizer_value_identity_parameter_rows_or_new_HC_schedule_bound']);p=parent['eager_HC_SP_parent']
            for name,s in r['producers'].items():
                a,t,d=s['API'],s['TASK'],s['definition'];tk=di,t['task_record_rowid'];ck('unique:'+str(tk),tk not in seen);seen.add(tk);roles[name]+=1
                aa=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone();tt=db.execute('SELECT *FROM TASK WHERE rowid=?',(tk[1],)).fetchone();dd=db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
                ck('API_TASK_CID:'+str(tk),all(aa[k]==a[k]for k in aa.keys())and all(tt[k]==t[k]for k in tt.keys())and ss[aa['name']]==a['name_text']and t['connectionId']in cids[a['connectionId']]and a['api_rowid']in t['native_linked_API_rowids'])
                ck('COMPUTE:'+str(tk),all((ss[dd[k[:-5]]]if k.endswith('_text')else dd[k])==v for k,v in d.items()))
                fs=s['source_frames'];fk={(f['id'],f['stackDepth'],f['stack'])for f in fs};ck('frames:'+str(tk),bool(fs)and len(fs)==len(fk)and fk<=frames and all(f['id']==a['callchainId']and f['value']==ss[f['stack']]for f in fs))
                ag,rs=p['SP_allGather']['API'],p['output_reduceScatter']['API'];ck('scope:'+str(tk),a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(rs['startNs']))
            s=r['producers'];q=int(s['WEIGHT']['definition']['outputShapes_text'].strip('"').split(',')[0]);m=p['SP_allGather']['COMM']['count']//5120;g=int(p['SWA_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[m,q,g]+=1
            ck('QR_and_two_activation_quants',s['QR_NORM']['definition']['inputShapes_text']==f'"{q},1280;1280"'and all(s[n]['definition']['inputShapes_text']==f'"{q},1280"'and s[n]['definition']['inputDataTypes_text']=='DT_BF16'and s[n]['definition']['outputDataTypes_text']=='INT8;FLOAT'for n in ['QB_INPUT_QUANT','INDEX_INPUT_QUANT']))
            ck('typed_distinct_projection_inputs',s['INDEX_Q']['definition']['inputDataTypes_text']=='INT8;INT8;DT_BF16;FLOAT'and s['INDEX_Q']['definition']['outputShapes_text']==f'"{q},4096"'and s['WEIGHT']['definition']['inputShapes_text']==f'"{q},5120;32,5120"'and s['WEIGHT']['definition']['inputDataTypes_text']=='DT_BF16;DT_BF16')
            ck('weight_cast_chain',s['WEIGHT']['definition']['outputDataTypes_text']=='DT_BF16'and s['WEIGHT_FLOAT']['definition']['outputDataTypes_text']==s['WEIGHT_SCALE']['definition']['outputDataTypes_text']=='FLOAT'and s['WEIGHT_FP16']['definition']['inputDataTypes_text']=='FLOAT'and s['WEIGHT_FP16']['definition']['outputDataTypes_text']=='FLOAT16')
            def overlap(a,b):return max(0,min(a['endNs'],b['endNs'])-max(a['startNs'],b['startNs']))
            w=s['WEIGHT']['TASK'];iq=s['INDEX_Q']['TASK'];quant=parent['stages']['QUERY_QUANT']['TASK'];ob=r['observed']
            ck('observed_order_stream_overlap',ob['IndexQ_TASK_end_before_Weight_TASK_start']==(iq['endNs']<=w['startNs'])and ob['Weight_and_IndexQuant_same_stream']==(w['streamId']==quant['streamId'])and ob['Weight_RoPE_TASK_interval_overlap_ns']==overlap(w,s['INDEX_ROPE']['TASK']))
            stats['query_before_weight']+=ob['IndexQ_TASK_end_before_Weight_TASK_start'];stats['weight_quant_same_aux']+=ob['Weight_and_IndexQuant_same_stream'];stats['weight_rope_interval_overlap']+=ob['Weight_RoPE_TASK_interval_overlap_ns']>0
            if p['C2_KV']:
                ck('C2_quant_interval',ob['IndexQuant_C2KV_TASK_interval_overlap_ns']==overlap(quant,p['C2_KV']['TASK']));stats['C2_frontiers']+=1;stats['quant_C2KV_interval_overlap']+=ob['IndexQuant_C2KV_TASK_interval_overlap_ns']>0
            else:ck('no invented_C2_interval',ob['IndexQuant_C2KV_TASK_interval_overlap_ns']is None)
        db.close()
    ck('192x9',len(prof['records'])==prof['frontiers']==192 and len(seen)==prof['source_bound_producer_TASKs']==1728 and len(roles)==9 and all(n==192 for n in roles.values()))
    ck('query_sender_geometries',geoms=={(800,800,6400):64,(816,816,6528):64,(834,834,6668):56,(834,830,6668):8})
    cov=read(O/'hc-index-profile-coverage-and-streams-v139.json');old=read(O/'index-projection-native-frontiers-v106.json');oldkeys={(r['database_index'],r['stages']['IndexQ']['task_record_rowid'])for r in old['frontiers']};newkeys={(r['database_index'],r['producers']['INDEX_Q']['TASK']['task_record_rowid'])for r in prof['records']}
    ck('832known_frontiers',len(oldkeys)==640 and len(newkeys)==192 and not oldkeys&newkeys and len(oldkeys|newkeys)==cov['combined_known_saved_index_frontiers']==832 and cov['legacy_same_aux_stream']==640 and cov['eager_observations']==dict(stats)=={'query_before_weight':192,'weight_quant_same_aux':192,'weight_rope_interval_overlap':176,'C2_frontiers':72,'quant_C2KV_interval_overlap':66})
    def weighted(r,variant):
        fork=variant!='baseline';early=variant not in ['baseline','HC_fork_only'];late=variant in ['early_weight_late_join','compact_product_in_aux'];compact=variant=='compact_product_in_aux'
        nd={'y':(r['y'],[]),'c':(r['c'],[]),'u':(r['n'],['y']if fork else['y','c']),'q':(r['q'],['u']),'w':(r['w'],['u']if early else['q']),'rope':(r['r'],['q']),
            'join':(0,['rope']if late else['rope','w']),'cast':(r['t'],['join']),'other':(r['d'],['cast']),'quant':(r['v'],['cast','w']),'cd':(r['cd']if early else 0,['c'])}
        if compact:nd.update(wcast=(r['f'],['quant']),alpha=(r['p'],['wcast']),indexready=(0,['alpha','other']))
        else:nd['indexready']=(r['f'],['other','quant','w'])
        nd.update(cont=(r['o'],['indexready']),postready=(0,['cont','cd']));memo={}
        def end(k):
            if k not in memo:memo[k]=nd[k][0]+max([end(p)for p in nd[k][1]]or[0])
            return memo[k]
        return end('postready')
    tm=read(O/'hc-index-aux-stream-and-coeff-DAGs-v139.json');cls=Counter()
    for i,r in enumerate(tm['cases']):
        ck('DAG:'+str(i),all(weighted(r,k)==r[k]for k in ['baseline','HC_fork_only','early_weight','early_weight_late_join','compact_product_in_aux']))
        tag='COMPACT_FASTER'if r['compact_product_in_aux']<r['early_weight_late_join']else'COMPACT_SLOWER'if r['compact_product_in_aux']>r['early_weight_late_join']else'TIED';cls[tag]+=1
    ck('12288joint_cases',tm['case_count']==len(tm['cases'])==12288 and dict(cls)==tm['compact_vs_separate_FP16_classes']and cls['COMPACT_FASTER']>0 and cls['COMPACT_SLOWER']>0)
    def half(x):
        try:return struct.unpack('>e',struct.pack('>e',float(x)))[0]
        except OverflowError:return math.copysign(math.inf,x)
    wit=read(O/'hc-index-scalar-round-and-cycle-witnesses-v139.json');r=wit['intermediate_FP16'];w=r['BF16_projection']*r['weight_scale'];qs=r['FP16_query_scale']
    ck('intermediate_FP16_not_identity',half(w)==r['required_first_FP16_weight']and half(half(w)*qs)==r['required_FP16_product']and half(w*qs)==r['product_if_first_FP16_round_dropped']and r['required_FP16_product']!=r['product_if_first_FP16_round_dropped'])
    ck('finite_unit_product_reference',math.isfinite(r['required_FP16_product'])and half(r['required_FP16_product']*half(1))==r['required_FP16_product'])
    r=wit['FP16_too_early'];ck('cast_after_scale',half(r['BF16_value']/64)==r['required_scaled_FP16']and math.isinf(half(r['BF16_value'])))
    r=wit['RoPE_amax'];ck('amax_after_RoPE',r['post_scale']<r['pre_scale']and round(r['nonrot_value']/r['pre_scale'])==r['wrong_pre_RoPE_quant']and round(r['nonrot_value']/r['post_scale'])==r['correct_post_RoPE_quant']and r['wrong_pre_RoPE_quant']!=r['correct_post_RoPE_quant'])
    r=wit['partial_head'];ck('complete_head_needed',r['wrong_early_quant']==127 and r['correct_after_full_head_quant']==1)
    adj=defaultdict(list)
    for a,b in wit['fused_alpha_wait_cycle']['edges']:adj[a].append(b)
    def cycle(k,path):return True if k in path else any(cycle(v,path|{k})for v in adj[k])
    ck('changed_first_reader_cycle',any(cycle(k,set())for k in list(adj))and len(wit['fused_alpha_wait_cycle']['acyclic_repairs'])==3)
    work=read(O/'hc-index-product-delivery-and-work-v139.json');ck('4cuts_4geometries',len(work['variants'])==4 and len(work['models'])==4)
    ck('additional_existing_QLI_variant',work['existing_QLI_unit_scale_variant']['id']=='COMPACT_ALPHA_WITH_PERSISTENT_UNIT_SCALE')
    for r in work['models']:
        m,q,g=r['sender_rows'],r['query_rows'],r['global_KV_rows'];ck('scalar_cut_contents',r['frontiers']==geoms[m,q,g]and r['Weight_BF16_or_scaled_FP16_query_value_content']==r['query_head_scale_FP16_content']==r['compact_alpha_FP16_content']==64*q and r['Weight_fixed_sender_value_content']==64*m and r['both_old_scalar_planes_private_store_read_value_IO']==256*q and r['compact_alpha_private_store_read_value_IO']==r['additional_alpha_IO_if_old_planes_kept']==128*q and r['QLI_declared_M16_product_broadcast_store_read_value_IO']==2048*q)
        ck('unit_plane_cost',r['persistent_unit_scale_read_value_content']==r['persistent_unit_plane_at_this_query_extent_content']==64*q and r['compact_alpha_plus_unit_plane_scalar_value_IO']==r['compact_alpha_private_store_read_value_IO']+r['persistent_unit_scale_read_value_content']==192*q)
        ck('query_and_panel_contents',r['query_INT8_value_content']==4096*q and r['query_INT8_plus_scale_content']==4160*q and r['query_BF16_4096_content']==8192*q and r['QA_Weight_joint_input_one_read_reference']==10240*q and r['separate_two_input_reads_reference']==20480*q and r['optional_packedN1312_to_two_contiguous_outputs_value_IO']==5248*q)
        ck('matrix_plane_contents',r['QA_Weight_matrix_value_contents']==r['QA_BF16_weights']+r['Weight_BF16_weights']==13434880 and r['all_named_matrix_planes_value_content']==r['QA_Weight_matrix_value_contents']+r['HC_FLOAT_coeff_matrix_value_contents']+r['IndexQ_INT8_matrix_value_content']+r['IndexQ_BF16_dequant_plane_content']==20652032 and r['no_actual_HBM_wire_allocation_necessary_all_algorithm_or_native_tiling_proof'])
    comp=read(O/'hc-index-composition-clauses-v139.json');ck('two_scoped_pairs',comp['new_pairs']==[['F01','F09'],['F01','F10']]and comp['new_pair_clauses']==2 and not comp['all_runtime_gates_proved']and not comp['whole_pair_or_all_subsets_closed'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v138.json')
        for pair in comp['new_pairs']:ck('actually_pending:'+str(pair),not any({r['a'],r['b']}==set(pair)for r in rel['clauses']))
        manifest=read(O/'checkpoint-manifest-v138.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v139.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v139.json');ck('150_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==150 and len({tuple(sorted((r['a'],r['b'])))for r in rel['clauses']})==150 and len(rel['higher_order_groups'])==26)
        for c in comp['clauses']:
            x=next(r for r in rel['clauses']if r['id']==c['id']);ck('conditional_pair:'+c['id'],{x['a'],x['b']}=={c['a'],c['b']}and not x['all_runtime_gates_proved']and x['critical_path_gain_us']is None)
        cv=read(O/'candidate-relation-coverage-v139.json');nv=read(O/'candidate-pair-navigation-v139.json');ck('150_445',cv['scoped_conditional_pairs']==150 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==445)
        for gid,member in comp['group_extensions']:
            gg=next(r for r in rel['higher_order_groups']if r['id']==gid);ck('open_group:'+member,member in gg['members']and not gg['all_subsets_analyzed']and not gg['legal_joint_fusion_proved'])
        a=read(O/'operator-analysis-coverage-ledger-v138.json');b=read(O/'operator-analysis-coverage-ledger-v139.json');ck('native_global_gaps',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        m=read(O/'module-phase-branch-matrix-v139.json');ck('358_26',len(m['modules'])==26 and sum(len(x['branches'])for x in m['modules'])==358)
        for r in read(O/'candidate-record-index-v139.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v139.json')
    for p in manifest['artifacts']:ck('manifest:'+p['path'],pin(p))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,new_eager_frontiers=192,new_source_bound_producer_TASKs=1728,combined_known_saved_index_frontiers=832,joint_DAG_cases=12288,new_scoped_pairs=2,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-index-preflight-v139.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
