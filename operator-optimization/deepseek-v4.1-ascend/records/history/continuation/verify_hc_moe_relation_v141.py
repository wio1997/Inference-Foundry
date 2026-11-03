"""Read-only independent raw-CID/epoch and mathematical contract validation."""
from pathlib import Path
from collections import Counter,defaultdict
from bisect import bisect_right
import argparse,json,hashlib,sqlite3,struct
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for x in read(O/'hc-moe-inputs-v141.json')['inputs']:ck('input:'+x['path'],pinned(x))
    for x in read(O/'hc-moe-source-contract-v141.json')['parts']:ck('source:'+str((x['path'],x['first_line'])),pinned(x)and Path(x['path']).read_text().splitlines()[x['first_line']-1:x['last_line']]==x['source'])
    prof=read(O/'hc-moe-native-frontiers-v141.json');parents={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sp-eager-native-frontiers-v134.json')['records']};groups=defaultdict(list);bridges=defaultdict(list);seen=set();roles=Counter();api_only=[];geoms=Counter();guard_count=Counter();native_bridge_count=0
    for r in prof['records']:groups[r['database_index']].append(r)
    for g in prof['bridges']:bridges[g['database_index']].append(g)
    for ref in prof['databases']:
        di=ref['database_index'];ck('dbpin:'+str(di),pinned(ref['raw_database']));ck('cachepin:'+str(di),pinned(ref['cached_native']))
        db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};frames={(x['id'],x['stackDepth'],x['stack'])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};cids=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cids[x['id']].add(x['connectionId'])
        rawapis={x['api_rowid']:x for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')};rawtasks={x['task_record_rowid']:x for x in db.execute('SELECT rowid AS task_record_rowid,*FROM TASK')};rawdefs={x['globalTaskId']:x for x in db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO')}
        def validate(s,tag,unique=False):
            a,t,d=s['API'],s['TASK'],s['definition'];aa=rawapis[a['api_rowid']]
            ck('API:'+tag,all(a[k]==aa[k]for k in aa.keys())and a['name_text']==ss[aa['name']])
            if 'source_frames'in s:
                fs=s['source_frames'];ck('frames:'+tag,bool(fs)and all((f['id'],f['stackDepth'],f['stack'])in frames and f['id']==a['callchainId']and f['value']==ss[f['stack']]for f in fs))
            if t is None:
                ck('APIonly:'+tag,d is None and s['evidence_grade']=='SOURCE_API_ONLY_NO_LINKED_NATIVE_TASK'and not s['native_TASK_or_Compute_inferred_by_time']and not any(x['connectionId']in cids[a['connectionId']]for x in rawtasks.values()))
                return
            tt=rawtasks[t['task_record_rowid']];dd=rawdefs[t['globalTaskId']]
            ck('TASK_CID:'+tag,all(t[k]==tt[k]for k in tt.keys())and a['api_rowid']in t['native_linked_API_rowids']and t['connectionId']in cids[a['connectionId']])
            ck('Compute:'+tag,all((ss[dd[k[:-5]]]if k.endswith('_text')else dd[k])==v for k,v in d.items()))
            if unique:
                key=di,t['task_record_rowid'];ck('unique:'+tag,key not in seen);seen.add(key)
        for r in groups[di]:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];p=parents[key];s=r['stages'];ck('HCparent:'+str(key),r['HC_parent']==p and not r['actual_router_IDs_counts_or_gate_params_or_MC2_internal_services_bound'])
            a0,a1=p['stages']['FFN_PRE']['API'],p['stages']['FFN_POST']['API']
            for role,x in s.items():
                tag=str((key,role));validate(x,tag,True);roles[role]+=x['TASK']is not None
                a=x['API'];ck('FFNscope:'+tag,a['globalTid']==a0['globalTid']and int(a0['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(a1['startNs']))
                if x['TASK']is None:api_only.append(dict(role=role,database_index=di,forward_API_rowid=key[1],source_block_ordinal=key[2],API_rowid=a['api_rowid']))
            m=int(s['FFN_NORM']['definition']['outputShapes_text'].strip('"').split(',')[0]);pp=int(s['TOPK']['definition']['outputShapes_text'].strip('"').split(',')[0]);geoms[m,pp]+=1
            ck('NormCasttypes',s['FFN_NORM']['definition']['inputShapes_text']==f'"{m},5120;5120"'and s['FFN_NORM']['definition']['outputShapes_text']==f'"{m},5120;{m},5120"'and s['FFN_NORM']['definition']['outputDataTypes_text']=='DT_BF16;FLOAT')
            ck('RouterFLOAT',s['ROUTER']['definition']['inputShapes_text']==f'"{m},5120;384,5120"'and s['ROUTER']['definition']['inputDataTypes_text']=='FLOAT;FLOAT'and s['ROUTER']['definition']['outputDataTypes_text']=='FLOAT')
            ck('TopKMC2geometry',s['TOPK']['definition']['outputShapes_text']==f'"{pp},6;{pp},6;{pp},384"'and s['MC2']['definition']['inputShapes_text']==f'"{pp},5120;24,5120,4608;24,2304,5120;{pp},6;110592;122880;{pp},6;{pp}"'and s['MC2']['definition']['inputDataTypes_text']=='DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT;BOOL')
            ck('SharedQuantActDown',s['SHARED_QUANT']['definition']['outputDataTypes_text']=='INT8;FLOAT'and s['SHARED_ACT']['definition']['inputDataTypes_text'].startswith('INT32;')and s['SHARED_ACT']['definition']['outputShapes_text']==f'"{m},2304;{m}"'and s['SHARED_DOWN']['definition']['outputDataTypes_text']=='DT_BF16')
            if s['SHARED_UP']['definition']:ck('SharedUpINT32',s['SHARED_UP']['definition']['outputDataTypes_text']=='INT32')
            if s['MERGE']['definition']:ck('MergeBF16',s['MERGE']['definition']['inputShapes_text']==f'"{m},5120;{m},5120;"'and s['MERGE']['definition']['outputDataTypes_text']=='DT_BF16')
            q,mc=s['SHARED_QUANT']['TASK'],s['MC2']['TASK'];ck('recorded_no_shared_overlap',r['current_shared_after_MC2_same_stream']is True and q['streamId']==mc['streamId']and q['startNs']>=mc['endNs'])
        # Establish the exact per-thread intervening TASK set from raw API/CID ranges, not nearest timestamps.
        gaps=defaultdict(list)
        for g in bridges[di]:gaps[g['CPU_gap']['globalTid']].append(g)
        for gg in gaps.values():gg.sort(key=lambda x:int(x['CPU_gap']['startNs']))
        starts={tid:[int(g['CPU_gap']['startNs'])for g in gg]for tid,gg in gaps.items()};expected=defaultdict(set);bycid=defaultdict(list)
        for t in rawtasks.values():bycid[t['connectionId']].append(t)
        for aid,a in rawapis.items():
            if a['globalTid']not in gaps:continue
            j=bisect_right(starts[a['globalTid']],int(a['startNs']))-1
            if j<0:continue
            g=gaps[a['globalTid']][j];lo,hi=int(g['CPU_gap']['startNs']),int(g['CPU_gap']['endNs'])
            if not lo<=int(a['startNs'])<=int(a['endNs'])<=hi:continue
            for cid in cids[a['connectionId']]:
                for t in bycid[cid]:
                    if t['globalTaskId']in rawdefs:expected[g['forward_API_rowid'],g['from_block']].add(t['task_record_rowid'])
        for g in bridges[di]:
            prev=parents[di,g['forward_API_rowid'],g['from_block']];nxt=parents[di,g['forward_API_rowid'],g['to_block']]
            ck('bridgeparents',g['to_block']==g['from_block']+1 and g['previous_FFN_Post']==prev['stages']['FFN_POST']==nxt['previous_FFN_Post']and g['previous_FFN_Pre']==prev['stages']['FFN_PRE']==nxt['previous_FFN_Pre']and g['next_HC']==nxt['stages']['HC'])
            ck('bridgeguards',g['next_Engram_positive_APIs']==nxt['Engram_positive_APIs']and g['next_aux_mean']==nxt['aux_mean']and g['next_aux_allGather']==nxt['aux_allGather']and not g['all_readers_or_mutation_alias_closed'])
            ck('bridgecompleteCIDset',set(x['TASK']['task_record_rowid']for x in g['source_linked_intervening_TASKs'])==expected[g['forward_API_rowid'],g['from_block']])
            for j,x in enumerate(g['source_linked_intervening_TASKs']):validate(x,'bridge:'+str((di,g['forward_API_rowid'],g['from_block'],j)))
            guard_count['engram']+=bool(g['next_Engram_positive_APIs']);guard_count['aux']+=bool(g['next_aux_mean']);guard_count['no_compute']+=not g['source_linked_intervening_TASKs'];native_bridge_count+=len(g['source_linked_intervening_TASKs'])
        db.close()
    ck('8638+2',len(seen)==prof['source_bound_TASKs']==8638 and api_only==prof['API_only_frontiers']and len(api_only)==2 and len(prof['records'])==960 and roles==dict(FFN_NORM=960,ROUTER=960,TOPK=960,MC2=960,SHARED_QUANT=960,SHARED_UP=959,SHARED_ACT=960,SHARED_DOWN=960,MERGE=959))
    ck('936guardscopes',len(prof['bridges'])==936 and guard_count==dict(engram=48,aux=72,no_compute=816)and geoms=={(800,816):320,(816,816):320,(834,834):320})
    obs=read(O/'hc-moe-observed-frontiers-v141.json');ck('observed_scope',obs['merge_TASK_bound']==959 and obs['shared_after_MC2_same_stream']==960 and obs['API_only_roles']==api_only and obs['Engram_guards']==48 and obs['aux_guards']==72 and obs['no_intervening_compute']==816)
    def dag(p,fork,overlap,fuse,price):
        nd={'ca':(p['attention_coeff'],[]),'attn':(p['attention_out'],[]),'z':(p['post1'],['ca','attn']),'pre':(p['pre4_delay'],['ca']),'y':(p['collapse'],['z','pre']),'c':(p['ffn_coeff']+(p['overhead']if price else 0),['z']),'u':(p['norm'],['y']if fork else['y','c']),'route':(p['router_topk_mc2']+(p['overhead']if price else 0),['u']),'shared':(p['shared_body']+(p['shared_penalty']if price else 0),['u']if overlap else['route'])}
        if fuse:nd['post']=(p['down']+p['merge']+p['post2']+(p['overhead']if price else 0),['shared','route','c'])
        else:nd.update(down=(p['down'],['shared']),merge=(p['merge'],['down','route']),post=(p['post2'],['merge','c']))
        done={}
        def end(k):
            if k not in done:done[k]=nd[k][0]+max([end(x)for x in nd[k][1]]or[0])
            return done[k]
        return end('post')
    tm=read(O/'hc-moe-readiness-DAGs-v141.json');classes=Counter();trade=Counter()
    for i,r in enumerate(tm['cases']):
        p=r['parameters']
        for k,f,o,fu,pr in [('baseline',False,False,False,False),('FFN_early_only',True,False,False,False),('shared_overlap_only',False,True,False,False),('joint_ideal',True,True,False,False),('joint_repriced',True,True,False,True),('late_fused',True,True,True,False),('late_fused_repriced',True,True,True,True)]:ck('DAG:'+str((i,k)),dag(p,f,o,fu,pr)==r[k])
        classes['FASTER'if r['joint_repriced']<r['baseline']else'SLOWER'if r['joint_repriced']>r['baseline']else'TIED']+=1;trade['FUSED_LATER'if r['late_fused']>r['joint_ideal']else'FUSED_EQUAL'if r['late_fused']==r['joint_ideal']else'FUSED_EARLIER']+=1
    ck('reference_dag_not_speedup',tm['count']==len(tm['cases'])==3072 and dict(classes)==tm['joint_repriced_classes']==dict(FASTER=2140,SLOWER=912,TIED=20)and dict(trade)==tm['fixed_service_fused_tradeoff']==dict(FUSED_LATER=1536,FUSED_EQUAL=1536)and tm['not_physical_lower_or_upper_bound']and tm['actual_speedup']is None)
    def b16(x):
        b=struct.unpack('<I',struct.pack('<f',x))[0];b=(b+32767+((b>>16)&1))&0xffff0000
        return struct.unpack('<f',struct.pack('<I',b))[0]
    w=read(O/'hc-moe-rounding-and-owner-witnesses-v141.json');x=w['merge_round'];ck('Merge_round_counterexample',b16(x['S']+x['R'])==x['rounded_merge']==1 and b16(x['coefficient']*x['rounded_merge'])==x['required_post']==1.5 and b16(x['coefficient']*(x['S']+x['R']))==x['illegal_distributed_post']==1.5078125)
    x=w['post_round'];ck('Post_round_counterexample',[b16(y)for y in x['pre_round_post']]==x['rounded_post']and b16(sum(x['rounded_post']))==x['required_collapse']==1 and b16(sum(x['pre_round_post']))==x['illegal_skip_Post_round']==1.0078125)
    x=w['normcast_round'];ck('FLOAT_is_widened_BF16',b16(x['unrounded_normalized'])==x['correct_FLOAT_plane']!=x['illegal_FLOAT_plane'])
    x=w['lost_full4H'];ck('collapsed_not_full4H',sum(a*b for a,b in zip(x['X'],x['premix']))==sum(a*b for a,b in zip(x['Xprime'],x['premix']))==10 and x['coeff_X']!=x['coeff_Xprime']and not x['frozen_loaded_weight_witness'])
    x=w['receiver_vs_owner'];ck('receiver_not_Mlocal_k',x['local_owned_M']*x['local_topK']==x['incorrect_receiver_work']==0 and x['peer_assignments_received']==x['correct_processed_assignments']==7 and not x['skip_MC2_collective_legal'])
    for x in read(O/'hc-moe-work-and-lifetimes-v141.json')['geometry']:
        m,pp,h,i,e=x['M'],x['P'],x['H'],x['I'],x['E'];ck('work_interfaces',x['n']==geoms[m,pp]and x['router_classical_ops']==2*m*h*e and x['shared_classical_ops']==6*m*h*i and x['full_FLOAT_plane_store_read']==8*m*h and x['chosen_BF16_router_reread']==2*m*h and x['chosen_FLOAT_elision_net_reference']==6*m*h and x['Down_BF16_private_interface']==x['Merge_BF16_private_interface']==4*m*h and x['joint_Down_Merge_private_interface_union']==8*m*h and x['shared_DQ_private_quant_scale_interface']==2*m*(h+4)and x['full4H_Post_plane_content']==8*m*h and x['full4H_output_deletion_credit']==0 and x['padded_BF16_input_content_delta']==2*(pp-m)*h and x['source_dispatch_record_stride']==h+32 and x['semantic_quantized_row_plus_scale']==h+4 and x['physical_HBM_bytes_or_extra_allocation_or_saved_time']is None)
    comp=read(O/'hc-moe-composition-v141.json');ck('4_pairs',comp['new_pairs']==[['F01',x]for x in ['F20','F21','F22','F23']]and comp['new_pair_clauses']==4 and not comp['all_runtime_gates_proved']and not comp['whole_pair_or_all_subsets_closed'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v140.json')
        for pair in comp['new_pairs']:ck('pending:'+str(pair),not any({x['a'],x['b']}==set(pair)for x in rel['clauses']))
        for gid,member in comp['group_extensions']:ck('newgroupmember:'+str((gid,member)),member not in next(g for g in rel['higher_order_groups']if g['id']==gid)['members'])
        manifest=read(O/'checkpoint-manifest-v140.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v141.json');ck('not_done',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v141.json');ck('157_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==157 and len({tuple(sorted((x['a'],x['b'])))for x in rel['clauses']})==157 and len(rel['higher_order_groups'])==26)
        for c in comp['clauses']:
            x=next(r for r in rel['clauses']if r['id']==c['id']);ck('conditional:'+c['id'],{x['a'],x['b']}=={c['a'],c['b']}and not x['all_runtime_gates_proved']and x['critical_path_gain_us']is None)
        for gid,member in comp['group_extensions']:
            g=next(r for r in rel['higher_order_groups']if r['id']==gid);ck('group_open:'+str((gid,member)),member in g['members']and not g['all_subsets_analyzed']and not g['legal_joint_fusion_proved'])
        cv=read(O/'candidate-relation-coverage-v141.json');nv=read(O/'candidate-pair-navigation-v141.json');ck('157_438',cv['scoped_conditional_pairs']==157 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==438)
        a=read(O/'operator-analysis-coverage-ledger-v140.json');b=read(O/'operator-analysis-coverage-ledger-v141.json');ck('nativegaps_unchanged',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        a=read(O/'candidate-profile-ledger-v140.json');b=read(O/'candidate-profile-ledger-v141.json');ck('broadgaps_unchanged',[x.get('conclusion_changing_gap')for x in a['candidates']]==[x.get('conclusion_changing_gap')for x in b['candidates']])
        mm=read(O/'module-phase-branch-matrix-v141.json');ck('358_26',len(mm['modules'])==26 and sum(len(x['branches'])for x in mm['modules'])==358)
        for r in read(O/'candidate-record-index-v141.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v141.json')
    for x in manifest['artifacts']:ck('manifest:'+x['path'],pinned(x))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,frontiers=960,TASK_bindings=8638,API_only_roles=2,interlayer_bridges=936,intervening_native_bridge_TASKs=native_bridge_count,Engram_guards=48,aux_guards=72,no_intervening_compute=816,DAGs=3072,new_scoped_pairs=4,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-moe-preflight-v141.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
