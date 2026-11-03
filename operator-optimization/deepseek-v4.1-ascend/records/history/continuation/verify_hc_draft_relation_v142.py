"""Independent raw identity,epoch,demand and sufficient-schedule checks."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,sqlite3,struct,math
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(n,v):
        assert v,n
        checks.append(n)
    def pin(x):
        b=Path(x['path']).read_bytes();return len(b)==x['bytes']and hashlib.sha256(b).hexdigest()==x['sha256']
    for x in read(O/'hc-draft-inputs-v142.json')['inputs']:ck('input:'+x['path'],pin(x))
    for x in read(O/'hc-draft-source-contract-v142.json')['parts']:ck('source:'+str((x['path'],x['first_line'])),pin(x)and Path(x['path']).read_text().splitlines()[x['first_line']-1:x['last_line']]==x['source'])
    p=read(O/'hc-draft-native-frontiers-v142.json');parents={(r['database_index'],r['source_order_ordinal']):r for r in read(O/'context-sp-profile-frontiers-v130.json')['records']};groups=defaultdict(list);seen=set();roles=Counter();ancestry=Counter();geoms=Counter();unknown=0
    target={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sp-eager-native-frontiers-v134.json')['records']};obs=read(O/'hc-draft-observed-frontiers-v142.json');auxby=defaultdict(list)
    for r in obs['target_eager_aux_frontiers']:auxby[r['database_index']].append(r)
    for r in p['records']:groups[r['database_index']].append(r)
    for ref in p['databases']:
        di=ref['database_index'];ck('rawpin:'+str(di),pin(ref['raw_database']));ck('cachepin:'+str(di),pin(ref['cached_native']))
        db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};fs={(x['id'],x['stackDepth'],x['stack'])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};ck('missingstrings:'+str(di),sorted({x[2]for x in fs if x[2]not in ss})==ref['unresolved_raw_stack_ids']);cids=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cids[x['id']].add(x['connectionId'])
        aa={x['api_rowid']:x for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')};tt={x['task_record_rowid']:x for x in db.execute('SELECT rowid AS task_record_rowid,*FROM TASK')};dd={x['globalTaskId']:x for x in db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO')}
        def api(a,tag):
            raw=aa[a['api_rowid']];ck('API:'+tag,all(raw[k]==a[k]for k in raw.keys())and ss[raw['name']]==a['name_text'])
        def stage(s,tag,unique=True):
            a,t,d=s['API'],s['TASK'],s['definition'];api(a,tag);raw=tt[t['task_record_rowid']];de=dd[t['globalTaskId']]
            ck('TASK/CID:'+tag,all(raw[k]==t[k]for k in raw.keys())and a['api_rowid']in t['native_linked_API_rowids']and t['connectionId']in cids[a['connectionId']])
            ck('Compute:'+tag,all((ss[de[k[:-5]]]if k.endswith('_text')else de[k])==v for k,v in d.items()))
            ff=s['selected_frames'];ck('frames:'+tag,bool(ff)and all((f['id'],f['stackDepth'],f['stack'])in fs and f['value']==ss.get(f['stack'])and f['id']==a['callchainId']for f in ff))
            if unique:
                key=di,t['task_record_rowid'];ck('unique:'+tag,key not in seen);seen.add(key)
        def has(s,path,line):return any(path+f'({line}):'in(f['value']or'')for f in s['selected_frames'])
        for r in groups[di]:
            key=di,r['source_order_ordinal'];par=parents[key];ck('parent:'+str(key),r['context_parent']==par and not r['actual_next_token_seed_or_current_proposal_metadata_values_bound'])
            stage(r['repeat'],'repeat:'+str(key));stage(r['seed'],'seed:'+str(key));ck('repeat/seedsource',has(r['repeat'],'/models/deepseek_v41/dspark.py',249)and has(r['seed'],'/spec_decode/llm_base_proposer.py',1584))
            anc='DUMMY'if has(r['repeat'],'/spec_decode/dspark_proposer.py',421)else'PROPOSE'if has(r['repeat'],'/spec_decode/llm_base_proposer.py',1394)else'UNKNOWN';ck('ancestry',anc==r['source_ancestry']and anc!='UNKNOWN');ancestry[anc]+=1
            ra,sa=r['repeat']['API'],r['seed']['API'];lo,hi=int(ra['endNs']),int(sa['startNs']);ck('window',r['CPU_window']==dict(globalTid=ra['globalTid'],startNs=ra['endNs'],endNs=sa['startNs'])and ra['globalTid']==sa['globalTid']and lo<hi and len(r['layers'])==3)
            def inside(s):
                a=s['API'];return a['globalTid']==ra['globalTid']and lo<=int(a['startNs'])<=int(a['endNs'])<=hi
            m=int(r['layers'][0]['stages']['HC']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[par['context_rows'],par['query_KV_rows'],m,anc]+=1
            for j,l in enumerate(r['layers']):
                ck('SPparent',l['source_layer_ordinal']==j and l['context_SP_parent']==par['layers'][j]and not l['actual_request_weight_cache_or_native_new_publication_identity'])
                api(l['SP_API'],'SP'+str((key,j)));api(l['RS_API'],'RS'+str((key,j)));ck('SPCID',l['context_SP_parent']['draft_SP_allGather']['connectionId']in cids[l['SP_API']['connectionId']]and l['SP_API']['globalTid']==ra['globalTid'])
                for role,line,op in [('HC',889,'HcPre'),('NORM',896,'RmsNorm'),('POST',902,'HcPost'),('FFN_PRE',905,'HcPre'),('FFN_NORM',846,'RmsNormCast'),('FFN_POST',919,'HcPost')]:
                    s=l['stages'][role];stage(s,str((key,j,role)));roles[role]+=1;ck('source_scope:'+role,inside(s)and has(s,'/models/deepseek_v41/dspark.py',265)and has(s,'/models/deepseek_v41/model.py',line)and s['definition']['opType_text']==op)
                st=l['stages'];ck('typedHC',st['HC']['definition']['inputShapes_text']==st['FFN_PRE']['definition']['inputShapes_text']==f'"{m},4,5120;24,20480;3;24;{m},4"'and st['HC']['definition']['outputShapes_text']==f'"{m},5120;{m},4;{m},4,4;{m},4"'and st['FFN_NORM']['definition']['outputDataTypes_text']=='DT_BF16;FLOAT')
                ck('CPUlayerorder',int(st['HC']['API']['endNs'])<=int(st['NORM']['API']['startNs'])and int(st['NORM']['API']['endNs'])<=int(l['SP_API']['startNs'])and int(l['RS_API']['endNs'])<=int(st['POST']['API']['startNs'])and int(st['POST']['API']['endNs'])<=int(st['FFN_PRE']['API']['startNs'])<=int(st['FFN_NORM']['API']['startNs'])<int(st['FFN_POST']['API']['startNs']))
                if j:ck('layerlink',r['layers'][j-1]['stages']['FFN_POST']['TASK']['endNs']<=st['HC']['TASK']['startNs'])
            c=r['collapse']
            for role,s in c.items():stage(s,'collapse:'+str((key,role)));roles[role]+=1;ck('collapsesource',inside(s)and has(s,'/models/deepseek_v41/dspark.py',273)and has(s,'/models/deepseek_v41/model.py',856))
            ck('typedcollapse',c['CAST_IN']['definition']['inputDataTypes_text']=='DT_BF16'and c['CAST_IN']['definition']['outputDataTypes_text']=='FLOAT'and c['MUL']['definition']['outputDataTypes_text']==c['SUM']['definition']['outputDataTypes_text']=='FLOAT'and c['CAST_OUT']['definition']['inputDataTypes_text']=='FLOAT'and c['CAST_OUT']['definition']['outputDataTypes_text']=='DT_BF16'and c['CAST_OUT']['definition']['outputShapes_text']==f'"{m},5120"')
            ck('collapsechain',r['layers'][-1]['stages']['FFN_POST']['TASK']['endNs']<=c['CAST_IN']['TASK']['startNs']<=c['CAST_IN']['TASK']['endNs']<=c['MUL']['TASK']['startNs']<=c['MUL']['TASK']['endNs']<=c['SUM']['TASK']['startNs']<=c['SUM']['TASK']['endNs']<=c['CAST_OUT']['TASK']['startNs'])
            allst=[r['repeat'],r['seed']]+list(c.values())+[s for l in r['layers']for s in l['stages'].values()];unknown+=sum(f['value']is None for s in allst for f in s['selected_frames'])
        for x in auxby[di]:
            key=di,x['forward_API_rowid'],x['entering_layer'];orig=target[key];ck('auxparent',x['entering_layer']in [37,38,39]and x['mean']==orig['aux_mean']and x['HC']==orig['stages']['HC']and x['allGather']==orig['aux_allGather']and x['input_residual_preceding_FFN']==orig['previous_FFN_Post']and not x['actual_later_draft_request_bound'])
            for k in ['mean','HC','input_residual_preceding_FFN']:stage(x[k],'aux:'+str((key,k)),False)
            ck('auxbeforecurrentHC',int(x['mean']['API']['endNs'])<=int(x['HC']['API']['startNs'])and x['input_residual_preceding_FFN']['TASK']['endNs']<=x['mean']['TASK']['startNs']<=x['mean']['TASK']['endNs']<=x['HC']['TASK']['startNs'])
        db.close()
    ck('2496_native',len(seen)==2496 and len(p['records'])==104 and roles==dict(HC=312,NORM=312,POST=312,FFN_PRE=312,FFN_NORM=312,FFN_POST=312,CAST_IN=104,MUL=104,SUM=104,CAST_OUT=104)and ancestry==p['source_ancestry']==dict(DUMMY=16,PROPOSE=88))
    ck('positiveframes',unknown==obs['unresolved_selected_frame_references']==0 and obs['raw_DBs_with_unresolved_stack_ids']==13)
    ck('72targetaux',len(obs['target_eager_aux_frontiers'])==obs['source_bound_target_aux']==72)
    for r in obs['records']:ck('observation',r['context_before_repeat']and all(r['HC_before_SP'])and r['collapse_after_last_FFN']and not r['actual_same_client_step_or_tensor_bank_bound'])
    for x in obs['geometry']:ck('geometry',geoms[x['context_rows'],x['query_rows'],x['local_M'],x['ancestry']]==x['n'])
    def dag(p,early,fork,price):
        nd={'meta':(p['metadata'],[]),'main':(3,[])}
        for j in range(3):nd['c'+str(j)]=(p['context']+j+(p['extra']if price else 0),['meta','main']if j==0 else['c'+str(j-1)])
        nd['e']=(p['embedding'],['meta']if early else['c2']);prev='e'
        for j in range(3):
            y,c,q,a,z,x=['%s%d'%(k,j)for k in ['y','coeff','q','a','z','x']];nd[y]=(1,[prev]);nd[c]=(p['coeff']+(p['extra']if price else 0),[prev]);nd[q]=(p['query']+(p['query_penalty']if price else 0),[y]if fork else[y,c]);nd[a]=(p['attention'],[q,'c'+str(j)]);nd[z]=(2,[a,c]);nd[x]=(p['ffn'],[z]);prev=x
        nd['end']=(5+p['head_markov'],[prev]);done={}
        def end(k):
            if k not in done:done[k]=nd[k][0]+max([end(z)for z in nd[k][1]]or[0])
            return done[k]
        end('end');return done
    tm=read(O/'hc-draft-readiness-DAGs-v142.json');classes=Counter();interactions=Counter()
    for i,r in enumerate(tm['cases']):
        for k,e,f,pr in [('baseline',False,False,False),('HC_fork_only',False,True,False),('early_draft_only',True,False,False),('joint_ideal',True,True,False),('joint_repriced',True,True,True)]:
            d=dag(r['parameters'],e,f,pr);ck('DAG:'+str((i,k)),d['end']==r[k]['proposal_ready']and d['c2']==r[k]['context_bank_last_read']and all(all(d[base+str(j)]==r[k]['layers'][j][role]for role,base in [('Y','y'),('coeff','coeff'),('Q','q'),('context','c'),('attention','a'),('Post','z'),('FFNPost','x')])for j in range(3)))
        b=r['baseline']['proposal_ready'];v=r['joint_repriced']['proposal_ready'];classes['FASTER'if v<b else'SLOWER'if v>b else'TIED']+=1;s=b-r['HC_fork_only']['proposal_ready']+b-r['early_draft_only']['proposal_ready'];j=b-r['joint_ideal']['proposal_ready'];interactions['OVERLAPPING_CREDITS'if j<s else'SYNERGY'if j>s else'EQUAL']+=1
    ck('1536DAGscope',len(tm['cases'])==tm['count']==1536 and dict(classes)==tm['classes']==dict(FASTER=992,SLOWER=544)and dict(interactions)==tm['ideal_interactions']==dict(EQUAL=768,OVERLAPPING_CREDITS=768))
    w=read(O/'hc-draft-value-epoch-and-demand-witnesses-v142.json');x=w['mean_vs_collapsed'];ck('mean_not_HCY',sum(a*b for a,b in zip(x['X'],x['premix']))==x['HC_Y']==1 and sum(x['X'])/4==x['aux_mean']==2.5)
    x=w['finite_first_Y'];ck('finite_firstY',[sum(x['premix'][j]*x['repeat'][j][h]for j in range(4))for h in range(2)]==x['Y']==x['embedding']and not x['coefficients_or_full_first_Post_dead'])
    ck('nonfinite_firstY',math.isnan(sum(p*float('inf')for p in [1,0,0,0]))and not w['nonfinite_first_Y']['unconditional_equality'])
    def f32(x):return struct.unpack('>f',struct.pack('>f',x))[0]
    x=w['coefficient_prefold_round'];s=0.
    for row in x['weights']:
        for a,b in zip(row,x['E']):s=f32(s+f32(a*b))
    folded=[sum(row[h]for row in x['weights'])for h in range(2)];ck('prefold_round_counterexample',s==x['original_sequential_FP32']==0 and folded==x['real_prefold_weights']and sum(a*b for a,b in zip(folded,x['E']))==x['prefold_FP32']==1 and not x['loaded_HF32_or_native_accumulation_attested'])
    x=w['raw_confidence_not_normalized']
    def b16(x):
        u=struct.unpack('<I',struct.pack('<f',x))[0];return struct.unpack('<f',struct.pack('<I',(u+32767+((u>>16)&1))&0xffff0000))[0]
    ck('raw_confidence_observer', [b16(a/math.sqrt(a*a+x['epsilon']))for a in x['raw']]==x['BF16_normalized']==[1,1]and x['constructed_confidence_scores'][0]!=x['constructed_confidence_scores'][1]and not x['frozen_confidence_weights_or_active_branch_bound'])
    x=w['selected_query_needs_other_KV'];ck('demand_KV_fence',x['query']==0 and sum(x['values'])/len(x['values'])==x['selected_query_output']==6 and x['values'][-1]==x['wrong_remove_unselected_KV_output']==10 and not x['earlier_HC_rows_deletable'])
    x=w['same_shape_wrong_epoch'];ck('epoch_identity',len(x['old']['positions'])==len(x['new']['positions'])==x['shape'][0]and x['old']['mask']!=x['new']['mask']and x['old']['generation']!=x['new']['generation']and not x['metadata_reuse_legal'])
    x=w['bank_retirement'];ck('retirement',x['new_write_time']<x['old_reader_time']and x['old_reader_observes_after_unsafe_reuse']==x['new_value']!=x['expected_old_reader_value']==x['old_value'])
    x=w['proposal_not_accepted'];ck('keep_not_next',len(x['proposal'])==x['keep_count']==2 and x['all_accepted_sampled'][-1]==x['all_accepted_next']!=x['reject_next']==x['early_reject_sampled'][1]and not x['next_ID_determined_by_keep'])
    g=w['generation_graph'];edges=set(map(tuple,g['edges']));cy=g['cycle'];ck('future_cutoff_cycle',all((a,b)in edges|{tuple(g['illegal_future_cutoff_edge'])}for a,b in zip(cy,cy[1:]))and cy[0]==cy[-1]and tuple(g['illegal_future_cutoff_edge'])not in edges and g['optional_confidence_not_runtime_attested'])
    for x in read(O/'hc-draft-work-and-lifetimes-v142.json')['geometry']:
        c,q,m,h=x['context_rows'],x['query_rows'],x['local_M'],x['H'];ck('work_scope',x['n']==geoms[c,q,m,x['ancestry']]and x['global_repeat_value_content']==8*q*h and x['local_repeat_value_content']==8*m*h and x['local_embedding_value_content']==2*m*h and x['HC_coeff_classical_matrix_reference_per_layer']==2*m*24*4*h and x['unchanged_current_context_three_KV_matrix_reference']==3*2*c*h*512 and x['early_HC_schedule_matrix_work_credit']==0 and x['raw_draft_collapsed_global_value_content']==2*q*h and x['context_bank_content']==2*c*h and not x['raw_draft_confidence_observer_closed']and x['physical_HBM_or_allocation_or_time']is None)
    comp=read(O/'hc-draft-composition-v142.json');ck('5relations',comp['new_pairs']==[['F01',x]for x in ['F28','F29','F30','F31','F35']]and comp['new_pair_clauses']==5 and not comp['whole_F01_or_pair_or_all_subsets_closed']and not comp['all_runtime_gates_proved'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v141.json')
        for pair in comp['new_pairs']:ck('pending:'+str(pair),not any({x['a'],x['b']}==set(pair)for x in rel['clauses']))
        for gid,member in comp['group_extensions']:ck('newgroupmember:'+str((gid,member)),member not in next(g for g in rel['higher_order_groups']if g['id']==gid)['members'])
        manifest=read(O/'checkpoint-manifest-v141.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v142.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v142.json');ck('162_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==162 and len({tuple(sorted((x['a'],x['b'])))for x in rel['clauses']})==162 and len(rel['higher_order_groups'])==26)
        ck('F01star_scoped_not_closed',sum('F01'in[x['a'],x['b']]for x in rel['clauses'])==comp['F01_star_conditional_pairs_after']==34)
        for c in comp['clauses']:
            x=next(r for r in rel['clauses']if r['id']==c['id']);ck('conditional:'+c['id'],{x['a'],x['b']}=={c['a'],c['b']}and not x['all_runtime_gates_proved']and x['critical_path_gain_us']is None)
        for gid,member in comp['group_extensions']:
            x=next(g for g in rel['higher_order_groups']if g['id']==gid);ck('groupsopen',member in x['members']and not x['all_subsets_analyzed']and not x['legal_joint_fusion_proved'])
        cv=read(O/'candidate-relation-coverage-v142.json');nv=read(O/'candidate-pair-navigation-v142.json');ck('162_433',cv['scoped_conditional_pairs']==162 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==433)
        a=read(O/'operator-analysis-coverage-ledger-v141.json');b=read(O/'operator-analysis-coverage-ledger-v142.json');ck('gapsunchanged',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        a=read(O/'candidate-profile-ledger-v141.json');b=read(O/'candidate-profile-ledger-v142.json');ck('broadergaps',[x.get('conclusion_changing_gap')for x in a['candidates']]==[x.get('conclusion_changing_gap')for x in b['candidates']])
        mm=read(O/'module-phase-branch-matrix-v142.json');ck('358_26',len(mm['modules'])==26 and sum(len(x['branches'])for x in mm['modules'])==358)
        for x in read(O/'candidate-record-index-v142.json')['model_candidates']:ck('master:'+x['id'],hashlib.sha256(Path(x['master_record']).read_bytes()).hexdigest()==x['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v142.json')
    for x in manifest['artifacts']:ck('manifest:'+x['path'],pin(x))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,draft_frontiers=104,draft_layers=312,native_TASK_bindings=2496,reused_target_aux_frontiers=72,DAGs=1536,new_scoped_pairs=5,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-draft-preflight-v142.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
