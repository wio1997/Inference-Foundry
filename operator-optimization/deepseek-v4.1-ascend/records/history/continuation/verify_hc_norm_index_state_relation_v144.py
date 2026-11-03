"""Verify typed source-key unions,pair arithmetic,bufferretirement andfuturedeadlines."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,hashlib,sqlite3,struct,math
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(k,v):
        assert v,k
        checks.append(k)
    def pin(x):
        b=Path(x['path']).read_bytes();return len(b)==x['bytes']and hashlib.sha256(b).hexdigest()==x['sha256']
    for x in read(O/'hc-norm-index-state-inputs-v144.json')['inputs']:ck('input:'+x['path'],pin(x))
    for x in read(O/'hc-norm-index-state-source-contract-v144.json')['parts']:ck('source:'+str((x['path'],x['first_line'])),pin(x)and Path(x['path']).read_text().splitlines()[x['first_line']-1:x['last_line']]==x['source'])
    key=lambda r:(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal'])
    profiles={n:{key(r):r for r in read(O/n)['records']}for n in ['hc-norm-projection-native-frontiers-v143.json','hc-index-query-weight-native-producers-v139.json','hc-sparse-eager-native-frontiers-v138.json','hc-c1-native-consumers-v135.json','hc-c2-native-consumers-v136.json']};prof=read(O/'hc-norm-index-state-native-joins-v144.json');bydb=defaultdict(list);seen=set();roles=Counter();modes=Counter();geom=Counter();orders=Counter()
    for r in prof['records']:bydb[r['database_index']].append(r)
    obs={tuple(x['key']):x for x in read(O/'hc-norm-index-state-observed-v144.json')['records']}
    for ref in prof['databases']:
        di=ref['database_index'];ck('rawpin',pin(ref['raw_database']));ck('cachepin',pin(ref['cached_native']));db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};fs={(x['id'],x['stackDepth'],x['stack'])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};cids=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cids[x['id']].add(x['connectionId'])
        aa={x['api_rowid']:x for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')};tt={x['task_record_rowid']:x for x in db.execute('SELECT rowid AS task_record_rowid,*FROM TASK')};dd={x['globalTaskId']:x for x in db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO')}
        for r in bydb[di]:
            k=key(r);s=r['stages'];n=profiles['hc-norm-projection-native-frontiers-v143.json'][k];i=profiles['hc-index-query-weight-native-producers-v139.json'][k];sp=profiles['hc-sparse-eager-native-frontiers-v138.json'][k];expected={role:n['stages'][role]for role in ['HC','INPUT_NORM','QA','QA_NORM','QA_QUANT','QB','SWA_KV','QUERY_ROPE']};expected.update({role:v for role,v in i['producers'].items()if role not in ['QR_NORM','QB_INPUT_QUANT']});expected.update({role:sp['stages'][role]for role in ['QLI','PREPARE','QUERY_QUANT','SMLA']})
            mode='INDEX_QUERY_WEIGHT_ONLY'
            if k in profiles['hc-c1-native-consumers-v135.json']:
                mode='C1';expected['C1_PROJ']=n['stages']['C1_PROJ'];expected.update({'CACHE_'+role:v for role,v in profiles['hc-c1-native-consumers-v135.json'][k]['consumers'].items()if role not in ['SWA_NORM','SWA_ROPE']})
            if k in profiles['hc-c2-native-consumers-v136.json']:
                mode='C2';expected.update({role:n['stages'][role]for role in ['C2_INPUT_CAST','C2_KV','C2_SCORE']});expected.update({'CACHE_'+role:v for role,v in profiles['hc-c2-native-consumers-v136.json'][k]['consumers'].items()if role not in ['INPUT_CAST','SWA_NORM','SWA_ROPE']})
            ck('canonicalunion:'+str(k),s==expected and r['mode']==mode and not r['actual_tensor_data_edge_pointer_parameter_or_state_epoch_bound']);modes[mode]+=1
            for role,z in s.items():
                a,t,d=z['API'],z['TASK'],z['definition'];ar=aa[a['api_rowid']];tr=tt[t['task_record_rowid']];de=dd[t['globalTaskId']];tag=str((k,role));ff=z.get('selected_frames',z.get('source_frames'));tk=di,t['task_record_rowid']
                ck('API:'+tag,all(a[x]==ar[x]for x in ar.keys())and a['name_text']==ss[ar['name']]);ck('TASKCID:'+tag,all(t[x]==tr[x]for x in tr.keys())and a['api_rowid']in t['native_linked_API_rowids']and t['connectionId']in cids[a['connectionId']]);ck('Compute:'+tag,all((ss[de[x[:-5]]]if x.endswith('_text')else de[x])==v for x,v in d.items()));ck('frames:'+tag,bool(ff)and all((f['id'],f['stackDepth'],f['stack'])in fs and f['id']==a['callchainId']and f['value']==ss[f['stack']]for f in ff));ck('unique:'+tag,tk not in seen);seen.add(tk);roles[role]+=1
            m=int(s['INPUT_NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]);q=int(s['QA']['definition']['inputShapes_text'].strip('"').split(',')[0]);t=int(s['SWA_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);geom[mode,m,q,t]+=1;ck('typed_index',s['WEIGHT']['definition']['inputShapes_text']==f'"{q},5120;32,5120"'and s['WEIGHT']['definition']['inputDataTypes_text']=='DT_BF16;DT_BF16'and s['INDEX_Q']['definition']['inputDataTypes_text'].startswith('INT8;INT8;')and s['INDEX_INPUT_QUANT']['definition']['inputShapes_text']==f'"{q},1280"'and s['WEIGHT_FP16']['definition']['outputDataTypes_text']=='FLOAT16'and s['QUERY_QUANT']['definition']['outputDataTypes_text']=='INT8;FLOAT16')
            o=obs[k];ck('baseorders',o['mode']==mode and (o['M'],o['Q'],o['T'])==(m,q,t)and o['inputRMS_before_Weight']==(s['INPUT_NORM']['TASK']['endNs']<=s['WEIGHT']['TASK']['startNs'])and o['QRnorm_before_IndexQ']==(s['QA_NORM']['TASK']['endNs']<=s['INDEX_Q']['TASK']['startNs'])and not o['actual_Z_or_QR_quantizer_equal_or_state_epoch_bound'])
            if mode=='C2':
                ck('C2types',s['C2_INPUT_CAST']['definition']['inputDataTypes_text']=='DT_BF16'and s['C2_INPUT_CAST']['definition']['outputDataTypes_text']=='FLOAT'and s['C2_KV']['definition']['inputDataTypes_text']==s['C2_SCORE']['definition']['inputDataTypes_text']=='FLOAT;FLOAT'and s['CACHE_POOL']['definition']['outputDataTypes_text']=='DT_BF16'and s['CACHE_POOL_NORM']['definition']['inputShapes_text']==f'"{t},512;512"')
                current=dict(QueryRoPE_before_C2cast=s['QUERY_ROPE']['TASK']['endNs']<=s['C2_INPUT_CAST']['TASK']['startNs'],IndexRoPE_before_C2cast=s['INDEX_ROPE']['TASK']['endNs']<=s['C2_INPUT_CAST']['TASK']['startNs'],ringupdate_before_poolnorm=s['CACHE_RING_UPDATE']['TASK']['endNs']<=s['CACHE_POOL_NORM']['TASK']['startNs'],IndexQuant_intersects_C2KV=max(s['QUERY_QUANT']['TASK']['startNs'],s['C2_KV']['TASK']['startNs'])<min(s['QUERY_QUANT']['TASK']['endNs'],s['C2_KV']['TASK']['endNs']))
                ck('C2orders',all(o[x]==v for x,v in current.items()));orders.update({x:int(v)for x,v in current.items()})
            if mode in ['C1','C2']:
                ck('keyqueryprecision',s['CACHE_INDEX_QUANT']['definition']['outputDataTypes_text']=='INT8;FLOAT'and s['CACHE_INDEX_SCALE_CAST']['definition']['inputDataTypes_text']=='FLOAT'and s['CACHE_INDEX_SCALE_CAST']['definition']['outputDataTypes_text']=='FLOAT16'and s['CACHE_LONG_STORE']['definition']['outputDataTypes_text']=='DT_BF16')
        db.close()
    ck('5064union',len(seen)==prof['source_bound_TASK_bindings']==5064 and len(prof['records'])==192 and roles==prof['native_role_counts']and modes==prof['modes']==dict(C2=72,C1=24,INDEX_QUERY_WEIGHT_ONLY=96)and prof['new_native_bindings_or_promotions']==0)
    ck('current_orders_notsemanticbarriers',dict(orders)==read(O/'hc-norm-index-state-observed-v144.json')['C2_recorded_orders']==dict(QueryRoPE_before_C2cast=72,IndexRoPE_before_C2cast=72,ringupdate_before_poolnorm=72,IndexQuant_intersects_C2KV=66))
    def fp(x):return struct.unpack('>f',struct.pack('>f',x))[0]
    def b16(x):
        u=struct.unpack('>I',struct.pack('>f',x))[0];return struct.unpack('>f',struct.pack('>I',(u+32767+((u>>16)&1))&0xffff0000))[0]
    def refpair(scores,vals,one=False,comp=False):
        a,b=map(fp,scores);x,y=map(fp,vals);hi=max(a,b)
        if not one:
            ea,eb=[fp(math.exp(fp(v-hi)))for v in [a,b]];den=fp(ea+eb);p,q=fp(ea/den),fp(eb/den)
        else:
            e=fp(math.exp(fp(min(a,b)-hi)));den=fp(1+e);pmax,pmin=fp(1/den),fp(e/den)
            if comp:pmin=fp(1-pmax)
            p,q=(pmax,pmin)if a>=b else(pmin,pmax)
        return dict(p=[p,q],out=fp(fp(p*x)+fp(q*y)))
    alg=read(O/'hc-norm-index-state-pair-algebra-v144.json');bad=0
    for i,x in enumerate(alg['cases']):
        a=refpair(x['scores'],x['values']);b=refpair(x['scores'],x['values'],True);c=refpair(x['scores'],x['values'],True,True);ck('pair:'+str(i),a==x['reference']==b==x['one_exp']and c==x['complement']);bad+=a!=c
    ck('784finitepairreferences',len(alg['cases'])==alg['count']==alg['reference_one_exp_matches']==784 and bad==alg['complement_changes']==640)
    wit=read(O/'hc-norm-index-state-pool-and-precision-witnesses-v144.json');x=wit['small_probability'];a=refpair(x['scores'],x['values']);b=refpair(x['scores'],x['values'],True,True);ck('smallp_notcomplement',a==x['reference']and b==x['one_minus_pmax']and a['p'][0]==1 and a['p'][1]>0 and b['p'][1]==0 and b16(a['out'])==x['BF16_reference']==.2060546875 and x['BF16_complement']==0)
    def realpool(kv,sc):
        return [sum(math.exp(sc[j][h]-max(row[h]for row in sc))*kv[j][h]for j in range(2))/sum(math.exp(sc[j][h]-max(row[h]for row in sc))for j in range(2))for h in range(2)]
    x=wit['pool_not_homogeneous'];a=realpool(x['K'],x['scores']);c=x['common_input_factor'];b=realpool([[c*y for y in row]for row in x['K']],[[c*y for y in row]for row in x['scores']]);ck('pool_direction',a==x['pooled']and b==x['scaled_pooled']and a[0]/a[1]==x['original_direction_ratio']and b[0]/b[1]==x['scaled_direction_ratio']and abs(a[0]/a[1]-b[0]/b[1])>4 and x['following_scalar_RMS_cannot_restore_direction'])
    x=wit['old_ring_current_factor'];a=realpool([x['old_K'],x['current_K']],[x['old_score'],x['current_score']]);b=realpool([x['old_K'],[2*y for y in x['current_K']]],[x['old_score'],[2*y for y in x['current_score']]]);ck('oldstate_scale',a==x['original']and b==x['current_factor2']and abs(a[0]/a[1]-b[0]/b[1])>.5 and not x['one_global_rescale_suffices'])
    x=wit['mixed_weight_precision'];ck('mixed_weight_notdowngrade',b16(x['FLOAT_weight'])==x['BF16_weight']==1 and x['FLOAT_reference_output']==x['FLOAT_weight']!=x['BF16_downgraded_output']and not x['actual_HF32_accumulation_policy_attested'])
    x=wit['lifetime_fence'];ck('retirement_fences',x['Z_reader_end']<x['FLOAT_projection_read_end']<x['pool_done']<x['next_actual_old_ring_read']<x['ring_done']and x['projected_plane_discard_after_pool_wrong']and x['Z_need_not_wait_ring']and not x['ring_commit_meets_fixed_future_deadline'])
    def dag(p,early,fork,price):
        nd={'y':(p['hc_y'],[]),'c':(p['hc_coeff'],[]),'z':(p['input_norm'],['y','c']),'q':(p['query']+(p['extra']if price else 0),['z']),'meta':(p['metadata'],[]),'cast':(1,['z','meta']if early else['q','meta']),'proj':(p['projection']+(p['extra']if price else 0),['cast']),'oldgen':(p['old_generation'],[]),'oldread':(1,['proj','oldgen']),'pool':(p['pool'],['oldread']),'ring':(p['ring'],['oldread']if fork else['pool']),'norm':(p['latent_norm'],['pool']if fork else['pool','ring']),'endpoints':(p['key_long'],['norm']),'qi':(2,['q']),'qli':(2,['qi','endpoints']),'attn':(3,['q','endpoints','qli']),'post':(0,['c','attn'])};done={}
        def end(k):
            if k not in done:done[k]=nd[k][0]+max([end(z)for z in nd[k][1]]or[0])
            return done[k]
        end('post');end('ring');return done
    tm=read(O/'hc-norm-index-state-readiness-DAGs-v144.json');classes=Counter();future=Counter()
    for i,x in enumerate(tm['cases']):
        p=x['parameters']
        for k,e,f,pr in [('baseline',False,False,False),('early_C2_only',True,False,False),('norm_state_fork_only',False,True,False),('joint_ideal',True,True,False),('joint_repriced',True,True,True)]:
            d=dag(p,e,f,pr);r=x[k];ck('DAG:'+str((i,k)),all(d[node]==r[role]for role,node in [('Z','z'),('Q','q'),('CAST','cast'),('PROJ','proj'),('OLD_READ','oldread'),('POOL','pool'),('RING','ring'),('NORM','norm'),('ENDPOINTS','endpoints'),('QLI','qli'),('Post_input','post')])and r['Z_reader_sufficient_release']==max(d['q'],d['cast'])and r['FLOAT_input_sufficient_release']==d['proj']and r['projected_KVScore_sufficient_release']==max(d['pool'],d['ring'])and r['next_ring_read_deadline']==d['post']+p['future_gap']and r['future_ring_deadline_satisfied']==(d['ring']<=d['post']+p['future_gap']))
        a,b=x['joint_repriced']['Post_input'],x['baseline']['Post_input'];s='FASTER'if a<b else'SLOWER'if a>b else'TIED';classes[s]+=1;ok=x['joint_repriced']['future_ring_deadline_satisfied'];future['FUTURE_READY'if ok else'FUTURE_VIOLATED']+=1;future[('READY_'if ok else'VIOLATED_')+s]+=1
    ck('6144DAGscope',len(tm['cases'])==tm['count']==6144 and dict(classes)==tm['joint_repriced_classes']==dict(FASTER=4128,SLOWER=1792,TIED=224)and dict(future)==tm['future_deadline_classes']and future['FUTURE_VIOLATED']==855 and future['VIOLATED_FASTER']==665 and not tm['physical_bound_or_actual_gain'])
    for x in read(O/'hc-norm-index-state-work-and-lifetimes-v144.json')['geometry']:
        mode,m,q,t,h=x['mode'],x['M'],x['Q'],x['T'],x['H'];ck('work_base',x['frontiers']==geom[mode,m,q,t]and x['input_BF16_Z_row_content']==2*h and x['IndexWeight_QA_shared_row_input_reference']==2*q*h and not x['all_Z_reader_or_alias_state_observer_gates_closed']and x['actual_E_B_U_or_physical_HBM_time_or_allocation']is None)
        if mode=='C2':ck('C2interface_union',x['C2_two_FLOAT_matrix_classical_ops']==4*t*h*512 and x['C2_separate_FLOAT_input_interface']==x['C2_cast_joint_full_union_reference']==12*t*h and x['C2_joint_existing_FLOAT_plane_read_credit']==4*t*h and x['C2_after_joint_FLOAT_plane_interface']==8*t*h and x['C2_projected_FP32_KVScore_plane_content']==4096*t and x['C2_valid_current_projection_row_elision_from_pool_norm_or_even_state']==0 and x['tagged_all_QA_Weight_SWA_C2_panel_row_value_content']==7744)
    comp=read(O/'hc-norm-index-state-composition-v144.json');ck('6pairs',comp['new_pairs']==[['F02',x]for x in ['F09','F10','F11','F17','F18','F19']]and comp['new_pair_clauses']==6 and not comp['all_runtime_gates_proved']and not comp['whole_pair_or_all_subsets_closed'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v143.json')
        for pair in comp['new_pairs']:ck('pending:'+str(pair),not any({x['a'],x['b']}==set(pair)for x in rel['clauses']))
        for gid,m in comp['group_extensions']:ck('newgroup',m not in next(g for g in rel['higher_order_groups']if g['id']==gid)['members'])
        manifest=read(O/'checkpoint-manifest-v143.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v144.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0);rel=read(O/'candidate-relation-clauses-v144.json');ck('173_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==173 and len({tuple(sorted((x['a'],x['b'])))for x in rel['clauses']})==173 and len(rel['higher_order_groups'])==26)
        for c in comp['clauses']:
            x=next(r for r in rel['clauses']if r['id']==c['id']);ck('conditional', {x['a'],x['b']}=={c['a'],c['b']}and not x['all_runtime_gates_proved']and x['critical_path_gain_us']is None)
        for gid,m in comp['group_extensions']:
            x=next(g for g in rel['higher_order_groups']if g['id']==gid);ck('groupopen',m in x['members']and not x['all_subsets_analyzed']and not x['legal_joint_fusion_proved'])
        cv=read(O/'candidate-relation-coverage-v144.json');nv=read(O/'candidate-pair-navigation-v144.json');ck('173_422',cv['scoped_conditional_pairs']==173 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==422)
        a=read(O/'operator-analysis-coverage-ledger-v143.json');b=read(O/'operator-analysis-coverage-ledger-v144.json');ck('nativegaps',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        a=read(O/'candidate-profile-ledger-v143.json');b=read(O/'candidate-profile-ledger-v144.json');ck('broadgaps',[x.get('conclusion_changing_gap')for x in a['candidates']]==[x.get('conclusion_changing_gap')for x in b['candidates']])
        mm=read(O/'module-phase-branch-matrix-v144.json');ck('358_26',len(mm['modules'])==26 and sum(len(x['branches'])for x in mm['modules'])==358)
        for x in read(O/'candidate-record-index-v144.json')['model_candidates']:ck('master:'+x['id'],hashlib.sha256(Path(x['master_record']).read_bytes()).hexdigest()==x['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v144.json')
    for x in manifest['artifacts']:ck('manifest:'+x['path'],pin(x))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,frontiers=192,reused_native_TASK_union=5064,C2=72,C1=24,Index_only=96,pair_reference_cases=784,complement_changed_cases=640,DAGs=6144,chosen_future_deadline_violations=855,new_scoped_pairs=6,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-norm-index-state-preflight-v144.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
