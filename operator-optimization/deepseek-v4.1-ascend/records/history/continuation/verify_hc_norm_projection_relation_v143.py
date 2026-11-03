"""Independent raw attribution and typed normalization/carrier model checks."""
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
    for x in read(O/'hc-norm-projection-inputs-v143.json')['inputs']:ck('input:'+x['path'],pin(x))
    for x in read(O/'hc-norm-projection-source-contract-v143.json')['parts']:ck('source:'+str((x['path'],x['first_line'])),pin(x)and Path(x['path']).read_text().splitlines()[x['first_line']-1:x['last_line']]==x['source'])
    p=read(O/'hc-norm-projection-native-frontiers-v143.json');parent={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sp-eager-native-frontiers-v134.json')['records']};ropes={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-rope-latent-native-frontiers-v140.json')['query_records']};indices={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-index-query-weight-native-producers-v139.json')['records']};bydb=defaultdict(list);seen=set();roles=Counter();geom=Counter();flags=Counter();holes=[];oldQR=0
    c1={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-c1-native-consumers-v135.json')['records']};c2={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-c2-native-consumers-v136.json')['records']}
    for r in p['records']:bydb[r['database_index']].append(r)
    for ref in p['databases']:
        di=ref['database_index'];ck('rawpin',pin(ref['raw_database']));ck('cachepin',pin(ref['cached_native']));db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};fs={(x['id'],x['stackDepth'],x['stack'])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};cids=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cids[x['id']].add(x['connectionId'])
        aa={x['api_rowid']:x for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')};tt={x['task_record_rowid']:x for x in db.execute('SELECT rowid AS task_record_rowid,*FROM TASK')};dd={x['globalTaskId']:x for x in db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO')}
        def stage(s,tag):
            a,t,d=s['API'],s['TASK'],s['definition'];raw=aa[a['api_rowid']];ck('API:'+tag,all(a[k]==raw[k]for k in raw.keys())and a['name_text']==ss[raw['name']]);ff=s.get('selected_frames',s.get('source_frames'));ck('sourceframes:'+tag,bool(ff)and all((f['id'],f['stackDepth'],f['stack'])in fs and f['id']==a['callchainId']and f['value']==ss[f['stack']]for f in ff))
            if t is None:
                ck('APIonly:'+tag,d is None and s['evidence_grade']=='SOURCE_API_ONLY_NO_LINKED_NATIVE_TASK'and not s['time_nearest_repair']and not any(x['connectionId']in cids[a['connectionId']]for x in tt.values()));return
            raw=tt[t['task_record_rowid']];de=dd[t['globalTaskId']];ck('TASKCID:'+tag,all(t[k]==raw[k]for k in raw.keys())and a['api_rowid']in t['native_linked_API_rowids']and t['connectionId']in cids[a['connectionId']]);ck('COMPUTE:'+tag,all((ss[de[k[:-5]]]if k.endswith('_text')else de[k])==v for k,v in d.items()));key=di,t['task_record_rowid'];ck('unique:'+tag,key not in seen);seen.add(key)
        for r in bydb[di]:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];orig=parent[key];s=r['stages'];ck('parent',not r['actual_input_tensor_gamma_rstd_or_weight_epoch_identity']and r['SP_allGather']==orig['SP_allGather']and r['output_reduceScatter']==orig['output_reduceScatter']and s['HC']==orig['stages']['HC']and s['INPUT_NORM']==orig['stages']['NORM']and s['QA']==orig['QA']and s['SWA_KV']==orig['SWA_KV'])
            for k,v in ropes[key]['stages'].items():ck('reuseRoPE',s[k]==v)
            for role,z in s.items():
                stage(z,str((key,role)));roles[role]+=z['TASK']is not None
                if z['TASK']is None:holes.append(dict(key=list(key),role=role,API_rowid=z['API']['api_rowid']))
            for role,line in [('QA_NORM',477),('QA_QUANT',478)]:
                z=s[role];a=z['API'];ff=z['selected_frames'];ck('QRsource_scope',any(f'/attention/dsa_v41.py({line}):'in f['value']for f in ff)and a['globalTid']==orig['SP_allGather']['API']['globalTid']and int(orig['SP_allGather']['API']['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(orig['output_reduceScatter']['API']['startNs']))
                if key in indices:
                    old=indices[key]['producers']['QR_NORM'if role=='QA_NORM'else'QB_INPUT_QUANT'];ck('old192QRsame',z['API']==old['API']and z['TASK']==old['TASK']and all(old['definition'][k]==v for k,v in z['definition'].items()));oldQR+=1
            m=int(s['INPUT_NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]);q=int(s['QA']['definition']['inputShapes_text'].strip('"').split(',')[0]);n=int(s['SWA_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);geom[m,q,n]+=1
            ck('inputtypes',s['INPUT_NORM']['definition']['inputShapes_text']==f'"{m},5120;5120"'and s['INPUT_NORM']['definition']['outputShapes_text']==f'"{m},5120;{m},1"'and s['INPUT_NORM']['definition']['outputDataTypes_text']=='DT_BF16;FLOAT')
            ck('projectiontypes',s['QA']['definition']['inputShapes_text']==f'"{q},5120;1280,5120"'and s['QA']['definition']['inputDataTypes_text']=='DT_BF16;DT_BF16'and s['SWA_KV']['definition']['inputShapes_text']==f'"{n},5120;512,5120"')
            ck('QRtypes',s['QA_NORM']['definition']['inputShapes_text']==f'"{q},1280;1280"'and s['QA_NORM']['definition']['outputDataTypes_text']=='DT_BF16;FLOAT'and s['QA_QUANT']['definition']['inputShapes_text']==f'"{q},1280"'and s['QA_QUANT']['definition']['outputDataTypes_text']=='INT8;FLOAT')
            ck('QBtoRoPE',s['QB']['definition']['inputDataTypes_text']=='INT8;INT8;DT_BF16;FLOAT'and s['QB']['definition']['outputShapes_text']==f'"{q},32768"'and s['QUERY_ROPE']['definition']['outputShapes_text']==f'"{q},1,64,512"'and s['SWA_NORM']['definition']['inputShapes_text']==f'"{n},512;512"')
            ck('currentstageorder',s['QA']['TASK']['endNs']<=s['QA_NORM']['TASK']['startNs']<=s['QA_NORM']['TASK']['endNs']<=s['QA_QUANT']['TASK']['startNs']<=s['QA_QUANT']['TASK']['endNs']<=s['QB']['TASK']['startNs']and s['QUERY_ROPE']['TASK']['startNs']>=max(s['QB']['TASK']['endNs'],s['SWA_STORE']['TASK']['endNs']))
            if 'C1_PROJ'in s:
                ck('C1typed',s['C1_PROJ']==orig['C1']and s['C1_NORM']==c1[key]['consumers']['C1_NORM']and s['C1_PROJ']['definition']['inputShapes_text']==f'"{n},5120;512,5120"'and s['C1_NORM']['definition']['inputShapes_text']==f'"{n},512;512"');flags['C1']+=1
            if 'C2_KV'in s:
                ck('C2typed',s['C2_INPUT_CAST']==c2[key]['consumers']['INPUT_CAST']and s['C2_KV']==orig['C2_KV']and s['C2_SCORE']==orig['C2_SCORE']and s['C2_INPUT_CAST']['definition']['inputDataTypes_text']=='DT_BF16'and s['C2_INPUT_CAST']['definition']['outputDataTypes_text']=='FLOAT'and s['C2_KV']['definition']['inputDataTypes_text']==s['C2_SCORE']['definition']['inputDataTypes_text']=='FLOAT;FLOAT');flags['C2']+=1
            flags['INDEX_WEIGHT']+=key in indices
        db.close()
    ck('10824bindings',len(seen)==p['TASK_bindings']==10824 and roles==p['native_role_counts']and holes==p['API_only_roles']==[]and all(roles[k]==960 for k in ['HC','INPUT_NORM','QA','QA_NORM','QA_QUANT','QB','QUERY_ROPE','SWA_KV','SWA_NORM','SWA_ROPE','SWA_STORE'])and roles['C1_PROJ']==roles['C1_NORM']==24 and roles['C2_INPUT_CAST']==roles['C2_KV']==roles['C2_SCORE']==72 and oldQR==384 and p['new_QR_source_roles']==1920)
    ck('960geometry',len(p['records'])==960 and geom=={(800,800,6400):320,(816,816,6528):320,(834,834,6668):280,(834,830,6668):40}and flags==dict(C1=24,C2=72,INDEX_WEIGHT=192))
    obs=read(O/'hc-norm-projection-observed-and-work-v143.json');ck('readercohorts',obs['other_Z_positive_reader_frontiers']==dict(flags))
    for x in obs['records']:
        key=tuple(x['key']);orig=parent[key];ck('observations',x['QA_norm_native_bound']and x['QA_quant_native_bound']and x['C1']==bool(orig['C1'])and x['C2']==bool(orig['C2_KV'])and x['IndexWeight_positive']==(key in indices)and not x['normalizer_rstd_pointer_or_actual_same_Z_subrows_bound'])
    for x in obs['geometry']:
        m,q,n,h=x['M'],x['Q'],x['N'],x['H'];ck('work_interfaces',x['frontiers']==geom[m,q,n]and x['owner_input_norm_rows']==m and x['recipient_union_norm_rows']==n and x['separate_recipient_QA_KV_norm_rows']==n+q and x['recipient_extra_norm_rows']==n-m and x['current_Y_private_interface']==x['current_Z_sender_private_interface']==x['owner_HC_RMS_chosen_interface_credit']==4*m*h and x['current_global_Z_union_interface']==4*n*h+2*q*h and x['rawY_recipient_union_chosen_interface_credit']==4*m*h+2*q*h and x['joint_QA_KV_shared_row_input_read_reference']==2*q*h and x['rawY_recipient_full_row_interface']==4*n*h and x['QA_fullN_extra_classical_ops']==2*(n-q)*h*1280 and x['QA_KV_owner_classical_ops']==2*m*h*1792 and x['QA_KV_current_classical_ops']==2*h*(q*1280+n*512)and not x['all_Z_other_readers_closed']and x['changed_normalizer_plane_or_actual_HBM_saved']is None)
    w=read(O/'hc-norm-projection-stat-and-round-witnesses-v143.json');x=w['HCraw_not_collapsed']
    def collapse(xx,pp):return [sum(pp[i]*xx[i][j]for i in range(4))for j in range(2)]
    ck('rawstat_not_collapsed',sum(a*a for row in x['Xplus']for a in row)/8==sum(a*a for row in x['Xminus']for a in row)/8==x['raw_mean_both']==.25 and collapse(x['Xplus'],x['premix'])==x['Yplus']and collapse(x['Xminus'],x['premix'])==x['Yminus']and sum(a*a for a in x['Yplus'])/2==x['collapsed_mean_plus']==2 and x['collapsed_mean_minus']==0)
    x=w['initial_repeat_stat'];ck('restricted_initialstat',sum(a*a for row in x['repeat']for a in row)/8==sum(a*a for a in collapse(x['repeat'],x['premix']))/2==x['raw_mean']==x['collapsed_mean']==6.5 and not x['loaded_native_rstd_abi_or_reduction_equality_attested'])
    x=w['real_RMS_linear_RMS'];r=1/math.sqrt(.5+x['eps1']);v=sum(a*b for a,b in zip(x['Y'],x['B']));correct=r*v/math.sqrt((r*v)**2+x['eps2']);skip=v/math.sqrt(v*v+x['eps2']);adapt=v/math.sqrt(v*v+x['eps2']/(r*r));ck('adaptive_epsilon',r==x['r']and v==x['v']and correct==x['correct']and skip==x['illegal_delete_input_scalar']and abs(adapt-correct)<1e-14 and abs(adapt-x['adaptive_epsilon_result'])<1e-14 and abs(correct-skip)>.04)
    def f32(x):return struct.unpack('>f',struct.pack('>f',x))[0]
    def bf16(x):
        u=struct.unpack('>I',struct.pack('>f',x))[0];return struct.unpack('>f',struct.pack('>I',(u+32767+((u>>16)&1))&0xffff0000))[0]
    x=w['BF16_Z_GEMM'];y=x['Y_prefix'];mu=f32(sum(f32(a*a)for a in y)/x['H']);r=f32(1/math.sqrt(f32(mu+x['eps1'])));zf=[f32(a*r)for a in y];zb=list(map(bf16,zf));a=f32(f32(zb[0]*x['B_BF16_prefix'][0])+f32(zb[1]*x['B_BF16_prefix'][1]));b=f32(f32(zf[0]*x['B_BF16_prefix'][0])+f32(zf[1]*x['B_BF16_prefix'][1]));ck('Zround_fence',mu==x['mean_f32']and r==x['rstd_f32']and zf==x['Z_FLOAT_prefix']and zb==x['Z_BF16_prefix']and a==x['projection_from_rounded_Z']==-.07421875 and b==x['projection_from_unrounded_Z']==0 and not x['actual_frozen_weights_or_native_gemm_policy'])
    x=w['prefix_not_fullH'];ck('fullHstat',sum(a*a for a in x['prefix']+x['suffixA'])/4==x['meanA']and sum(a*a for a in x['prefix']+x['suffixB'])/4==x['meanB']and x['first_normalized_A']!=x['first_normalized_B']and not x['provisional_or_correction_recompute_algorithms_excluded'])
    x=w['gamma_cannot_commute'];raw=[sum(a*b for a,b in zip(row,x['B']))for row in x['Ycases']];correct=[sum(a*g*b for a,g,b in zip(row,x['gamma'],x['B']))for row in x['Ycases']];ck('feature_gamma',raw==x['raw_projection']==[1,1,0]and correct==x['correct_after_feature_gamma']==[1,2,-1]and not x['universal_scalar_gamma_exists']and x['real_gamma_fold_into_weights_allowed'])
    def dag(p,carrier,early,price):
        nd={'y':(p['hc_y'],[]),'c':(p['hc_coeff'],[]),'hc':(0,['y','c'])}
        if carrier=='baseline':nd.update(norm=(p['owner_norm']+4,['hc']),g=(p['gather'],['norm']))
        elif carrier=='owner_fused':nd.update(norm=(p['owner_norm']+(p['extra']if price else 0),['hc']),g=(p['gather'],['norm']))
        else:nd.update(rawg=(p['gather'],['hc']),g=(p['recipient_norm']+(p['extra']if price else 0),['rawg']))
        nd.update(qa=(p['qa'],['g']),kv=(p['kv'],['qa']),qr=(p['qr_quant']+(p['extra']if price and early else 0),['qa']),qb=(p['qb'],['qr','kv']),kt=(p['kv_tail']+(p['extra']if price and early else 0),['kv']if early else['kv','qr']),rope=(2,['qb','kt']),post=(3,['rope','c']));memo={}
        def end(k):
            if k not in memo:memo[k]=nd[k][0]+max([end(z)for z in nd[k][1]]or[0])
            return memo[k]
        end('post');return memo
    tm=read(O/'hc-norm-projection-readiness-DAGs-v143.json');classes=Counter()
    for i,r in enumerate(tm['cases']):
        for k,c,e,pr in [('baseline','baseline',False,False),('owner_fused','owner_fused',False,True),('early_KV_only','baseline',True,True),('owner_joint','owner_fused',True,True),('recipient_joint','rawY',True,True)]:
            d=dag(r['parameters'],c,e,pr);ck('DAG:'+str((i,k)),all(d[node]==r[k][role]for role,node in [('QA','qa'),('KV','kv'),('QR_DQ','qr'),('QB','qb'),('KV_PUBLISHED','kt'),('Query_RoPE','rope'),('Post_input','post')]))
        a,b=r['recipient_joint']['Post_input'],r['baseline']['Post_input'];classes['RECIPIENT_FASTER'if a<b else'RECIPIENT_SLOWER'if a>b else'TIED']+=1
    ck('DAG_scope',len(tm['cases'])==tm['count']==3072 and dict(classes)==tm['recipient_joint_classes']==dict(RECIPIENT_FASTER=528,RECIPIENT_SLOWER=2544)and tm['not_physical_lower_or_upper_bound'])
    comp=read(O/'hc-norm-projection-composition-v143.json');ck('5_pairs',comp['new_pairs']==[['F02',x]for x in ['F05','F06','F07','F08','F33']]and comp['new_pair_clauses']==5 and not comp['all_runtime_gates_proved']and not comp['whole_pair_or_all_subsets_closed'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v142.json')
        for pair in comp['new_pairs']:ck('pending:'+str(pair),not any({x['a'],x['b']}==set(pair)for x in rel['clauses']))
        for gid,m in comp['group_extensions']:ck('newgroup:'+str((gid,m)),m not in next(g for g in rel['higher_order_groups']if g['id']==gid)['members'])
        manifest=read(O/'checkpoint-manifest-v142.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v143.json');ck('notdone',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0);rel=read(O/'candidate-relation-clauses-v143.json');ck('167_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==167 and len({tuple(sorted((x['a'],x['b'])))for x in rel['clauses']})==167 and len(rel['higher_order_groups'])==26)
        for c in comp['clauses']:
            x=next(r for r in rel['clauses']if r['id']==c['id']);ck('conditional', {x['a'],x['b']}=={c['a'],c['b']}and not x['all_runtime_gates_proved']and x['critical_path_gain_us']is None)
        for gid,m in comp['group_extensions']:
            x=next(g for g in rel['higher_order_groups']if g['id']==gid);ck('groupsopen',m in x['members']and not x['all_subsets_analyzed']and not x['legal_joint_fusion_proved'])
        cv=read(O/'candidate-relation-coverage-v143.json');nv=read(O/'candidate-pair-navigation-v143.json');ck('167_428',cv['scoped_conditional_pairs']==167 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==428)
        a=read(O/'operator-analysis-coverage-ledger-v142.json');b=read(O/'operator-analysis-coverage-ledger-v143.json');ck('nativegaps',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        a=read(O/'candidate-profile-ledger-v142.json');b=read(O/'candidate-profile-ledger-v143.json');ck('broadgaps',[x.get('conclusion_changing_gap')for x in a['candidates']]==[x.get('conclusion_changing_gap')for x in b['candidates']])
        mm=read(O/'module-phase-branch-matrix-v143.json');ck('358_26',len(mm['modules'])==26 and sum(len(x['branches'])for x in mm['modules'])==358)
        for x in read(O/'candidate-record-index-v143.json')['model_candidates']:ck('master:'+x['id'],hashlib.sha256(Path(x['master_record']).read_bytes()).hexdigest()==x['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v143.json')
    for x in manifest['artifacts']:ck('manifest:'+x['path'],pin(x))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,frontiers=960,native_TASK_bindings=10824,QR_source_role_bindings=1920,old_QR_subset_rechecked=384,C1_frontiers=24,C2_frontiers=72,IndexWeight_reader_frontiers=192,DAGs=3072,new_scoped_pairs=5,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-norm-projection-preflight-v143.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
