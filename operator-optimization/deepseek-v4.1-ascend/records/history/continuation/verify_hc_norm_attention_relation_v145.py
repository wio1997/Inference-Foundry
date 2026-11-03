"""Verify native output/copy,collective labels,allownerrelease androwdomain accounting."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,sqlite3,struct
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(k,v):
        assert v,k
        checks.append(k)
    def pin(x):
        b=Path(x['path']).read_bytes();return len(b)==x['bytes']and hashlib.sha256(b).hexdigest()==x['sha256']
    for x in read(O/'hc-norm-attention-inputs-v145.json')['inputs']:ck('input:'+x['path'],pin(x))
    for x in read(O/'hc-norm-attention-source-contract-v145.json')['parts']:ck('source:'+str((x['path'],x['first_line'])),pin(x)and Path(x['path']).read_text().splitlines()[x['first_line']-1:x['last_line']]==x['source'])
    p=read(O/'hc-norm-attention-native-frontiers-v145.json');bydb=defaultdict(list);seen=set();roles=Counter();geom=Counter();labels=defaultdict(list);queue=0;obs=read(O/'hc-norm-attention-observed-and-work-v145.json');qproof=read(O/'communication-role-contract-v16.json')['queue_source'];receipt=json.loads(read(N/'unassigned-communication166-v16.json')['stdout'])
    for r in p['records']:bydb[r['database_index']].append(r)
    for ref in p['databases']:
        di=ref['database_index'];ck('rawpin',pin(ref['raw_database']));ck('cachepin',pin(ref['cached_native']));db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};fs={(x['id'],x['stackDepth'],x['stack'])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};dt={x['id']:x['name']for x in db.execute('SELECT *FROM ENUM_HCCL_DATA_TYPE')};ci=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):ci[x['id']].add(x['connectionId'])
        aa={x['api_rowid']:x for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')};tt={x['task_record_rowid']:x for x in db.execute('SELECT rowid AS task_record_rowid,*FROM TASK')};dd={x['globalTaskId']:x for x in db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO')}
        def api(a,tag):
            x=aa[a['api_rowid']];ck('API:'+tag,all(a[k]==x[k]for k in x.keys())and a['name_text']==ss[x['name']])
        for r in bydb[di]:
            k=di,r['forward_API_rowid'],r['source_block_ordinal'];s=r['stages'];m,q,t=r['carrier_sender_rows'],r['query_rows'],r['cache_rows'];g=int(s['WO_A']['definition']['inputShapes_text'].strip('"').split(',')[0]);geom[m,q,t,g]+=1
            for role,z in s.items():
                a,tr,d=z['API'],z['TASK'],z['definition'];tag=str((k,role));api(a,tag);raw=tt[tr['task_record_rowid']];de=dd[tr['globalTaskId']];ff=z.get('selected_frames',z.get('source_frames'));tk=di,tr['task_record_rowid'];ck('TASKCID:'+tag,all(tr[x]==raw[x]for x in raw.keys())and a['api_rowid']in tr['native_linked_API_rowids']and tr['connectionId']in ci[a['connectionId']]);ck('Compute:'+tag,all((ss[de[x[:-5]]]if x.endswith('_text')else de[x])==v for x,v in d.items()));ck('frames:'+tag,bool(ff)and all((f['id'],f['stackDepth'],f['stack'])in fs and f['id']==a['callchainId']and f['value']==ss[f['stack']]for f in ff));ck('unique:'+tag,tk not in seen);seen.add(tk);roles[role]+=1
            ck('padded_projectiontypes',g==8*m and s['WO_A']['definition']['inputShapes_text']==f'"{g},1,4096;1,4096,1024"'and s['WO_A']['definition']['outputDataTypes_text']=='DT_BF16'and s['WO_B']['definition']['inputShapes_text']==f'"{g},1024;5120,1024"'and s['WO_B']['definition']['opType_text']=='MatMulV3'and s['WO_B']['definition']['inputDataTypes_text']=='DT_BF16;DT_BF16'and s['OUTPUT_COPY']['definition']['inputShapes_text']==s['OUTPUT_COPY']['definition']['outputShapes_text']==f'"{g},5120"')
            ck('headpad_domain',('HEAD_ZERO'in s)==('HEAD_COPY'in s)==(q<m)and 'RS_ZERO'not in s and 'RS_CAT'not in s)
            if q<m:ck('headpadtypes',s['HEAD_ZERO']['definition']['outputShapes_text']==f'"{m},64,512"'and s['HEAD_COPY']['definition']['inputShapes_text']==s['HEAD_COPY']['definition']['outputShapes_text']==f'"{q},64,512"')
            for role,z in r['collectives'].items():
                api(z['API'],'comm'+str((k,role)));co=z['COMM'];raw=db.execute('SELECT *FROM COMMUNICATION_OP WHERE opId=?',(co['opId'],)).fetchone();la=r['recorded_labels'][role];ck('rawCOMM',dict(raw)==la['raw']and ss[raw['opName']]==la['opName_text']and ss[raw['groupName']]==co['group_name']==la['groupName_text']and dt[raw['dataType']]==co['dtype']=='BFP16'and raw['connectionId']==co['connectionId']and raw['count']==co['count']and raw['rankSize']==co['rankSize']==8)
                if 'queue_flow_proof'in z:
                    queue+=1;ck('queue_existing',z['queue_flow_proof']==qproof and di==8 and co['COMM_OP_rowid']==232 and co['connectionId']==44189 and z['API']['api_rowid']==44896 and z['API']['connectionId']is None and co['source_grade']=='REUSED_EXPLICIT_QUEUE_FLOW_PROOF')
                    qc=receipt['native_launch_context'][0];b=qc['queue_candidates'][0];e=b['frontend_candidates'][0];rawc=db.execute('SELECT rowid,*FROM CANN_API WHERE rowid=44189').fetchone();ck('queue_CANN',rawc['connectionId']==44189 and rawc['globalTid']==qc['CANN_API']['globalTid'])
                    for a in [b['Dequeue'],e['Enqueue']]:ck('queue_API',all(int(aa[a['api_rowid']][x])==int(a[x])for x in ['startNs','endNs','globalTid','connectionId'])and ss[aa[a['api_rowid']]['name']]==a['name'])
                    ck('queue_foreign_flow',ci[b['Dequeue']['connectionId']]==ci[e['Enqueue']['connectionId']]==set(qproof['decoded_flow'])=={1342919928}and e['same_process']and e['name_matches']and e['enqueue_precedes_dequeue']and (b['Dequeue']['globalTid']>>32)==(e['Enqueue']['globalTid']>>32))
                else:ck('commCID',co['connectionId']in ci[z['API']['connectionId']])
                rm=[dict(x)for x in db.execute('SELECT *FROM RANK_DEVICE_MAP')];ck('rankmap',la['rank_device_map']==rm and len(rm)==1);labels[role,co['group_name'],la['opName_text']].append(dict(database_index=di,forward_API_rowid=k[1],source_block_ordinal=k[2],rank=rm[0]['rankId'],COMM_OP_rowid=co['COMM_OP_rowid'],count=co['count']))
            ck('commgeometry',r['collectives']['SP_INPUT']['COMM']['count']==r['collectives']['OUTPUT_RS']['COMM']['count']==m*5120 and r['collectives']['HEAD_A2A']['COMM']['count']==m*8*512)
        db.close()
    ck('10280native',len(seen)==p['source_CID_TASK_bindings']==10280 and roles==p['native_role_counts']and roles['OUTPUT_COPY']==roles['WO_A']==roles['WO_B']==960 and roles['HEAD_ZERO']==roles['HEAD_COPY']==40 and queue==obs['reused_queue_flow_A2A']==1 and obs['direct_CID_A2A']==959 and p['new_output_roles']==2960)
    ck('domains',geom=={(800,800,6400,6400):320,(816,816,6528,6528):320,(834,834,6668,6672):280,(834,830,6668,6672):40})
    for x in p['recorded_label_cohorts']:
        kk=x['role'],x['group_name'],x['opName'];ck('labelcohort',sorted(x['members'],key=lambda v:(v['database_index'],v['forward_API_rowid']))==sorted(labels[kk],key=lambda v:(v['database_index'],v['forward_API_rowid']))and len(x['members'])==len({v['rank']for v in x['members']})==8 and len({v['source_block_ordinal']for v in x['members']})==1 and not x['request_tensor_identity_or_exporter_tag_global_semantics_proved']and not x['crossrank_timestamp_join_used'])
    ck('labelscope',len(labels)==p['label_cohort_count']==360 and p['collective_records']==2880 and p['clock_or_formal_crossrank_phase_promotions']==0)
    for x in obs['geometry']:
        m,q,t,g,h=x['M'],x['Q'],x['T'],x['G'],x['H'];hr=65536;ck('logicalwork',x['frontiers']==geom[m,q,t,g]and x['global_Z_materialized_content']==2*g*h and x['current_global_Z_QA_KV_interface']==2*h*(g+q+t)and x['legacy_v143_T_clipped_reference']==4*t*h+2*q*h and x['global_materialization_correction']==2*h*(g-t)and x['rawY_union_receiver_interface']==2*h*(g+t)and x['relative_QA_shared_read_credit_unchanged']==2*q*h and x['HEAD_A2A_remote_value_content']==14*m*4096 and x['head_send_buffer_content']==m*hr and x['received_head_buffer_content']==g*8192 and x['WO_B_copy_private_interface_reference']==x['remaining_stable_output_RS_interface']==4*g*h and x['combined_output_interface_union_reference']==8*g*h and x['WO_A_WO_B_classical_ops']==2*g*(4096*1024+1024*5120)and x['observed_RS_zero_cat_credit']==0 and x['actual_HBM_wire_allocation_or_gain']is None)
        if q<m:ck('padpack_union',x['head_pad_pack_old_selected_IO']==(3*m+2*q)*hr and x['head_pad_pack_direct_selected_IO']==(m+q)*hr)
    def ownerdag(p,f,pc,dc):
        nd={};P=8
        for r in range(P):nd['o'+str(r)]=(max(p['Y'][r],p['C'][r])+p['norm']+(p['extra']if f else 2),[])
        for r in range(P):
            if pc:
                ns=[r]if p['query_owners']=='ALIGNED'else list(range(P))
                for k in range(P):nd['jpQ'+str(k)]=(1+k%3+p['extra'],['o'+str(k)]);nd['jpK'+str(k)]=(2+k%2+p['extra'],['o'+str(k)])
                nd['q'+str(r)]=(p['Gp']+r%2+p['query'],['jpQ'+str(k)for k in ns]);nd['k'+str(r)]=(p['Gp']+r%2+p['kv'],['jpK'+str(k)for k in range(P)])
            else:nd['z'+str(r)]=(p['Gz']+r%2,['o'+str(k)for k in range(P)]);nd['q'+str(r)]=(p['query'],['z'+str(r)]);nd['k'+str(r)]=(p['kv'],['z'+str(r)])
            nd['s'+str(r)]=(8+(p['late_head']if r==p['slow_owner']else 0),['q'+str(r),'k'+str(r)])
        for r in range(P):nd['head'+str(r)]=(2+r%3,['s'+str(k)for k in range(P)]);nd['a'+str(r)]=(3+r%2,['head'+str(r)]);nd['b'+str(r)]=(4,['a'+str(r)]);nd['out'+str(r)]=(p['extra']if dc else p['copy'],['b'+str(r)])
        for r in range(P):nd['rs'+str(r)]=(2+r%2,['out'+str(k)for k in range(P)])
        done={}
        def end(k):
            if k not in done:done[k]=nd[k][0]+max([end(z)for z in nd[k][1]]or[0])
            return done[k]
        return dict(owner_Z=[end('o'+str(r))for r in range(P)],Query=[end('q'+str(r))for r in range(P)],KV_published=[end('k'+str(r))for r in range(P)],head_send=[end('s'+str(r))for r in range(P)],head_receive=[end('head'+str(r))for r in range(P)],WO_A=[end('a'+str(r))for r in range(P)],WO_B=[end('b'+str(r))for r in range(P)],stable_output=[end('out'+str(r))for r in range(P)],RS=[end('rs'+str(r))for r in range(P)],Post_input=[max(end('rs'+str(r)),p['C'][r])for r in range(P)])
    tm=read(O/'hc-norm-attention-owner-DAGs-v145.json');cl=Counter()
    for i,x in enumerate(tm['cases']):
        for name,f,pc,dc in [('baseline',False,False,False),('owner_norm_fused',True,False,False),('projected_only',False,True,False),('copy_direct_only',False,False,True),('joint',True,True,True)]:
            d=ownerdag(x['parameters'],f,pc,dc);ck('ownerDAG:'+str((i,name)),all(x[name][k]==v for k,v in d.items())and x[name]['end']==max(d['Post_input']))
        a,b=x['joint']['end'],x['baseline']['end'];cl['FASTER'if a<b else'SLOWER'if a>b else'TIED']+=1
    ck('3072ownerDAG',tm['count']==len(tm['cases'])==3072 and dict(cl)==tm['classes']==dict(FASTER=640,SLOWER=2432)and not tm['physical_bound_or_actual_speedup'])
    old=read(O/'hc-norm-index-state-readiness-DAGs-v144.json');rr=read(O/'hc-norm-attention-C2-release-refinement-v145.json');counts=Counter()
    for x in rr['cases']:
        src=old['cases'][x['old_case_index']];p0=src['parameters'];z=x['consumer_Z_ready'];r=x['joint_with_explicit_Z_release'];q=z+p0['query']+p0['extra'];cast=max(z,p0['metadata'])+1;proj=cast+p0['projection']+p0['extra'];oldr=max(proj,p0['old_generation'])+1;pool=oldr+p0['pool'];ring=oldr+p0['ring'];ep=pool+p0['latent_norm']+p0['key_long'];qli=max(q+2,ep)+2;post=max(p0['hc_coeff'],q+3,ep+3,qli+3);ck('C2allowner_release',z==max(x['owner_Z_ready'])+x['carrier_service']and z>=x['legacy_assumed_Z_ready']and r['PROJ']==proj and r['Post_input']==post and r['RING']==ring and r['future_ring_deadline_satisfied']==(ring<=post+p0['future_gap']))
        z0=x['legacy_assumed_Z_ready'];oq=z0+p0['query']+p0['extra'];oc=max(z0,p0['metadata'])+1;op=oc+p0['projection']+p0['extra'];orr=max(op,p0['old_generation'])+1;opl=orr+p0['pool'];org=orr+p0['ring'];oep=opl+p0['latent_norm']+p0['key_long'];oqli=max(oq+2,oep)+2;opost=max(p0['hc_coeff'],oq+3,oep+3,oqli+3)
        ck('zero_added_release_certificate',src['joint_repriced']['Z']==z0 and src['joint_repriced']['PROJ']==op and src['joint_repriced']['RING']==org and src['joint_repriced']['Post_input']==opost)
        delta=post-src['joint_repriced']['Post_input'];counts['LATER'if delta>0 else'UNCHANGED'if delta==0 else'EARLIER']+=1
    ck('6144scope_refined',len(rr['cases'])==rr['count']==rr['zero_added_release_reproduces_old_cases']==6144 and dict(counts)==rr['consumer_release_classes']==dict(LATER=3032,UNCHANGED=3112))
    w=read(O/'hc-norm-attention-map-release-and-score-witnesses-v145.json');x=w['same_counts_wrong_head_map'];out=[[[x['send'][r][t][g*x['local_heads']+h][d]for d in range(x['D'])]for h in range(x['local_heads'])]for g in range(x['P'])for r in range(x['P'])for t in range(x['M'])];ck('headmap',out==x['recv_rank_major']and out!=x['wrong_receiver_rows'])
    def f32(v):return struct.unpack('>f',struct.pack('>f',v))[0]
    x=w['input_round_changes_positive_dot_TopK'];score=lambda q:[f32(f32(q[0]*k[0])+f32(q[1]*k[1]))for k in x['keys']];a,b=score(x['unrounded_query']),score(x['rounded_query']);ck('scoreorder',a==x['unrounded_scores']and b==x['rounded_scores']and a[0]>a[1]>0 and b[1]>b[0]>0 and not x['current_native_QLI_quant_scores_or_frozen_values'])
    x=w['localZ_not_consumerZ'];ck('local_not_allowner',max(x['owner_Z'])+x['carrier_service']==x['correct_fullcarrier_consumerZ']==35 and x['wrong_localZ_projection_start']==1)
    x=w['crosslayer_row_pipeline'];full=max(max(r)for r in x['partial_ready_by_headgroup'])+1;rs=[max(r[j]for r in x['partial_ready_by_headgroup'])+1 for j in range(2)];ck('crosslayer_tile',full==x['current_full_RS_allowners']and rs==x['chosen_tile_RS']and max(full+t for t in x['FFN_next_ready_service'])==x['current_next_fullcarrier_gate']==201 and max(a+b for a,b in zip(rs,x['FFN_next_ready_service']))==x['tile_next_fullcarrier_gate']==103 and not x['native_partial_publish_or_gain'])
    comp=read(O/'hc-norm-attention-composition-v145.json');ck('fivepairs',comp['new_pairs']==[['F02',x]for x in ['F12','F13','F14','F15','F16']]and not comp['all_runtime_gates_proved'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v144.json')
        for pair in comp['new_pairs']:ck('pending',not any({x['a'],x['b']}==set(pair)for x in rel['clauses']))
        for gid,m in comp['group_extensions']:ck('newgroup',m not in next(g for g in rel['higher_order_groups']if g['id']==gid)['members'])
        man=read(O/'checkpoint-manifest-v144.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v145.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0);rel=read(O/'candidate-relation-clauses-v145.json');ck('178_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==178 and len({tuple(sorted((x['a'],x['b'])))for x in rel['clauses']})==178 and len(rel['higher_order_groups'])==26)
        for c in comp['clauses']:
            x=next(r for r in rel['clauses']if r['id']==c['id']);ck('conditional',{x['a'],x['b']}=={c['a'],c['b']}and not x['all_runtime_gates_proved']and x['critical_path_gain_us']is None)
        cv=read(O/'candidate-relation-coverage-v145.json');nv=read(O/'candidate-pair-navigation-v145.json');ck('178_417',cv['scoped_conditional_pairs']==178 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==417)
        a=read(O/'operator-analysis-coverage-ledger-v144.json');b=read(O/'operator-analysis-coverage-ledger-v145.json');ck('nativegaps',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        a=read(O/'candidate-profile-ledger-v144.json');b=read(O/'candidate-profile-ledger-v145.json');ck('broadgaps',[x.get('conclusion_changing_gap')for x in a['candidates']]==[x.get('conclusion_changing_gap')for x in b['candidates']]);mm=read(O/'module-phase-branch-matrix-v145.json');ck('358_26',len(mm['modules'])==26 and sum(len(x['branches'])for x in mm['modules'])==358)
        for x in read(O/'candidate-record-index-v145.json')['model_candidates']:ck('master',hashlib.sha256(Path(x['master_record']).read_bytes()).hexdigest()==x['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');man=read(O/'checkpoint-manifest-v145.json')
    for x in man['artifacts']:ck('manifest:'+x['path'],pin(x))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,frontiers=960,native_TASK_bindings=10280,new_output_role_bindings=2960,collective_records=2880,recorded_label_cohorts=360,reused_queue_flow_A2A=1,owner_DAG_cases=3072,C2_release_refined_cases=6144,new_scoped_pairs=5,new_native_or_phase_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-norm-attention-preflight-v145.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
