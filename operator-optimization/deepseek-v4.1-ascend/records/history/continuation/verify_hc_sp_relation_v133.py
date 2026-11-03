"""Native/source keys and conditional HC/SP graph checks; no performance acceptance."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,T,L,read,write
from aux_sp_models_v131 import cycle


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pinned(p):
        b=Path(p['path']).read_bytes();return len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256']
    for p in read(O/'hc-sp-inputs-v133.json')['inputs']:ck('input:'+p['path'],pinned(p))
    source=read(O/'hc-sp-source-and-branch-contract-v133.json')
    for p in source['parts']:ck('source:'+str((p['path'],p['first_line'])),pinned(p) and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    src='\n'.join('\n'.join(p['source'])for p in source['parts']);ck('premix_source_choice','tilingData->hasPreMix != 0 ? preMixLocal : mixes01ReduceLocal'in src and 'pre_mix = hidden_states.new_zeros'in src and 'hidden_states[:n] = layer.engram'in src)
    prof=read(O/'hc-sp-native-frontiers-v133.json');bydb=defaultdict(list);seen=set();seen_payload={};sp_seen=set();counts=Counter();guards=Counter()
    for r in prof['records']:bydb[r['database_index']].append(r)
    old={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in read(O/'HC-attention-fork-native-frontiers-v93.json')['frontiers']}
    olddraft={(r['database_index'],r['source_order_ordinal']):r for r in read(O/'context-sp-profile-frontiers-v130.json')['records']}
    edge={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in read(O/'HC-operator-profile-and-edges-v58.json')['exact_graph_role_edges']if r['kind']=='CROSSLAYER_POST_TO_PRE'}
    rawrefs={r['database_index']:r for r in read(O/'host-launch-original-API-coverage-v126.json')['databases']}
    for ref in prof['databases']:
        di=ref['database_index'];ck('cachepin:'+str(di),pinned(ref));raw=ref['raw_database'];ck('rawpin:'+str(di),pinned(raw) and raw['sha256']==rawrefs[di]['sha256'])
        db=sqlite3.connect(Path(raw['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};cidmap=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cidmap[x['id']].add(x['connectionId'])
        def native(s):
            t,d=s['TASK'],s['definition'];key=(di,t['task_record_rowid'])
            if key in seen:
                ck('reused_TASK_consistent:'+str(key),s==seen_payload[key]);return
            seen.add(key);seen_payload[key]=s;rr=db.execute('SELECT *FROM TASK WHERE rowid=?',(key[1],)).fetchone();dd=db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
            ck('TASK:'+str(key),all(rr[k]==t[k]for k in rr.keys()))
            for k,v in d.items():
                rawk=k[:-5]if k.endswith('_text')else k
                ck('COMPUTE:'+str((key,k)),(strings[dd[rawk]]if k.endswith('_text')else dd[rawk])==v)
            for a in s['APIs']:
                aa=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone()
                ck('API_CID:'+str((di,a['api_rowid'])),all((int(aa[k])==int(a[k])if k in ['startNs','endNs']else aa[k]==a[k])for k in aa.keys()if k in a) and t['connectionId']in cidmap[a['connectionId']])
            for f in s['selected_model_frames']:
                ff=db.execute('SELECT *FROM PYTORCH_CALLCHAINS WHERE id=? AND stackDepth=? AND stack=?',(f['id'],f['stackDepth'],f['stack'])).fetchone()
                ck('source_frame:'+str((key,f['stackDepth'])),ff is not None and strings[f['stack']]==f['value'] and any(a['callchainId']==f['id']for a in s['APIs']))
        for r in bydb[di]:
            g=r['SP_allGather'];key=(di,g['COMM_OP_rowid']);assert key not in sp_seen;sp_seen.add(key)
            gg=db.execute('SELECT *FROM COMMUNICATION_OP WHERE rowid=?',(key[1],)).fetchone();ck('SP_COMM:'+str(key),all(gg[k]==g[k]for k in ['count','rankSize','connectionId','startNs','endNs']) and gg['dataType']==11 and g['dtype']=='BFP16' and g['rankSize']==8)
            for s in r['stages'].values():native(s)
            for k in ['previous_FFN_Pre','previous_FFN_Post']:
                if r[k]:native(r[k])
            st=r['stages'];hc,norm,post,ffn=[st[k]['TASK']for k in ['HC','NORM','POST','NEXT_FFN_PRE']]
            ck('HC_norm_SP_Post_FFN_order:'+str(key),hc['endNs']<=norm['startNs']<=norm['endNs']<=g['startNs']<=g['endNs']<=post['startNs']<=post['endNs']<=ffn['startNs'])
            ck('external_mix_typed:'+str(key),st['HC']['definition']['inputDataTypes_text']=='DT_BF16;FLOAT;FLOAT;FLOAT;FLOAT' and st['HC']['definition']['inputShapes_text'].endswith(f';{g["count"]//5120},4"') and not r['exact_pointer_premix_values_or_new_handoff_bound'])
            if r['domain']=='TARGET_GRAPH':
                counts['graph']+=1;k=(di,r['graph_ordinal'],r['block_ordinal']);f=old[k];guards[r['previous_input_guard']]+=1
                ck('old_graph_keys:'+str(k),all(st[a]['TASK']['task_record_rowid']==f[b]for a,b in [('HC','attention_HcPre_TASK'),('NORM','input_norm_TASK'),('POST','attention_HcPost_TASK'),('NEXT_FFN_PRE','FFN_HcPre_TASK')]) and r['root_bound']==f['root_bound'] and r['SWA_role_unique']==f['SWA_role_unique'] and r['KV_BF16_candidates']==f['KV_BF16_candidates'])
                if k[2]:
                    prev=old[di,k[1],k[2]-1];x=edge[k]
                    ck('previous_pre_Post_generation:'+str(k),r['previous_FFN_Pre']['TASK']['task_record_rowid']==prev['FFN_HcPre_TASK'] and r['previous_FFN_Post']['TASK']['task_record_rowid']==x['producer_task_rowid'] and r['crosslayer_record']==x and r['previous_FFN_Post']['TASK']['endNs']<=hc['startNs'])
                    expected='AFTER_ENGRAM_WRITE'if k[2]in [1,14]else'AUX_READER_STILL_LIVE'if k[2]in [37,38,39]else'NO_INTERVENING_COMPUTE_RECORDED'
                    ck('intervening_guard:'+str(k),r['previous_input_guard']==expected and bool(x['intervening_compute_by_start'])==(k[2]in [1,14,37,38,39]))
                else:ck('firstmix_not_historical_value',r['previous_FFN_Pre']is None and r['previous_input_guard']=='INITIAL_PREMIX_SOURCE_ONLY')
            else:
                counts['draft']+=1;f=olddraft[di,r['source_order_ordinal']];j=r['draft_layer_ordinal'];oldl=f['layers'][j]
                ck('old_draft_keys',g==oldl['draft_SP_allGather'] and r['positive_query_KV']==oldl['query_KV'] and r['positive_metadata_None_zero']==oldl['dummy_zero_output_witness'])
                for name,line in [('HC',889),('NORM',896),('POST',902),('NEXT_FFN_PRE',905),('FFN_POST',919)]:ck('positive_model_role:'+str((key,name)),any(f'({line}): forward'in x['value']for x in st[name]['selected_model_frames']))
                if j:ck('draft_previous_generation',r['previous_FFN_Pre']['TASK']['endNs']<=r['previous_FFN_Post']['TASK']['startNs']<=r['previous_FFN_Post']['TASK']['endNs']<=hc['startNs'])
        db.close()
    ck('3200_312_80',(counts['graph'],counts['draft'],len(sp_seen))==(3200,312,3512) and sum(not r['SWA_role_unique']for r in prof['records']if r['domain']=='TARGET_GRAPH')==80)
    ck('160_240_2720',dict(guards)==prof['target_previous_input_guards']=={'INITIAL_PREMIX_SOURCE_ONLY':80,'AFTER_ENGRAM_WRITE':160,'AUX_READER_STILL_LIVE':240,'NO_INTERVENING_COMPUTE_RECORDED':2720})
    # Independently compute the four DAG completion times by topological service.
    def dag(r,split,project):
        dur={};deps={}
        for p,(c,y,j)in enumerate(zip(r['C'],r['Y'],r['J'])):
            dur['c'+str(p)]=c;dur['y'+str(p)]=y;deps['c'+str(p)]=[];deps['y'+str(p)]=[]
            dur['h'+str(p)]=0;deps['h'+str(p)]=['y'+str(p)]+([]if split else['c'+str(p)])
            dur['j'+str(p)]=j if project else 0;deps['j'+str(p)]=['h'+str(p)]
        dur['g']=r['Gk']if project else r['Gh'];deps['g']=['j'+str(p)for p in range(len(r['C']))]
        dur['a']=r['Ak']if project else r['Ar'];deps['a']=['g'];dur['join']=0;deps['join']=['a']+['c'+str(p)for p in range(len(r['C']))]
        done={}
        def end(n):
            if n not in done:done[n]=max([end(p)for p in deps[n]]or[0])+dur[n]
            return done[n]
        return end('join')
    model=read(O/'hc-sp-rank-fork-schedule-models-v133.json');classes=Counter();ck('9349models',len(model['cases'])==model['case_count']==9349)
    for i,r in enumerate(model['cases']):
        B,S,P,C=[dag(r,a,b)for a,b in [(False,False),(True,False),(False,True),(True,True)]]
        ck('schedule:'+str(i),(B,S,P,C)==tuple(r[k]for k in ['baseline','split_only','projection_only','combined']) and S<=B and C<=P)
        d=(B-C)-((B-S)+(B-P));classes['ADDITIVE'if d==0 else'APPARENT_SUPERADDITIVE'if d>0 else'OVERLAPPING_CREDITS']+=1
    ck('both_nonadditive_directions',dict(classes)==model['credit_classes'] and classes['OVERLAPPING_CREDITS']>0 and classes['APPARENT_SUPERADDITIVE']>0 and not model['hardware_optimum_or_measured_gain_proved'])
    for r in read(O/'hc-sp-interface-work-and-resource-cut-v133.json')['cases']:
        M=r['M'];ck('interface:'+str(M),r['raw_X_content']==40960*M and r['old_premix_content']==16*M and r['Y_content']==r['Z_content']==10240*M and r['current_pre_post_comb_content']==96*M and r['both_private_interface_budget']==40960*M and r['retained_coefficient_cast_workspace_value_IO_reference']==163840*M and not r['physical_HBM_allocation_or_necessary_bytes_established'])
    w=read(O/'hc-sp-reader-generation-and-row-witnesses-v133.json');ck('reader_retirement',max(w['input_retirement']['X_readers_last_service_units'].values())==w['input_retirement']['earliest_all_named_readers_retired']==12>w['input_retirement']['carrier_ready'])
    x=w['old_vs_current_premix'];ck('different_premix_generation',sum(a*b for a,b in zip(x['X'],x['old_premix']))==x['Y_old']==1 and sum(a*b for a,b in zip(x['X'],x['current_pre']))==x['Y_wrong_current']==2.5)
    e=w['branch_DAGs']['external'];n=w['branch_DAGs']['no_external'];ck('conditional_dependency',not cycle(e) and not cycle(n) and ['CURRENT_PRE','Y']not in e and ['CURRENT_PRE','Y']in n and ['POST_COMB','POST']in e and ['POST_COMB','Y']not in n)
    scope=read(O/'hc-sp-composition-scope-v133.json');ck('three_scoped_pairs',scope['new_pairs']==[['F01','F32'],['F02','F32'],['F03','F32']] and scope['new_native_nodes']==scope['new_groups']==0 and not scope['all_runtime_all_subsets_or_native_fusion_closed'])
    ck('960_eager_target_remain',scope['all_saved_attention_SP_calls']==4472 and scope['newly_joined_target_graph_frontiers']+scope['newly_joined_draft_frontiers']+scope['other_eager_target_SP_not_newly_joined']==4472 and scope['other_eager_target_SP_not_newly_joined']==960)
    if args.preflight:
        old=read(O/'candidate-relation-coverage-v132.json')
        for pair in scope['new_pairs']:ck('actually_pending:'+str(pair),next(r for r in old['pairs']if {r['a'],r['b']}==set(pair))['scope_state']!='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL')
        manifest=read(O/'checkpoint-manifest-v132.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v133.json');ck('open_progress',cp['status']=='PROGRESS' and not cp['overall_complete'] and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v133.json');ck('142_26',len(rel['clauses'])==len({tuple(sorted((r['a'],r['b'])))for r in rel['clauses']})==rel['clause_count']==rel['selected_pair_count']==142 and len(rel['higher_order_groups'])==26)
        for i,pair in enumerate(scope['new_pairs'],1):
            r=next(r for r in rel['clauses']if r['id']==f'REL133-{i:02d}');ck('conditional_relation:'+str(i),{r['a'],r['b']}==set(pair) and not r['all_runtime_gates_proved'] and r['critical_path_gain_us']is None)
        g=next(g for g in rel['higher_order_groups']if g['id']=='HG93-HC-ATTENTION-FORK-DEADLINE');ck('group_member_added_not_closed','F32'in g['members'] and not g['all_subsets_analyzed'] and not g['legal_joint_fusion_proved'])
        cov=read(O/'candidate-relation-coverage-v133.json');nav=read(O/'candidate-pair-navigation-v133.json');ck('142_453',cov['scoped_conditional_pairs']==142 and cov['remaining_pair_relations']==nav['remaining_central_pair_reconciliation']==453)
        fp=read(O/'candidate-operator-footprints-v133.json');fm={r['id']:set(r['modules'])for r in fp['candidates']}
        ck('draft_F01_module_link','M19'in fm['F01'] and 'F01'in next(r for r in read(O/'module-profile-ledger-v133.json')['modules']if r['id']=='M19')['candidate_links'])
        for row in nav['pairs']:ck('module_pair_association:'+str((row['a'],row['b'])),set(row['common_module_associations'])==fm[row['a']]&fm[row['b']])
        m=read(O/'module-phase-branch-matrix-v133.json');ck('358_26',len(m['modules'])==26 and sum(len(r['branches'])for r in m['modules'])==358)
        a=read(O/'operator-analysis-coverage-ledger-v132.json');b=read(O/'operator-analysis-coverage-ledger-v133.json');ck('native_gaps_unchanged',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v133.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE' and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v133.json')
    for p in manifest['artifacts']:ck('manifest:'+p['path'],pinned(p))
    out=dict(status='PASS',checks=len(checks),preflight=args.preflight,target_frontiers=3200,draft_frontiers=312,schedule_cases=9349,new_scoped_pairs=3,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-sp-preflight-v133.json',out)
    print(json.dumps(out))


if __name__=='__main__':main()
