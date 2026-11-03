"""Verify composed operator identities and bounded reference claims, not just report shape."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=0
 def ck(k,v):
  nonlocal checks
  checks+=1
  assert v,k
 def pin(z):
  p=Path(z['path']);return p.stat().st_size==z['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==z['sha256']
 for z in read(O/'post-projection-inputs-v151.json')['inputs']:ck('input_pin',pin(z))
 a=read(O/'hc-norm-projection-native-frontiers-v143.json');b=read(O/'post-pre-native-frontiers-v147.json');x=read(O/'post-projection-native-joins-v151.json');u=read(O/'post-projection-interface-unions-v151.json');w=read(O/'post-projection-version-and-union-witnesses-v151.json');rw=read(O/'post-projection-work-and-storage-domains-v151.json');co=read(O/'post-projection-composition-v151.json');binding={(z['database_index'],z['TASK']['task_record_rowid']):z for z in x['bindings']};old={(z['database_index'],z['stages']['HC']['TASK']['task_record_rowid']):(i,z)for i,z in enumerate(a['records'])};actual={};bydb=defaultdict(list)
 for z in x['bindings']:bydb[z['database_index']].append(z)
 for f in ['target_intra','target_cross','draft_intra','draft_cross']:
  for i,z in enumerate(b[f]):
   r=b['bindings'][z['right']];key=(z['database_index'],r['TASK']['task_record_rowid'])
   if key in old:actual[f,i]=old[key]
 ck('all_shared_HC_rebuilt',len(actual)==len(x['records'])==936 and{f for f,i in actual}=={'target_cross'})
 counts=Counter();geo=Counter();used=[];intra={(z['database_index'],z['forward_API_rowid'],z['block']):z for z in b['target_intra']};obs={tuple(z['key']):z for z in read(O/'hc-norm-projection-observed-and-work-v143.json')['records']}
 for z in x['records']:
  ei=z['post_pre_edge_index'];fi,q=actual['target_cross',ei];p=b['target_cross'][ei];di=z['database_index'];hc=b['bindings'][p['right']];o=obs[di,q['forward_API_rowid'],q['source_block_ordinal']];ck('same_exact_HC_and_context',fi==z['projection_frontier_index']and hc['TASK']==q['stages']['HC']['TASK']and hc['API']==q['stages']['HC']['API']and z['forward_API_rowid']==p['forward_API_rowid']==q['forward_API_rowid']and z['to_block']==p['to_block']==q['source_block_ordinal']and z['from_block']==p['from_block'])
  expected=dict(q['stages']);expected.update(PREV_FFN_POST=b['bindings'][p['left']],PREVIOUS_FFN_PREMIX_PROVIDER=b['bindings'][p['incoming_premix_provider']],FUTURE_ATTENTION_POST=b['bindings'][intra[di,z['forward_API_rowid'],z['to_block']]['left']]);ck('all_role_keys',set(expected)==set(z['native_role_keys']))
  for k,v in expected.items():
   key=(di,v['TASK']['task_record_rowid']);ck('exact_role_binding',z['native_role_keys'][k]==list(key)and binding[key]['TASK']==v['TASK']and binding[key]['API']==v['API']);used.append(key)
  ck('actual_geometry',tuple(z[k]for k in ['M','Q','N'])==tuple(o[k]for k in ['M','Q','N'])and z['H']==5120);ck('oldnew_flags',z['Engram_positive']==bool(p['Engram_positive_APIs'])and z['aux_mean_positive']==(p['aux_mean']is not None)and z['C1_positive']==o['C1']and z['C2_positive']==o['C2']and z['IndexWeight_positive']==o['IndexWeight_positive']);ck('no_native_tensor_or_phase_promotion',not z['actual_new_handoff_values_storage_parameter_or_phase_proved']and not z['absence_of_other_readers_inferred_from_no_positive_marker'])
  counts['Engram_positive']+=z['Engram_positive'];counts['aux_mean_positive']+=z['aux_mean_positive'];counts['C1']+=z['C1_positive'];counts['C2']+=z['C2_positive'];counts['IndexWeight_positive']+=z['IndexWeight_positive'];geo[z['M'],z['Q'],z['N']]+=1
 ck('role_uniqueness',len(used)==len(set(used))==len(binding)==x['unique_TASK_bindings']==x['role_bindings']==13368);ck('actual_positive_counts',counts==x['positive_counts']=={'Engram_positive':48,'aux_mean_positive':72,'C1':24,'C2':72,'IndexWeight_positive':192});ck('actual_geometries',geo=={(800,800,6400):312,(816,816,6528):312,(834,834,6668):273,(834,830,6668):39})
 for spec in x['databases']:
  di=spec['database_index'];d=sqlite3.connect('file:'+spec['raw_database']['path']+'?mode=ro',uri=True);d.row_factory=sqlite3.Row;ss=dict(d.execute('select id,value from STRING_IDS'));ts={z['task_record_rowid']:dict(z)for z in d.execute('select rowid AS task_record_rowid,* from TASK')};pa={z['api_rowid']:dict(z)for z in d.execute('select rowid AS api_rowid,* from PYTORCH_API')};cc={z['globalTaskId']:dict(z)for z in d.execute('select rowid AS compute_record_rowid,* from COMPUTE_TASK_INFO')};ci=defaultdict(set)
  for z in d.execute('select id,connectionId from CONNECTION_IDS'):ci[z[1]].add(z[0])
  for z in bydb[di]:
   t=z['TASK'];api=z['API'];r=ts[t['task_record_rowid']]
   for k,v in r.items():ck('raw_TASK_'+k,t[k]==v)
   ck('raw_TASK_type',ss[t['taskType']]==t['task_type_name']);p=pa[api['api_rowid']]
   for k,v in p.items():
    if k in api:ck('raw_API_'+k,int(api[k])==int(v)if k in ['startNs','endNs']else api[k]==v)
   ck('raw_API_name',api['name_text']==ss[p['name']]);ck('explicit_NATIVE_API_family',api['connectionId']in ci[t['connectionId']]);c=cc[t['globalTaskId']];saved=z['definition'];ck('COMPUTE_ID',saved['globalTaskId']==c['globalTaskId']==t['globalTaskId'])
   for k in ['opType','inputShapes','outputShapes','inputDataTypes','outputDataTypes']:
    key=k+'_text'
    if key in saved:ck('COMPUTE_'+k,saved[key]==ss[c[k]])
  d.close()
 seen=set();overcredits=0
 for z in u['cases']:
  key=(z['M'],z['Q'],z['N'],z['Engram'],z['aux'],z['C1'],tuple(z['actions']));ck('case_unique',key not in seen);seen.add(key);ports=z['baseline_ports'];removed=set(z['removed_ports']);a0=set(z['actions']);expected=set()
  if'UPDATED_X_PRE_FORWARD'in a0:expected.add('UPDATED_X_PRE_LOAD')
  if'Y_NORM_READ_FORWARD'in a0:expected.add('Y_NORM_LOAD')
  if'PRIVATE_Y_INTERFACE_INTERNAL'in a0:expected.update(['Y_STORE','Y_NORM_LOAD'])
  if'Z_SENDER_READ_FORWARD'in a0:expected.add('Z_SENDER_LOAD')
  if'QA_KV_ROW_INPUT_SHARE'in a0:expected.add('QA_Z_LOAD')
  if'C1_KV_ROW_INPUT_SHARE'in a0 and z['C1']:expected.add('C1_Z_LOAD')
  ck('union_not_additive',removed==expected and z['remaining_ports']=={k:v for k,v in ports.items()if k not in expected}and z['logical_union_credit_bytes']==sum(ports[k]for k in expected)and sum(ports.values())-sum(z['remaining_ports'].values())==z['logical_union_credit_bytes']and z['overcredit']==z['incorrect_additive_action_credit']-z['logical_union_credit_bytes']>=0)
  ck('raw_future_and_version_stores_retained',all(k in z['remaining_ports']for k in ['X0_STORE','UPDATED_X_FUTURE_POST_LOAD'])and z['updated_X_source']==('ENGRAM_UPDATED_X1'if z['Engram']else'POST_X0')and(not z['Engram']or all(k in z['remaining_ports']for k in ['ENGRAM_X0_LOAD','X1_STORE']))and(not z['aux']or'OLD_X0_AUX_LOAD'in z['remaining_ports']))
  ck('panel_domain_and_native_gates',z['applicable_in_declared_panel_domain']==(z['C1']or'C1_KV_ROW_INPUT_SHARE'not in a0)and not z['actual_native_gain_proved']);overcredits+=z['overcredit']>0
 ck('case_scope',len(seen)==u['count']==2048 and overcredits>0 and not u['native_gates_closed'])
 mean=lambda xx:[sum(r[j]for r in xx)/len(xx)for j in range(len(xx[0]))]
 v=w['Engram_old_new_versions'];ck('Engram_and_aux_counterexamples',mean(v['Post_X0'])==v['aux_expected_old_X0_mean']==v['incorrect_Pre_if_Engram_bypassed']and mean(v['Engram_updated_X1'])==v['Pre_expected_fixed_quarter_mix_X1']==v['incorrect_aux_from_X1']and v['incorrect_aux_from_X1']!=v['aux_expected_old_X0_mean']and not v['actual_native_parameter_or_pointer_binding']);v=w['collapsed_Y_not_full_raw'];ck('Y_not_residual_injective',v['raw_A']!=v['raw_B']and mean(v['raw_A'])==mean(v['raw_B'])==v['equal_quarter_mix_Y_A']==v['equal_quarter_mix_Y_B']and v['future_Post_residual_A']!=v['future_Post_residual_B']and not v['full_raw_reconstruction_from_Y_proved'])
 for v in rw['geometry']:
  M,Q,NN,H=(v[k]for k in ['M','Q','N','H']);ck('logical_common_prefix_count_once',v['PostPre_chosen_logical_input_port']==8*M*H and v['PostPre_plus_owner_private_Y_chosen_credit']==12*M*H and v['PostPre_plus_rawY_recipient_chosen_credit']==12*M*H+2*Q*H);ck('row_work_repricing',v['QA_full_N_extra_classical_ops']==2*(NN-Q)*H*1280 and v['recipient_union_norm_rows']-v['owner_norm_rows']==v['extra_union_norm_rows']==NN-M and v['all_actual_receiver_row_maps_precision_capacity_and_traffic_unknown'])
 ck('storage_domain',rw['storage_reference']['conditional_all_distinct_simultaneously_retained_bytes_per_row']==40960+10240+10240+4608==66048 and rw['actual_numeric_physical_ceiling_or_MFU_lowest']=='NOT_ESTABLISHED'and rw['logical_credits_not_HBM_or_time']);ck('five_scoped_pairs',len(co['clauses'])==5 and co['new_pairs']==[['F03',s]for s in ['F05','F06','F07','F08','F33']]and co['new_groups']==0 and not co['all_runtime_gates_proved']and not co['whole_pair_or_all_subsets_closed'])
 if args.preflight:
  man=read(O/'checkpoint-manifest-v150.json');oldrel=read(O/'candidate-relation-clauses-v150.json')
  for v in co['clauses']:ck('not_existing_pair',not any({z['a'],z['b']}=={v['a'],v['b']}for z in oldrel['clauses']))
 else:
  man=read(O/'checkpoint-manifest-v151.json');r=read(O/'REVIEW-CHECKPOINT-v151.json');ck('progress',r['status']=='PROGRESS'and not r['overall_complete']);rel=read(O/'candidate-relation-clauses-v151.json');cv=read(O/'candidate-relation-coverage-v151.json');ck('197_26_398',len(rel['clauses'])==cv['scoped_conditional_pairs']==197 and len(rel['higher_order_groups'])==26 and cv['remaining_pair_relations']==398)
  for gid,cid in co['group_extensions']:ck('group_scope_only',cid in next(z for z in rel['higher_order_groups']if z['id']==gid)['members']and not next(z for z in rel['higher_order_groups']if z['id']==gid)['all_subsets_analyzed'])
  a0=read(O/'operator-analysis-coverage-ledger-v150.json');a1=read(O/'operator-analysis-coverage-ledger-v151.json');ck('native_global_unchanged',a0['regimes']==a1['regimes']and all(a0[k]==a1[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']));a0=read(O/'candidate-profile-ledger-v150.json');a1=read(O/'candidate-profile-ledger-v151.json');ck('broader_gaps_preserved',[z.get('conclusion_changing_gap')for z in a0['candidates']]==[z.get('conclusion_changing_gap')for z in a1['candidates']]);mx=read(O/'module-phase-branch-matrix-v151.json');ck('358_26',len(mx['modules'])==26 and sum(len(z['branches'])for z in mx['modules'])==358)
  for z in read(O/'candidate-record-index-v151.json')['model_candidates']:ck('master_SHA',hashlib.sha256(Path(z['master_record']).read_bytes()).hexdigest()==z['master_sha256'])
  ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');ck('runtime_G14_open',any(z['id']=='G14'and not z['closed']for z in read(O/'coverage-current-obligations-v151.json')['obligations']))
 for z in man['artifacts']:ck('manifest',pin(z))
 result=dict(status='PASS',checks=checks,preflight=args.preflight,native_shared_HC_joins=936,native_role_bindings=13368,Engram_positive=48,aux_mean_positive=72,interface_reference_cases=2048,new_pair_clauses=5,new_groups=0,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 if args.preflight:write(O/'post-projection-preflight-v151.json',result)
 print(json.dumps(result))


if __name__=='__main__':main()
