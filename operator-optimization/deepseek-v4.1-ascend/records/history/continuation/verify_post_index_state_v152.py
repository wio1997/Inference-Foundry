"""Independent raw identities, versioned interface unions and integer arena contracts."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,sqlite3,hashlib,json
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=0
 def ck(k,v):
  nonlocal checks
  checks+=1
  assert v,k
 def pin(z):
  p=Path(z['path']);return p.stat().st_size==z['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==z['sha256']
 for z in read(O/'post-index-state-inputs-v152.json')['inputs']:ck('input_pin',pin(z))
 x=read(O/'post-index-state-native-joins-v152.json');a=read(O/'hc-norm-index-state-native-joins-v144.json');p=read(O/'post-projection-native-joins-v151.json');pp={tuple(z['native_role_keys']['HC']):z for z in p['records']};pb={(z['database_index'],z['TASK']['task_record_rowid']):z for z in p['bindings']};bb={(z['database_index'],z['TASK']['task_record_rowid']):z for z in x['bindings']};bydb=defaultdict(list);modes=Counter();used=[]
 for z in x['bindings']:bydb[z['database_index']].append(z)
 ck('exhaustive_old_frontiers',len(x['records'])==len(a['records'])==x['count']==192)
 for z in x['records']:
  r=a['records'][z['old_v144_frontier_index']];di=r['database_index'];q=pp[di,r['stages']['HC']['TASK']['task_record_rowid']];hc=pb[tuple(q['native_role_keys']['HC'])];ck('exact_shared_HC_and_context',r['stages']['HC']['TASK']==hc['TASK']and r['stages']['HC']['API']==hc['API']and z['forward_API_rowid']==r['forward_API_rowid']==q['forward_API_rowid']and z['source_block_ordinal']==r['source_block_ordinal']==q['to_block']and z['old_v151_PostPre_edge_index']==q['post_pre_edge_index']);expected=dict(r['stages'])
  for nm in ['PREV_FFN_POST','PREVIOUS_FFN_PREMIX_PROVIDER','FUTURE_ATTENTION_POST']:expected[nm]=pb[tuple(q['native_role_keys'][nm])]
  ck('all_role_keys',set(expected)==set(z['native_role_keys']))
  for k,v in expected.items():
   key=di,v['TASK']['task_record_rowid'];sv=bb[key];ck('old_role_binding',z['native_role_keys'][k]==list(key)and sv['TASK']==v['TASK']and sv['API']==v['API']);used.append(key)
  ck('mode_geometry_flags',z['mode']==r['mode']and(z['M'],z['Q'],z['T'],z['Engram_positive'],z['aux_mean_positive'])==(q['M'],q['Q'],q['N'],q['Engram_positive'],q['aux_mean_positive']));ck('C2_ring_explicit',('CACHE_RING_UPDATE'in z['native_role_keys'])==(z['mode']=='C2'));ck('not_new_tensor_or_phase',not z['actual_tensor_range_layout_weight_or_state_epoch_bound']and not z['new_native_operand_or_phase_proved']);modes[z['mode']]+=1;modes['Engram_positive']+=z['Engram_positive'];modes['aux_mean_positive']+=z['aux_mean_positive']
 ck('5640_unique_bindings',len(used)==len(set(used))==len(bb)==x['unique_TASK_bindings']==x['role_bindings']==5640);ck('actual_modes',modes==x['modes_and_positive_counts']=={'C2':72,'C1':24,'INDEX_QUERY_WEIGHT_ONLY':96,'Engram_positive':24,'aux_mean_positive':0})
 for spec in x['databases']:
  di=spec['database_index'];d=sqlite3.connect('file:'+spec['raw_database']['path']+'?mode=ro',uri=True);d.row_factory=sqlite3.Row;ss=dict(d.execute('select id,value from STRING_IDS'));ts={r['task_record_rowid']:dict(r)for r in d.execute('select rowid AS task_record_rowid,* from TASK')};apis={r['api_rowid']:dict(r)for r in d.execute('select rowid AS api_rowid,* from PYTORCH_API')};cs={r['globalTaskId']:dict(r)for r in d.execute('select rowid AS compute_record_rowid,* from COMPUTE_TASK_INFO')};ci=defaultdict(set)
  for r in d.execute('select id,connectionId from CONNECTION_IDS'):ci[r[1]].add(r[0])
  for z in bydb[di]:
   t=z['TASK'];r=ts[t['task_record_rowid']]
   for k,v in r.items():ck('raw_TASK_'+k,t[k]==v)
   ck('TASK_type',ss[t['taskType']]==t['task_type_name']);api=z['API'];r=apis[api['api_rowid']]
   for k,v in r.items():
    if k in api:ck('raw_API_'+k,int(api[k])==int(v)if k in ['startNs','endNs']else api[k]==v)
   ck('API_label',api['name_text']==ss[r['name']]);ck('explicit_native_CID_family',api['connectionId']in ci[t['connectionId']]);c=cs[t['globalTaskId']];saved=z['definition'];ck('COMPUTE_ID',saved['globalTaskId']==c['globalTaskId']==t['globalTaskId'])
   for k in ['opType','inputShapes','outputShapes','inputDataTypes','outputDataTypes']:
    if k+'_text'in saved:ck('COMPUTE_'+k,saved[k+'_text']==ss[c[k]])
  d.close()
 u=read(O/'post-index-state-interface-unions-v152.json');seen=set();nvalid=0;over=0
 for z in u['cases']:
  key=(z['mode'],z['M'],z['Q'],z['T'],z['Engram'],z['aux'],tuple(z['actions']));ck('case_unique',key not in seen);seen.add(key);acts=set(z['actions']);rm=set()
  if'UPDATED_X_PRE_FORWARD'in acts:rm.add('UPDATED_X_PRE_LOAD')
  if'PRIVATE_Y_INTERNAL'in acts:rm.update(['Y_STORE','Y_NORM_LOAD'])
  if'QA_WEIGHT_SHARED_BF16_INPUT'in acts:rm.add('WEIGHT_Z_LOAD')
  if z['mode']=='C2':
   if'C2_JOINT_FLOAT_PLANE_READ'in acts:rm.add('C2_SCORE_FLOAT_LOAD')
   if'C2_CAST_JOINT_INPUT_INTERNAL'in acts:rm.update(['C2_FLOAT_STORE','C2_KV_FLOAT_LOAD','C2_SCORE_FLOAT_LOAD'])
  ports=z['baseline_ports'];remain={k:v for k,v in ports.items()if k not in rm};ck('interface_union',rm==set(z['removed_ports'])and remain==z['remaining_ports']and z['logical_union_credit_bytes']==sum(ports[k]for k in rm)==sum(ports.values())-sum(remain.values())and z['overcredit']==z['incorrect_additive_credit']-z['logical_union_credit_bytes']>=0);ck('raw_stores_future_Post_and_Engram_retained',all(k in remain for k in ['X0_STORE','UPDATED_X_FUTURE_POST_LOAD'])and(not z['Engram']or all(k in remain for k in ['ENGRAM_X0_LOAD','X1_STORE']))and(not z['aux']or'OLD_X0_AUX_LOAD'in remain));ck('C2_projected_plane_not_discarded',z['mode']!='C2'or all(k in remain for k in ['PROJECTED_KVSCORE_STORE','PROJECTED_POOL_LOAD']));valid=z['mode']=='C2'or not any(k.startswith('C2_')for k in acts);ck('declared_domain',z['applicable_in_declared_domain']==valid and not z['all_runtime_reader_or_native_traffic_gates_closed']);nvalid+=valid;over+=z['overcredit']>0
 ck('1536_scoped_cases',len(seen)==u['count']==1536 and nvalid>0 and over>0 and not u['physical_bytes_time_or_actual_allocation'])
 demand=read(O/'hc-c2-current-row-demand-proof-v137.json');valid_cases=0
 for z in demand['cases']:
  s,n=z['s'],z['n']
  if not z['valid']:continue
  current=set(range(s,s+n));completed=[t for t in current if t%2==1];pool={v for t in completed for v in [t-1,t]}&current;tail={s+n-1}if n>0 and(s+n-1)%2==0 else set();ck('all_current_rows_pool_or_tail',pool|tail==current and z['current_rows']==len(current)and z['current_pool_rows']==len(pool)and z['current_pool_or_even_tail_rows']==len(pool|tail));valid_cases+=1
 ck('bounded_domain_plus_symbolic_gate',len(demand['cases'])==demand['case_count']==33024 and valid_cases>0 and not demand['source_native_or_actual_metadata_test'])
 v=read(O/'post-index-state-work-and-arena-budget-v152.json');seen=set();fit=0;unfit=0
 for z in v['budget_cases']:
  B,C,D,S=(z[k]for k in ['baseline_B','shared_pool_page_divisor','D_new_global_bytes','released_original_state_pages']);ck('arena_sum_and_baseline',D==sum(z['new_buffer_bytes'].values())and B==z['baseline_pages']*C+z['baseline_residue']and 0<=z['baseline_residue']<C and z['only_additional_distinct_global_copies_charged']and z['existing_raw_X_is_not_new_free_capacity']);fits=B>=D;ck('budget_not_negative',fits==z['allocation_budget_satisfied'])
  if fits:
   exact=S+(B-D)//C-B//C;ceil=(D-z['baseline_residue']+C-1)//C;ck('exact_integer_allocator_formula',z['new_global_pages']==(B-D)//C and z['net_free_page_change']==exact==S-ceil and S-(D+C-1)//C<=exact<=S-D//C);fit+=1
  else:ck('no_allocation_claim_when_budget_fails',z['new_global_pages']is None and z['net_free_page_change']is None);unfit+=1
  ck('no_native_allocator_proof',not z['actual_allocator_alias_intervals_or_state_release_proved'])
 ck('budget_cases',len(v['budget_cases'])==v['budget_case_count']==2304 and fit>0 and unfit>0 and v['native_allocation_capacity_speedup_or_MFU_lowest']=='NOT_ESTABLISHED')
 for z in v['geometry']:
  M,Q,TT,H=(z[k]for k in ['M','Q','T','H']);ck('prefix_and_query_logical_ports',z['PostPre_chosen_logical_input_port']==8*M*H and z['QA_Weight_shared_query_input_reference']==2*Q*H)
  if z['mode']=='C2':ck('C2_work_union_and_row_demand',z['C2_two_FLOAT_direct_classical_ops']==4*TT*H*512 and z['C2_cast_joint_FLOAT_plane_union_credit']==3*z['C2_FLOAT_plane_joint_existing_read_credit']==12*TT*H and z['C2_projected_KVScore_plane_value_bytes']==4096*TT and z['current_projection_rows_removed_from_pool_or_even_tail_policy']==0)
 e=read(O/'post-index-state-lifetime-contract-v152.json')['logical_last_read_example'];ck('distinct_readfrontiers_not_whole_TASK_barrier',max(e['Z_last_direct_reads'].values())==e['Z_earliest_release_in_this_declared_read_contract']==8 and max(e['FLOAT_last_matrix_reads'].values())==e['FLOAT_earliest_release']==10 and max(e['projected_plane_last_reads'].values())==e['projected_earliest_release']==16<e['ring_task_end']and e['ring_committed_at']<e['actual_next_old_ring_read']and max(e['raw_updated_X_last_reads'].values())==e['raw_updated_X_earliest_release']==31 and not e['actual_native_read_or_completion_contract_attested'])
 co=read(O/'post-index-state-composition-v152.json');ck('seven_conditional_pairs',len(co['clauses'])==co['new_pair_clauses']==7 and co['new_pairs']==[['F03',c]for c in ['F09','F10','F11','F17','F18','F19','F34']]and not co['all_runtime_gates_proved']and not co['whole_pair_or_all_subsets_closed'])
 if args.preflight:
  man=read(O/'checkpoint-manifest-v151.json');old=read(O/'candidate-relation-clauses-v151.json')
  for z in co['clauses']:ck('not_old_pair',not any({r['a'],r['b']}=={z['a'],z['b']}for r in old['clauses']))
 else:
  man=read(O/'checkpoint-manifest-v152.json');cp=read(O/'REVIEW-CHECKPOINT-v152.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']);rel=read(O/'candidate-relation-clauses-v152.json');cv=read(O/'candidate-relation-coverage-v152.json');ck('204_26_391',len(rel['clauses'])==cv['scoped_conditional_pairs']==204 and len(rel['higher_order_groups'])==26 and cv['remaining_pair_relations']==391)
  for gid,cid in co['group_extensions']:g=next(z for z in rel['higher_order_groups']if z['id']==gid);ck('group_extension_scoped',cid in g['members']and not g['all_subsets_analyzed']and not g['legal_joint_fusion_proved'])
  a0=read(O/'operator-analysis-coverage-ledger-v151.json');a1=read(O/'operator-analysis-coverage-ledger-v152.json');ck('native_global_unchanged',a0['regimes']==a1['regimes']and all(a0[k]==a1[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']));a0=read(O/'candidate-profile-ledger-v151.json');a1=read(O/'candidate-profile-ledger-v152.json');ck('broad_gaps_preserved',[z.get('conclusion_changing_gap')for z in a0['candidates']]==[z.get('conclusion_changing_gap')for z in a1['candidates']]);mx=read(O/'module-phase-branch-matrix-v152.json');ck('358_26',len(mx['modules'])==26 and sum(len(z['branches'])for z in mx['modules'])==358)
  for z in read(O/'candidate-record-index-v152.json')['model_candidates']:ck('master_SHA',hashlib.sha256(Path(z['master_record']).read_bytes()).hexdigest()==z['master_sha256'])
  ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');ck('G14_open',any(z['id']=='G14'and not z['closed']for z in read(O/'coverage-current-obligations-v152.json')['obligations']))
 for z in man['artifacts']:ck('manifest',pin(z))
 result=dict(status='PASS',checks=checks,preflight=args.preflight,native_shared_HC_joins=192,native_role_bindings=5640,C2_ring_controls=72,Engram_positive=24,old_aux_positive=0,interface_cases=1536,arena_budget_cases=2304,current_row_demand_reference_cases=33024,new_pair_clauses=7,new_groups=0,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 if args.preflight:write(O/'post-index-state-preflight-v152.json',result)
 print(json.dumps(result))


if __name__=='__main__':main()
