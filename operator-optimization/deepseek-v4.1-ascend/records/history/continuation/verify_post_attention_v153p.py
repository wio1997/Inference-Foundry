"""Raw operator/control identities and bounded alternative graph contracts."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,sqlite3,hashlib,json,struct
from finalize_origin_coverage_v9 import N,O,T as TASK_DIR,L,read,write


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();assert not args.preflight,'post-only immutable correction: do not overwrite sealed preflight';checks=0
 def ck(k,v):
  nonlocal checks
  checks+=1
  assert v,k
 def pin(z):
  p=Path(z['path']);return p.stat().st_size==z['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==z['sha256']
 for z in read(O/'post-attention-inputs-v153.json')['inputs']:ck('input_pin',pin(z))
 x=read(O/'post-attention-native-joins-v153.json');a=read(O/'hc-norm-attention-native-frontiers-v145.json');b=read(O/'post-pre-native-frontiers-v147.json');idx={(z['database_index'],z['forward_API_rowid'],z['block']):z for z in b['target_intra']};binding={(z['database_index'],z['TASK']['task_record_rowid']):z for z in x['bindings']};cb={(z['COMM']['database_index'],z['COMM']['COMM_OP_rowid']):z for z in x['collective_bindings']};bydb=defaultdict(list);cbydb=defaultdict(list);used=[];comused=[];counts=Counter();geos=Counter()
 for z in x['bindings']:bydb[z['database_index']].append(z)
 for key,z in cb.items():cbydb[key[0]].append(z)
 ck('all960',x['count']==len(x['records'])==len(a['records'])==len(b['target_intra'])==960)
 for z in x['records']:
  r=a['records'][z['old_v145_frontier_index']];di=z['database_index'];q=idx[di,z['forward_API_rowid'],z['source_block_ordinal']];pre=b['bindings'][q['earlier_premix_and_Post_coeff_provider']];post=b['bindings'][q['left']];ck('same_HC_and_Post_ID',pre['TASK']==r['stages']['HC']['TASK']and pre['API']==r['stages']['HC']['API']and post['TASK']==r['stages']['POST']['TASK']and post['API']==r['stages']['POST']['API']and z['forward_API_rowid']==r['forward_API_rowid']and z['source_block_ordinal']==r['source_block_ordinal']);exp=dict(r['stages']);exp.update(NEXT_FFN_PRE=b['bindings'][q['right']],FUTURE_FFN_POST=b['bindings'][q['future_Post']]);ck('all_role_names',set(exp)==set(z['native_role_keys']))
  for nm,v in exp.items():
   key=di,v['TASK']['task_record_rowid'];ck('exact_role',z['native_role_keys'][nm]==list(key)and binding[key]['TASK']==v['TASK']and binding[key]['API']==v['API']);used.append(key)
  ck('collective_roles',set(z['collective_keys'])==set(r['collectives']))
  for nm,v in r['collectives'].items():
   key=di,v['COMM']['COMM_OP_rowid'];ck('old_collective_grade_unchanged',z['collective_keys'][nm]==list(key)and cb[key]==v);comused.append(key)
  ck('correct_M_Q_T_G',z['M']==r['carrier_sender_rows']and z['Q']==r['query_rows']and z['T']==r['cache_rows']and z['G']==z['P']*z['M']==8*z['M']);ck('no_phase_or_operand',not z['new_param_row_head_tag_or_actual_tensor_phase_proved']and z['old_collective_record_labels_not_crossrank_clock_or_value_proof']);geos[z['M'],z['Q'],z['T'],z['G']]+=1
  for nm in ['QLI','HEAD_ZERO','HEAD_COPY','CANDIDATE_COPY']:counts[nm]+=(nm in z['native_role_keys'])
 ck('12200_unique_roles',len(used)==len(set(used))==len(binding)==x['role_bindings']==x['unique_TASK_bindings']==12200);ck('2880_collective_control_unique',len(comused)==len(set(comused))==len(cb)==x['collective_control_count']==2880);ck('positive_markers',counts=={'QLI':192,'HEAD_ZERO':40,'HEAD_COPY':40,'CANDIDATE_COPY':24});ck('observed_geos',geos=={(800,800,6400,6400):320,(816,816,6528,6528):320,(834,830,6668,6672):40,(834,834,6668,6672):280})
 for spec in x['databases']:
  di=spec['database_index'];d=sqlite3.connect('file:'+spec['raw_database']['path']+'?mode=ro',uri=True);d.row_factory=sqlite3.Row;ss=dict(d.execute('select id,value from STRING_IDS'));ts={z['task_record_rowid']:dict(z)for z in d.execute('select rowid AS task_record_rowid,* from TASK')};pa={z['api_rowid']:dict(z)for z in d.execute('select rowid AS api_rowid,* from PYTORCH_API')};cs={z['globalTaskId']:dict(z)for z in d.execute('select rowid AS compute_record_rowid,* from COMPUTE_TASK_INFO')};com={z['COMM_OP_rowid']:dict(z)for z in d.execute('select rowid AS COMM_OP_rowid,* from COMMUNICATION_OP')};ci=defaultdict(set)
  for r in d.execute('select id,connectionId from CONNECTION_IDS'):ci[r[1]].add(r[0])
  for z in bydb[di]:
   t=z['TASK'];r=ts[t['task_record_rowid']]
   for k,v in r.items():ck('raw_TASK_'+k,t[k]==v)
   ck('raw_TASK_type',t['task_type_name']==ss[t['taskType']]);api=z['API'];r=pa[api['api_rowid']]
   for k,v in r.items():
    if k in api:ck('raw_API_'+k,int(api[k])==int(v)if k in ['startNs','endNs']else api[k]==v)
   ck('raw_API_name',api['name_text']==ss[r['name']]);ck('explicit_native_family',api['connectionId']in ci[t['connectionId']]);c=cs[t['globalTaskId']];sv=z['definition'];ck('native_COMPUTE_ID',sv['globalTaskId']==c['globalTaskId']==t['globalTaskId'])
   for k in ['opType','inputShapes','outputShapes','inputDataTypes','outputDataTypes']:
    if k+'_text'in sv:ck('native_COMPUTE_'+k,sv[k+'_text']==ss[c[k]])
  for z in cbydb[di]:
   c=z['COMM'];r=com[c['COMM_OP_rowid']]
   for k in ['opId','startNs','endNs','connectionId','count','rankSize']:ck('raw_COMM_'+k,c[k]==r[k])
   ck('raw_COMM_labels',c['operator']==ss[r['opType']]and c['group_name']==ss[r['groupName']]and c['algorithm_label']==ss[r['algType']]);api=z['API'];r=pa[api['api_rowid']];ck('COMM_source_API_raw',api['name_text']==ss[r['name']]and api['globalTid']==r['globalTid']and int(api['startNs'])==int(r['startNs'])and int(api['endNs'])==int(r['endNs']))
  d.close()
 u=read(O/'post-attention-interface-unions-v153.json');seen=set();over=0
 for z in u['cases']:
  key=z['M'],z['Q'],z['T'],z['G'],tuple(z['actions']);ck('unique_case',key not in seen);seen.add(key);a0=set(z['actions']);rm=set();ports=z['baseline_ports']
  if'WOB_PRIVATE_COPY_INTERFACE_INTERNAL'in a0:rm.update(['WO_B_PRIVATE_WRITE','OUTPUT_COPY_READ'])
  if'WOB_STABLE_RS_INTERFACES_INTERNAL'in a0:rm.update(['WO_B_PRIVATE_WRITE','OUTPUT_COPY_READ','STABLE_OUTPUT_WRITE','RS_INPUT_READ'])
  if'RS_POST_INTERFACE_INTERNAL'in a0:rm.update(['RS_OUTPUT_WRITE','POST_ATTN_INPUT_READ'])
  if'POST_PRE_INPUT_FORWARD'in a0:rm.add('FFN_PRE_UPDATED_RAW_READ')
  if'DIRECT_PAD_PACK'in a0 and z['Q']<z['M']:rm.update(['HEAD_ZERO_FULL_WRITE','HEAD_VALID_COPY_WRITE','HEAD_PADDED_PACK_READ'])
  if'INVERSE_ROPE_PACK_INTERFACE_INTERNAL'in a0:rm.update(['INV_ROPE_UPDATED_WRITE','HEAD_VALID_READ'])
  ck('ports_union',rm==set(z['removed_ports'])and z['remaining_ports']=={k:v for k,v in ports.items()if k not in rm}and z['logical_union_credit']==sum(ports[k]for k in rm)==sum(ports.values())-sum(z['remaining_ports'].values())and z['overcredit']==z['incorrect_additive_credit']-z['logical_union_credit']>=0);ck('raw_future_Post_retained',all(k in z['remaining_ports']for k in ['POST_UPDATED_RAW_WRITE','FUTURE_FFN_POST_RAW_READ'])and not z['native_handoff_all_reader_precision_or_progress_proved']);over+=z['overcredit']>0
 ck('case_domain',len(seen)==u['count']==256 and over>0 and not u['native_gain_or_physical_bound'])
 for z in u['geometry']:
  M,Q,T,G,H=(z[k]for k in ['M','Q','T','G','H']);ck('G_T_absolute_scope',z['current_carrier_global_Z_content']==2*G*H and z['full_G_materialization_correction']==2*(G-T)*H and z['full_G_materialization_correction']in[0,40960]);ck('output_union_and_Post_reference',z['output_full_interface_union_reference']==2*z['output_copy_private_reference']==8*G*H and z['output_full_RSPost_PostPre_union']==8*G*H+12*M*H and z['observed_RS_zero_cat_credit']==0 and not z['raw_future_FFN_Post_store_removal_proved'])
 c=read(O/'post-attention-projected-head-carrier-v153.json');ck('not_wire_ratio',c['logical_value_payload_ratio']==0.25 and not c['wire_or_latency_ratio_proved']and not c['whole_pair_native_or_physical_closed'])
 for z in c['geometry']:
  M,G,P=(z[k]for k in ['M','G','P']);ck('per_peer_carrier',z['old_BF16_A2A_value_bytes_per_peer']==4*z['new_projected_BF16_A2A_reference_bytes_per_peer']==2*M*4096);ck('rank_and_remote_values',z['old_rank_send_value_bytes']==P*z['old_BF16_A2A_value_bytes_per_peer']and z['new_rank_send_value_bytes']==P*z['new_projected_BF16_A2A_reference_bytes_per_peer']and z['old_remote_only_value_bytes']==4*z['new_remote_only_reference_value_bytes']==(P-1)*M*4096*2 and z['old_received_value_bytes']==4*z['new_received_reference_value_bytes']==2*G*4096);ck('weights_and_work_repriced',z['extra_weight_group_content_for_explicit_replication']==(P-1)*4096*1024*2==58720256 and z['all_P_group_weight_BF16_union']==P*z['old_one_group_weight_BF16_content']and z['old_recipient_WO_A_classical_ops']==z['new_owner_all_group_WO_A_classical_ops']==2*G*4096*1024 and not z['actual_native_allocation_wire_gain_or_replicated_layer_count_proved'])
  for q in z['replicated_layer_content_examples']:ck('replicated_L_not_actual',q['extra_weight_content']==q['L_replicated']*z['extra_weight_group_content_for_explicit_replication'])
 for z in c['finite_head_map_cases']:
  baseline=[];before=[];bad=[]
  for j in range(2):
   rows=[z['send'][r][m][j]for r in range(2)for m in range(2)];baseline.append([sum(v[k]*z['weights'][j][k]for k in range(2))for v in rows]);before.append([sum(z['send'][r][m][j][k]*z['weights'][j][k]for k in range(2))for r in range(2)for m in range(2)]);bad.append([sum(v[k]*z['weights'][0][k]for k in range(2))for v in rows])
  ck('head_map_projection_commutes_in_declared_reference',baseline==before==z['receive_then_project']==z['owner_project_then_exchange']and bad==z['wrong_one_group_weight_for_all_destinations']and bad!=before and not z['actual_native_mode_or_weight_binding'])
 s=read(O/'post-attention-source-and-carrier-contract-v153.json');v=s['partial_Post_SUM_round_witness']
 def bf(f):
  bits=struct.unpack('<I',struct.pack('<f',f))[0];bits=(bits+32767+((bits>>16)&1))&0xffff0000;return struct.unpack('<f',struct.pack('<I',bits))[0]
 p0,p1=v['partials_BF16'];k=v['fixed_post_coefficient_FLOAT'];ck('Post_SUM_round_counterexample',bf(k*bf(p0+p1))==v['original_SUM_BF16_then_Post_BF16']and bf(bf(k*p0)+bf(k*p1))==v['Post_on_each_BF16_partial_then_SUM_BF16']and v['original_SUM_BF16_then_Post_BF16']!=v['Post_on_each_BF16_partial_then_SUM_BF16']and not v['actual_native_policy_or_gain_proved'])
 co=read(O/'post-attention-composition-v153.json');ck('four_conditional_pairs',co['new_pairs']==[['F03',v]for v in['F12','F13','F14','F15']]and co['new_pair_clauses']==4 and not co['all_runtime_gates_proved']and not co['whole_pair_or_all_subsets_closed'])
 if args.preflight:
  man=read(O/'checkpoint-manifest-v152.json');rel=read(O/'candidate-relation-clauses-v152.json')
  for z in co['clauses']:ck('new_pair',not any({z['a'],z['b']}=={r['a'],r['b']}for r in rel['clauses']))
  for gid,cid in co['group_extensions']:ck('new_group_member',cid not in next(g for g in rel['higher_order_groups']if g['id']==gid)['members'])
 else:
  man=read(O/'checkpoint-manifest-v153.json');cp=read(O/'REVIEW-CHECKPOINT-v153.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']);rel=read(O/'candidate-relation-clauses-v153.json');cv=read(O/'candidate-relation-coverage-v153.json');ck('208_26_387',len(rel['clauses'])==cv['scoped_conditional_pairs']==208 and len(rel['higher_order_groups'])==26 and cv['remaining_pair_relations']==387)
  for gid,cid in co['group_extensions']:g=next(g for g in rel['higher_order_groups']if g['id']==gid);ck('group_remains_scoped',cid in g['members']and not g['all_subsets_analyzed']and not g['legal_joint_fusion_proved'])
  r=next(r for r in rel['clauses']if{r['a'],r['b']}=={'F03','F16'});ck('old_F03_F16_refined','post_attention_v153'in r)
  a0=read(O/'operator-analysis-coverage-ledger-v152.json');a1=read(O/'operator-analysis-coverage-ledger-v153.json');ck('native_global_unchanged',a0['regimes']==a1['regimes']and all(a0[k]==a1[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']));a0=read(O/'candidate-profile-ledger-v152.json');a1=read(O/'candidate-profile-ledger-v153.json');ck('broad_gaps_preserved',[z.get('conclusion_changing_gap')for z in a0['candidates']]==[z.get('conclusion_changing_gap')for z in a1['candidates']]);mx=read(O/'module-phase-branch-matrix-v153.json');ck('358_26',len(mx['modules'])==26 and sum(len(z['branches'])for z in mx['modules'])==358)
  for z in read(O/'candidate-record-index-v153.json')['model_candidates']:ck('master_SHA',hashlib.sha256(Path(z['master_record']).read_bytes()).hexdigest()==z['master_sha256'])
  ck('active',read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');ck('G14_open',any(z['id']=='G14'and not z['closed']for z in read(O/'coverage-current-obligations-v153.json')['obligations']))
 for z in man['artifacts']:ck('manifest',pin(z))
 result=dict(status='PASS',checks=checks,preflight=args.preflight,attentionPost_FFNPre_joins=960,native_role_bindings=12200,collective_control_records=2880,QLI_frontiers=192,head_pad_frontiers=40,interface_cases=256,projected_carrier_geometries=4,new_pair_clauses=4,new_groups=0,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 if args.preflight:write(O/'post-attention-preflight-v153.json',result)
 print(json.dumps(result))


if __name__=='__main__':main()
