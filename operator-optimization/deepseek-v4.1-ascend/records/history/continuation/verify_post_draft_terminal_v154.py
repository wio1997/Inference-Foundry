"""Verify draft/terminal native identities, exact representation limits and backward demand."""
from pathlib import Path
from collections import Counter,defaultdict,deque
from fractions import Fraction
import argparse,sqlite3,json,hashlib,struct,itertools
from finalize_origin_coverage_v9 import N,O,T as TASK_DIR,L as LOOP_DIR,read,write


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=0
 def ck(k,v):
  nonlocal checks
  checks+=1
  assert v,k
 def pin(z):
  p=Path(z['path']);return p.stat().st_size==z['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==z['sha256']
 for z in read(O/'post-draft-terminal-inputs-v154.json')['inputs']:ck('input_pin',pin(z))
 x=read(O/'post-draft-terminal-native-joins-v154.json');a=read(O/'hc-draft-native-frontiers-v142.json');b=read(O/'norm-terminal-native-frontiers-v146.json');p=read(O/'post-pre-native-frontiers-v147.json');bb=b['bindings'];pb=p['bindings'];d146={(z['database_index'],z['source_order_ordinal']):z for z in b['draft_frontiers']};di147={(z['database_index'],z['source_order_ordinal'],z['block']):z for z in p['draft_intra']};dc147={(z['database_index'],z['source_order_ordinal'],z['from_block']):z for z in p['draft_cross']};bindings={(z['database_index'],z['TASK']['task_record_rowid']):z for z in x['bindings']};used=[];cnt=Counter();bydb=defaultdict(list);cbydb=defaultdict(list)
 for z in x['bindings']:bydb[z['database_index']].append(z)
 for z in x['collectives']:cbydb[z['database_index']].append(z)
 ck('104_all_draft',len(x['draft_invocations'])==len(a['records'])==len(b['draft_frontiers'])==104)
 for z in x['draft_invocations']:
  r=a['records'][z['old_v142_record_index']];di=z['database_index'];q=d146[di,z['source_order_ordinal']];exp={'REPEAT':r['repeat'],'SEED_COPY':r['seed']};ck('source_ancestry_not_phase',z['ancestry']==r['source_ancestry']==q['ancestry']and not z['actual_seed_query_context_weight_bank_or_client_phase_bound']and not z['optional_confidence_execution_inferred']);owner=[]
  for j,l in enumerate(r['layers']):
   for nm,v in l['stages'].items():exp['L%d_%s'%(j,nm)]=v
   ck('same_Input_FFN_norms',l['stages']['NORM']['TASK']==bb[q['layers'][j]['INPUT_NORM']]['TASK']and l['stages']['FFN_NORM']['TASK']==bb[q['layers'][j]['FFN_NORM']]['TASK']);e=di147[di,z['source_order_ordinal'],j];sv=z['intra'][j]
   for nm,ix in [('POST',e['left']),('FFN_PRE',e['right']),('HC',e['earlier_premix_and_Post_coeff_provider']),('FFN_POST',e['future_Post'])]:ck('exact_intra_TASK_API',l['stages'][nm]['TASK']==pb[ix]['TASK']and l['stages'][nm]['API']==pb[ix]['API'])
   ck('intra_roles',sv==dict(block=j,left_role='L%d_POST'%j,right_role='L%d_FFN_PRE'%j,incoming_premix_role='L%d_HC'%j,future_Post_role='L%d_FFN_POST'%j));cnt['intra']+=1
   if j<2:
    e=dc147[di,z['source_order_ordinal'],j]
    for role,ix in [(l['stages']['FFN_POST'],e['left']),(r['layers'][j+1]['stages']['HC'],e['right']),(l['stages']['FFN_PRE'],e['incoming_premix_provider'])]:ck('exact_cross_TASK_API',role['TASK']==pb[ix]['TASK']and role['API']==pb[ix]['API'])
    cnt['cross']+=1
   shape=l['stages']['HC']['definition']['inputShapes_text'].strip('"').split(';')[0];owner.append(int(shape.split(',')[0]))
   for nm in ['POST','FFN_POST']:
    d=l['stages'][nm]['definition'];sh=d['inputShapes_text'].strip('"').split(';');ck('public_Post_param_shapes',len(sh)==4 and sh[2].endswith(',4')and sh[3].endswith(',4,4')and d['inputDataTypes_text'].split(';')[-2:]==['FLOAT','FLOAT'])
  for nm,v in r['collapse'].items():ck('same_terminal_collapse_TASK_API',v['TASK']==bb[q['collapse'][nm]]['TASK']and v['API']==bb[q['collapse'][nm]]['API']);exp['FINAL_COLLAPSE_'+nm]=v
  exp['HEAD_NORM']=bb[q['head_norm']]
  for nm,ix in q['main'].items():exp[nm]=bb[ix]
  ck('draft_all_roles',set(exp)==set(z['roles']))
  for nm,v in exp.items():key=(di,v['TASK']['task_record_rowid']);ck('role_identity',z['roles'][nm]==list(key)and bindings[key]['TASK']==v['TASK']and bindings[key]['API']==v['API']);used.append(key)
  ck('separate_owner_repeat_head_rows',z['owner_HC_rows_by_layer']==owner and set(owner)<={2,3}and z['repeat_rows']==z['head_norm_rows']==z['query_rows']and z['raw_collapse_to_global_head_requires_transport_and_row_mapping']);cnt[z['ancestry']]+=1
 ck('312_208_and_ancestry',cnt=={'intra':312,'cross':208,'DUMMY':16,'PROPOSE':88})
 for z in x['target_tails']:
  r=b['target_tails'][z['old_v146_tail_index']];di=z['database_index'];exp={nm:bb[ix]for nm,ix in r['stages'].items()};hits=[e for e in p['target_intra']if e['database_index']==di and pb[e['earlier_premix_and_Post_coeff_provider']]['TASK']['task_record_rowid']==exp['LAST_HC']['TASK']['task_record_rowid']];ck('unique_target_HC_context',len(hits)==1);e=hits[0];ck('same_last_FFNPost',exp['LAST_FFN_POST']['TASK']==pb[e['future_Post']]['TASK']and exp['LAST_FFN_POST']['API']==pb[e['future_Post']]['API']);exp.update(LAST_ATTENTION_POST=pb[e['left']],LAST_FFN_PRE=pb[e['right']]);ck('target_role_names',set(exp)==set(z['roles']))
  for nm,v in exp.items():key=(di,v['TASK']['task_record_rowid']);ck('target_role_identity',z['roles'][nm]==list(key)and bindings[key]['TASK']==v['TASK']and bindings[key]['API']==v['API']);used.append(key)
  ck('terminal_scope_not_nextPre',z['no_next_HCPre_in_reviewed_terminal_source_branch']and z['terminal_raw_snapshot_is_independent_publication']and not z['snapshot_readers_or_output_row_order_values_bound'])
 ck('3064_unique_rawroles',len(used)==len(set(used))==len(bindings)==x['unique_TASK_bindings']==x['role_bindings']==3064);ck('48_MTP_controls_preserved',x['collectives']==b['collectives']and len(x['collectives'])==48 and sum(len(c['raw_API_connection_TASK_rows'])for c in x['collectives'])==168)
 for spec in x['databases']:
  di=spec['database_index'];d=sqlite3.connect('file:'+spec['raw_database']['path']+'?mode=ro',uri=True);d.row_factory=sqlite3.Row;ss=dict(d.execute('select id,value from STRING_IDS'));ts={z['task_record_rowid']:dict(z)for z in d.execute('select rowid AS task_record_rowid,* from TASK')};pa={z['api_rowid']:dict(z)for z in d.execute('select rowid AS api_rowid,* from PYTORCH_API')};cs={z['globalTaskId']:dict(z)for z in d.execute('select rowid AS compute_record_rowid,* from COMPUTE_TASK_INFO')};cm={z['COMM_OP_rowid']:dict(z)for z in d.execute('select rowid AS COMM_OP_rowid,* from COMMUNICATION_OP')};ci=defaultdict(set);rev=defaultdict(set)
  for r in d.execute('select id,connectionId from CONNECTION_IDS'):ci[r[1]].add(r[0]);rev[r[0]].add(r[1])
  for z in bydb[di]:
   t=z['TASK'];r=ts[t['task_record_rowid']]
   for k,v in r.items():ck('raw_TASK_'+k,t[k]==v)
   ck('TASK_type',t['task_type_name']==ss[t['taskType']]);api=z['API'];r=pa[api['api_rowid']]
   for k,v in r.items():
    if k in api:ck('raw_API_'+k,int(api[k])==int(v)if k in['startNs','endNs']else api[k]==v)
   ck('API_name',api['name_text']==ss[r['name']]);ck('explicit_native_family',api['connectionId']in ci[t['connectionId']]);c=cs[t['globalTaskId']];sv=z['definition'];ck('COMPUTE_ID',sv['globalTaskId']==c['globalTaskId']==t['globalTaskId'])
   for k in['opType','inputShapes','outputShapes','inputDataTypes','outputDataTypes']:
    if k+'_text'in sv:ck('COMPUTE_'+k,sv[k+'_text']==ss[c[k]])
  for z in cbydb[di]:
   ck('raw_COMM',z['COMM']==cm[z['COMM']['COMM_OP_rowid']]);api=z['API'];r=pa[api['api_rowid']];ck('COMM_source_API',api['name_text']==ss[r['name']]and int(api['startNs'])==int(r['startNs'])and int(api['endNs'])==int(r['endNs']));ck('raw_API_family_controls_not_unique_Compute',z['raw_API_connection_ids']==sorted(rev[api['connectionId']])and z['raw_API_connection_TASK_rows']==[t for t in ts.values()if t['connectionId']in rev[api['connectionId']]]and z['raw_COMM_exact_connection_TASK_rowids']==[i for i,t in ts.items()if t['connectionId']==z['COMM']['connectionId']]and z['raw_connection_membership_not_unique_Compute_TASK_role_proof']and not z['formal_request_or_clock_binding'])
  d.close()
 rep=read(O/'post-draft-terminal-replay-representation-v154.json')
 for z in rep['cases']:
  H,d,rows=(z[k]for k in['H','Post_depth','owner_rows']);expected=2*H*(d+1)+80*d;ck('reference_input_content',z['declared_replay_input_content_per_row']==expected and z['all_original_resmix16_postmix4_FLOAT_content_per_row']==80*d and z['raw_4H_BF16_content_per_row']==8*H and z['raw_minus_declared_replay_input_content_per_row']==8*H-expected and z['declared_total_input_content']==rows*expected and not z['actual_native_encoder_or_consumer_proved'])
 v=rep['rank_round_counterexample'];E=v['embedding_BF16'];A=v['update_BF16'];aa=v['postmix_FLOAT'];lin=[[Fraction(E[j])+Fraction(aa[i])*Fraction(A[j])for j in range(4)]for i in range(4)]
 def determinant(m):
  total=Fraction(0)
  for perm in itertools.permutations(range(4)):
   inv=sum(perm[i]>perm[j]for i in range(4)for j in range(i+1,4));prod=Fraction((-1)**inv)
   for i in range(4):prod*=Fraction(m[i][perm[i]])
   total+=prod
  return total
 def bf(f):
  bits=struct.unpack('<I',struct.pack('<f',float(f)))[0];return struct.unpack('<f',struct.pack('<I',(bits+32767+((bits>>16)&1))&0xffff0000))[0]
 rounded=[[bf(y)for y in row]for row in lin];ck('rank_round_not_linear_shortcut',str(determinant(lin))==v['unrounded_determinant']=='0'and rounded==v['rounded_Post_BF16']and str(determinant(rounded))==v['rounded_determinant']and determinant(rounded)!=0 and v['rounded_rank']==4 and not v['actual_native_parity_proved'])
 dem=read(O/'post-draft-terminal-demand-and-interface-cases-v154.json');seen=set()
 for z in dem['cases']:
  key=tuple(z['head_unique_rows']),tuple(z['confidence_raw_rows']),z['fuse'];ck('case_unique',key not in seen);seen.add(key);head=set(z['head_unique_rows']);conf=set(z['confidence_raw_rows']);need=head|conf;ck('collapse_vs_ordered_head',z['collapse_unique_rows']==sorted(need)and set(z['ordered_head_multiset'])==head and z['K']==len(z['ordered_head_multiset']));pr=set();fu=set();ports=z['baseline_ports']
  for r in range(z['G']):
   pf={k for k in ports if k.startswith('R%d_'%r)and '_F_'in k}
   if r not in need:pr.update(pf|{'R%d_COLLAPSE_BF16_WRITE'%r})
   if r not in head:pr.add('R%d_HEAD_NORM_BF16_READ'%r)
   if z['fuse']and r in need:fu.update(pf)
   if z['fuse']and r in head:
    fu.add('R%d_HEAD_NORM_BF16_READ'%r)
    if r not in conf:fu.add('R%d_COLLAPSE_BF16_WRITE'%r)
  ck('terminal_union',pr==set(z['pruned_ports'])and fu==set(z['fused_remaining_ports'])and z['remaining_ports']=={k:v for k,v in ports.items()if k not in pr|fu}and z['logical_union_credit']==sum(ports[k]for k in pr|fu));ck('full_MTP_and_raw_confidence_retained',all(k in z['remaining_ports']for k in['FULL_RAW_SNAPSHOT_READ','FULL_RAW_SNAPSHOT_WRITE'])and all('R%d_COLLAPSE_BF16_WRITE'%r in z['remaining_ports']and'R%d_CONF_RAW_BF16_READ'%r in z['remaining_ports']for r in conf)and not z['all_native_reader_order_RNG_or_epoch_gates_closed'])
 ck('512_models_only',len(seen)==dem['count']==512 and not dem['physical_HBM_or_allocation_time_gain'])
 old=read(O/'norm-terminal-backward-demand-cases-v146.json')
 for z in old['cases']:
  n=z['N'];adj=defaultdict(set);roots=set()
  for layer in [0,1]:
   for row in range(n):
    adj[layer,'OUT',row].add((layer,'NORM',row));keys=range(row+1)if z['scheme']=='CAUSAL'else range(n)if z['scheme']=='DENSE'else[row]
    for k in keys:adj[layer,'OUT',row].add((layer,'KV',k))
    adj[layer,'KV',row].add((layer,'NORM',row));adj[layer,'NORM',row].add((layer,'RAW',row))
    if layer==1:adj[layer,'RAW',row].add((0,'OUT',row))
   for w in z['state_W']:roots.add((layer,'KV',w))
  roots|={(1,'OUT',u)for u in z['terminal_U']}|{(1,'RAW',v)for v in z['last_layer_aux_A']};todo=list(roots);reach=set()
  while todo:
   node=todo.pop()
   if node in reach:continue
   reach.add(node);todo.extend(adj[node]-reach)
  for layer,r in enumerate(z['result']['layers']):
   for kind,field in [('OUT','output_rows'),('OUT','query_rows'),('KV','KV_rows'),('NORM','input_norm_rows'),('RAW','input_raw_rows')]:ck('independent_backward_reachability',r[field]==sorted(i for l,k,i in reach if l==layer and k==kind))
 ck('12288_domain_not_all_model',old['count']==len(old['cases'])==12288 and not old['physical_lower_bound_or_actual_gain'])
 ep=read(O/'post-draft-terminal-source-and-epoch-contract-v154.json');graph=defaultdict(set)
 for u,v in ep['generation_edges']:graph[u].add(v)
 todo=[ep['illegal_retrocausal_edge'][1]];reached=set()
 while todo:
  u=todo.pop()
  if u in reached:continue
  reached.add(u);todo.extend(graph[u]-reached)
 ck('future_confidence_retroedge_cycles',ep['illegal_retrocausal_edge'][0]in reached)
 co=read(O/'post-draft-terminal-composition-v154.json');ck('four_scoped_pairs',co['new_pairs']==[['F03',s]for s in['F28','F29','F30','F35']]and co['new_pair_clauses']==4 and not co['whole_F03_or_pair_or_all_subsets_closed']and not co['all_runtime_gates_proved'])
 if args.preflight:
  man=read(O/'checkpoint-manifest-v153.json');rel=read(O/'candidate-relation-clauses-v153.json')
  for z in co['clauses']:ck('not_old_pair',not any({r['a'],r['b']}=={z['a'],z['b']}for r in rel['clauses']))
 else:
  man=read(O/'checkpoint-manifest-v154.json');cp=read(O/'REVIEW-CHECKPOINT-v154.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']);rel=read(O/'candidate-relation-clauses-v154.json');cv=read(O/'candidate-relation-coverage-v154.json');ck('212_26_383',len(rel['clauses'])==cv['scoped_conditional_pairs']==212 and len(rel['higher_order_groups'])==26 and cv['remaining_pair_relations']==383);partners={r['b']if r['a']=='F03'else r['a']for r in rel['clauses']if'F03'in[r['a'],r['b']]};ck('F03_star_scoped_not_whole',len(partners)==34 and not cp['overall_complete']);ck('old_F03_F26_refined','post_draft_terminal_v154'in next(r for r in rel['clauses']if{r['a'],r['b']}=={'F03','F26'}))
  a0=read(O/'operator-analysis-coverage-ledger-v153.json');a1=read(O/'operator-analysis-coverage-ledger-v154.json');ck('native_global_unchanged',a0['regimes']==a1['regimes']and all(a0[k]==a1[k]for k in['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']));a0=read(O/'candidate-profile-ledger-v153.json');a1=read(O/'candidate-profile-ledger-v154.json');ck('broad_gaps_preserved',[r.get('conclusion_changing_gap')for r in a0['candidates']]==[r.get('conclusion_changing_gap')for r in a1['candidates']]);mx=read(O/'module-phase-branch-matrix-v154.json');ck('358_26',len(mx['modules'])==26 and sum(len(r['branches'])for r in mx['modules'])==358)
  for r in read(O/'candidate-record-index-v154.json')['model_candidates']:ck('master_SHA',hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
  ck('active',read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(LOOP_DIR/'state.json')['status']=='EVALUATING');ck('G14_open',any(r['id']=='G14'and not r['closed']for r in read(O/'coverage-current-obligations-v154.json')['obligations']))
 for r in man['artifacts']:ck('manifest',pin(r))
 result=dict(status='PASS',checks=checks,preflight=args.preflight,draft_invocations=104,draft_intra_edges=312,draft_cross_edges=208,target_terminal_tails=24,native_role_bindings=3064,MTP_collective_controls=48,replay_input_content_cases=21,terminal_demand_interface_cases=512,backward_demand_reference_cases=12288,new_pair_clauses=4,new_groups=0,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 if args.preflight:write(O/'post-draft-terminal-preflight-v154.json',result)
 print(json.dumps(result))


if __name__=='__main__':main()
