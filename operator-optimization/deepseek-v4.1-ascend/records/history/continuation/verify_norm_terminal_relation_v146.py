"""Raw-native and independent value-graph audit; green checks do not close goal."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,math,sqlite3,statistics
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
 def ck(k,v):
  assert v,k
  checks.append(k)
 def pin(x):
  b=Path(x['path']).read_bytes();return len(b)==x['bytes']and hashlib.sha256(b).hexdigest()==x['sha256']
 for a in read(O/'norm-terminal-inputs-v146.json')['inputs']:ck('input:'+a['path'],pin(a))
 for a in read(O/'norm-terminal-source-contract-v146.json')['parts']:ck('source',pin(a)and Path(a['path']).read_text().splitlines()[a['first_line']-1:a['last_line']]==a['source'])
 p=read(O/'norm-terminal-native-frontiers-v146.json');bs=p['bindings'];bydb=defaultdict(list);cs=defaultdict(list);roles=Counter();seen=set()
 for i,r in enumerate(bs):bydb[r['database_index']].append((i,r))
 for r in p['collectives']:cs[r['database_index']].append(r)
 roleline={'MTP_SNAPSHOT':1600,'COLLAPSE_MUL':1602,'COLLAPSE_SUM':1602,'COLLAPSE_CAST_IN':1602,'COLLAPSE_CAST_OUT':1602,'TERMINAL_NORM':1605,'LAST_HC':889,'LAST_INPUT_NORM':896,'LAST_FFN_NORM':846,'LAST_FFN_POST':919,'AUX_MEAN':1561,'AUX_HC':889,'AUX_INPUT_NORM':896,'PREVIOUS_FFN_POST':919,'NEXT_INPUT_NORM':896,'DRAFT_INPUT_NORM':896,'DRAFT_FFN_NORM':846}
 for ref in p['databases']:
  di=ref['database_index'];ck('rawpin',pin(ref['raw_database']));ck('cachepin',pin(ref['cached_native']));db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};fs={(x['id'],x['stackDepth'],x['stack'])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};ci=defaultdict(set)
  for x in db.execute('SELECT *FROM CONNECTION_IDS'):ci[x['id']].add(x['connectionId'])
  apis={x['api_rowid']:x for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')};ts={x['task_record_rowid']:x for x in db.execute('SELECT rowid AS task_record_rowid,*FROM TASK')};ds={x['globalTaskId']:x for x in db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO')}
  def api(a):return all(a[k]==apis[a['api_rowid']][k]for k in apis[a['api_rowid']].keys())and a['name_text']==ss[apis[a['api_rowid']]['name']]
  for idx,r in bydb[di]:
   a,t,d=r['API'],r['TASK'],r['definition'];raw=ts[t['task_record_rowid']];dd=ds[t['globalTaskId']];key=di,t['task_record_rowid'];ck('API'+str(idx),api(a));ck('TASK'+str(idx),all(t[k]==raw[k]for k in raw.keys())and a['api_rowid']in t['native_linked_API_rowids']and t['connectionId']in ci[a['connectionId']]);ck('Compute'+str(idx),all((ss[dd[k[:-5]]]if k.endswith('_text')else dd[k])==v for k,v in d.items()));ck('frames'+str(idx),bool(r['selected_frames'])and all((f['id'],f['stackDepth'],f['stack'])in fs and f['id']==a['callchainId']and f['value']==ss[f['stack']]for f in r['selected_frames']));ck('unique',key not in seen);seen.add(key)
   for role in r['roles']:
    roles[role]+=1;f=[x['value']for x in r['selected_frames']]
    if role in roleline:ck('role_source',any('/models/deepseek_v41/model.py('+str(roleline[role])+'):'in x for x in f))
    elif role=='DRAFT_HEAD_NORM':ck('head_source',any('/models/deepseek_v41/dspark.py(332): compute_logits'in x for x in f))
    elif role.startswith('DRAFT_'):ck('draftcollapse_source',any('/models/deepseek_v41/dspark.py(273):'in x for x in f)and any('/models/deepseek_v41/model.py(856):'in x for x in f))
    else:ck('main_source',role in ['MAIN_CONTEXT_PROJ','MAIN_CONTEXT_NORM']and any('/models/deepseek_v41/dspark.py(282):'in x for x in f))
  ct={x['id']:x['name']for x in db.execute('SELECT *FROM ENUM_HCCL_DATA_TYPE')};cached=read(Path(ref['cached_native']['path']))
  for r in cs[di]:
   a,c=r['API'],r['COMM'];raw=db.execute('SELECT rowid AS COMM_OP_rowid,*FROM COMMUNICATION_OP WHERE rowid=?',(c['COMM_OP_rowid'],)).fetchone();ck('COMM',api(a)and dict(raw)==c and c['connectionId']in ci[a['connectionId']]and c['rankSize']==8 and r['dtype_text']==ct[c['dataType']]);ck('cached_COMMtasks',set(r['linked_TASK_rowids'])=={t['task_record_rowid']for t in cached['TASK_records']if a['api_rowid']in t.get('native_linked_API_rowids',[])}and not r['formal_request_or_clock_binding']);ck('raw_API_connection_family',r['raw_API_connection_ids']==sorted(ci[a['connectionId']])and r['raw_API_connection_TASK_rows']==[dict(t)for t in ts.values()if t['connectionId']in ci[a['connectionId']]]and r['raw_COMM_exact_connection_TASK_rowids']==[t['task_record_rowid']for t in ts.values()if t['connectionId']==c['connectionId']]and r['raw_connection_membership_not_unique_Compute_TASK_role_proof']);ck('COMMframes',all((f['id'],f['stackDepth'],f['stack'])in fs and f['value']==ss[f['stack']]for f in r['selected_frames']))
  missing={x[2]for x in fs if x[2]not in ss};ck('missing_unselected',missing==set(ref['unresolved_unselected_stack_ids']));db.close()
 ck('3528',len(seen)==p['unique_TASK_bindings']==3528 and sum(r['new_source_CID_binding']for r in bs)==p['new_source_CID_TASK_bindings']==248 and roles==p['roles']and sum(roles.values())==p['role_binding_count']==3648)
 oldt={(x['database_index'],x['N']):x for x in read(O/'heads-sampling-native-chains-v65.json')['target_tails']if x['N']>24};old_d={(x['database_index'],x['source_order_ordinal']):x for x in read(O/'hc-draft-native-frontiers-v142.json')['records']};old_m={(x['database_index'],x['forward_API_rowid'],x['to_block']):x for x in read(O/'hc-moe-native-frontiers-v141.json')['bridges']};obs=read(O/'norm-terminal-observed-and-work-v146.json');ge=Counter();en=[];su=[];dg=Counter();allrefs=set()
 def use(i):allrefs.add(i);return bs[i]
 for r,x in zip(p['target_tails'],obs['targets']):
  s={k:use(i)for k,i in r['stages'].items()};m=r['owner_rows'];g=8*m;t=r['cache_rows'];h=5120;ge[m,g,t]+=1;di=r['database_index'];ck('target_G',s['MTP_SNAPSHOT']['definition']['inputShapes_text']==f'"{g},20480"'and s['TERMINAL_NORM']['definition']['inputShapes_text']==f'"{g},5120;5120"'and s['LAST_INPUT_NORM']['definition']['inputShapes_text']==f'"{m},5120;5120"');ck('target_dtype',s['MTP_SNAPSHOT']['definition']['opType_text']=='TensorMove'and s['COLLAPSE_CAST_IN']['definition']['inputDataTypes_text']=='DT_BF16'and s['COLLAPSE_CAST_IN']['definition']['outputDataTypes_text']=='FLOAT'and s['COLLAPSE_CAST_OUT']['definition']['outputDataTypes_text']==s['TERMINAL_NORM']['definition']['inputDataTypes_text'].split(';')[0]=='DT_BF16')
  six=[s[k]for k in ['MTP_SNAPSHOT','COLLAPSE_CAST_IN','COLLAPSE_MUL','COLLAPSE_SUM','COLLAPSE_CAST_OUT','TERMINAL_NORM']];ck('same_oldtail_set',set(y['TASK']['task_record_rowid']for y in six)==set(y['TASK_rowid']for y in oldt[di,g]['stages']));a=r['forward_API'];ck('same_CPU_source_window',all(y['API']['globalTid']==a['globalTid']and int(a['startNs'])<=int(y['API']['startNs'])<=int(y['API']['endNs'])<=int(a['endNs'])for y in s.values()))
  lo=min(z['TASK']['startNs']for z in six);hi=max(z['TASK']['endNs']for z in six);dur=sum(z['TASK']['endNs']-z['TASK']['startNs']for z in six);en.append(hi-lo);su.append(dur);ck('observed',x['M']==m and x['G']==g and x['T']==t and x['current_selected_task_envelope_ns']==hi-lo and x['current_six_TASK_duration_sum_ns']==dur and x['actual_fusion_critical_path_gain_ns']is None and x['current_snapshot_then_collapse_norm_order']==all(a['TASK']['endNs']<=z['TASK']['startNs']for a,z in zip(six,six[1:])))
  for role,cidx in r['collectives'].items():
   c=p['collectives'][cidx];ck('MTP_commgeometry',c['database_index']==di and c['role']==role and c['COMM']['count']==m*(20480 if role=='MTP_HIDDEN_AG' else 4)and c['dtype_text']==('BFP16'if role=='MTP_HIDDEN_AG'else'FP32')and len(c['linked_TASK_rowids'])==(0 if role=='MTP_HIDDEN_AG' else 1))
 ck('targetgeometry',ge=={(800,6400,6400):8,(816,6528,6528):8,(834,6672,6668):8})
 for x in obs['target_geometry']:
  m,g,t,h=x['M'],x['G'],x['T'],x['H'];ck('targetinterfaces',x['frontiers']==ge[m,g,t]and h==5120 and x['input_norm_owner_Y_private_interface']==4*m*h and x['raw_MTP_value_content']==8*g*h and x['raw_MTP_snapshot_copy_interface']==16*g*h and x['premix_global_FLOAT_value_content']==16*g and x['collapse_X_FLOAT_private_interface']==x['collapse_product_FLOAT_private_interface']==32*g*h and x['collapse_sum_FLOAT_private_interface']==8*g*h and x['collapse_BF16_private_interface']==4*g*h and x['collapse_RMS_private_union']==sum(x[k]for k in ['collapse_X_FLOAT_private_interface','collapse_product_FLOAT_private_interface','collapse_sum_FLOAT_private_interface','collapse_BF16_private_interface'])==76*g*h and x['physical_IO_capacity_time_or_MFU']is None)
 for key,v in [('selected_target_envelope_ns',en),('selected_target_TASK_sum_ns',su)]:ck('timingstat',obs[key]==dict(min=min(v),median=statistics.median(v),max=max(v),count=len(v)))
 for r in p['draft_frontiers']:
  old=old_d[r['database_index'],r['source_order_ordinal']];hd=use(r['head_norm']);a=hd['API'];m=int(use(r['layers'][0]['INPUT_NORM'])['definition']['inputShapes_text'].strip('"').split(',')[0]);hr=int(hd['definition']['inputShapes_text'].strip('"').split(',')[0]);dg[r['context_rows'],r['query_rows'],m,hr,r['ancestry']]+=1;ck('draft_headwindow',a['globalTid']==old['seed']['API']['globalTid']and int(old['collapse']['CAST_OUT']['API']['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(old['seed']['API']['startNs'])and hd['definition']['opType_text']=='RmsNorm'and hd['definition']['inputDataTypes_text']=='DT_BF16;DT_BF16'and hr==r['query_rows'])
  for l,ol in zip(r['layers'],old['layers']):
   for k,ok in [('INPUT_NORM','NORM'),('FFN_NORM','FFN_NORM')]:ck('draft_reuse',use(l[k])['TASK']['task_record_rowid']==ol['stages'][ok]['TASK']['task_record_rowid'])
  for k,i in r['collapse'].items():ck('draftcollapse_reuse',use(i)['TASK']['task_record_rowid']==old['collapse'][k]['TASK']['task_record_rowid'])
  for i in r['main'].values():use(i)
 ck('draftgeometry',dg=={(x['context_rows'],x['query_rows'],x['input_norm_owner_rows'],x['head_norm_global_rows'],x['ancestry']):x['frontiers']for x in obs['draft_geometry']})
 for r in p['aux_frontiers']:
  for i in r['stages'].values():use(i)
 for r in p['interlayer_bridges']:
  old=old_m[r['database_index'],r['forward_API_rowid'],r['to_block']];ck('bridge',r['from_block']+1==r['to_block']and use(r['previous_FFN_Post'])['TASK']['task_record_rowid']==old['previous_FFN_Post']['TASK']['task_record_rowid']and r['Engram_positive']==bool(old['next_Engram_positive_APIs'])and r['aux_positive']==bool(old['next_aux_mean'])and not r['all_previous_Post_readers_bound']);use(r['next_input_norm'])
 ck('allrefs',allrefs==set(range(len(bs)))and len(p['target_tails'])==24 and len(p['draft_frontiers'])==104 and len(p['aux_frontiers'])==72 and len(p['interlayer_bridges'])==936 and len(p['collectives'])==48)
 # Independent graph ancestor closure for each selected row/value implementation.
 def ancestors(n,u,w,a,scheme):
  edges={}
  for l in range(2):
   for i in range(n):
    X=('X',l,i);Z=('Z',l,i);Q=('Q',l,i);K=('K',l,i);A=('ATTN',l,i);OUT=('O',l,i);win=list(range(i+1))if scheme=='CAUSAL'else list(range(n))if scheme=='DENSE'else[i]
    edges[X]=[('O',l-1,i)]if l else[];edges[Z]=[X];edges[Q]=[Z];edges[K]=[Z];edges[A]=[Q]+[('K',l,j)for j in win];edges[OUT]=[X,A]
  pending=[('O',1,i)for i in u]+[('K',l,i)for l in range(2)for i in w]+[('X',1,i)for i in a];seen=set()
  while pending:
   x=pending.pop()
   if x in seen:continue
   seen.add(x);pending.extend(edges[x])
  layers=[dict(layer=l,output_rows=[i for i in range(n)if('O',l,i)in seen],query_rows=[i for i in range(n)if('Q',l,i)in seen],KV_rows=[i for i in range(n)if('K',l,i)in seen],input_norm_rows=[i for i in range(n)if('Z',l,i)in seen],input_raw_rows=[i for i in range(n)if('X',l,i)in seen])for l in range(2)]
  return dict(layers=layers,total_input_norm_rows=sum(len(x['input_norm_rows'])for x in layers),initial_raw_rows=layers[0]['input_raw_rows'])
 dm=read(O/'norm-terminal-backward-demand-cases-v146.json');cases=set();full=0;over=0
 for x in dm['cases']:
  key=(x['scheme'],tuple(x['terminal_U']),tuple(x['state_W']),tuple(x['last_layer_aux_A']));ck('unique_case',key not in cases);cases.add(key);d=ancestors(x['N'],x['terminal_U'],x['state_W'],x['last_layer_aux_A'],x['scheme']);nr=d['total_input_norm_rows'];ck('independent_dependency_graph',d==x['result']);ck('interface_union',x['logical_Y_edge_pruning_credit']+x['logical_Y_edge_remaining_fusion_credit']==x['logical_Y_edge_joint_union']==4*dm['H']*8 and x['incorrect_full_fusion_plus_pruning']==x['logical_Y_edge_joint_union']+x['logical_Y_edge_pruning_credit']and not x['all_native_or_state_gates_closed'])
  if len(x['state_W'])==4:full+=1;ck('full_W_requires_all_norm_rows',nr==8)
  if x['incorrect_full_fusion_plus_pruning']>x['logical_Y_edge_joint_union']:over+=1
 ck('12288',len(cases)==dm['count']==12288 and full==768 and over>0 and not dm['physical_lower_bound_or_actual_gain'])
 w=read(O/'norm-terminal-value-row-epoch-witnesses-v146.json');x=w['same_Z_different_aux'];ck('aux_not_Z',x['aux_mean']==[sum(y[0]for y in r)/4 for r in [x['X0'],x['X1']]]and x['X0'][0]==x['X1'][0]and x['normalized_Z'][0]==x['normalized_Z'][1]and x['aux_mean'][0]!=x['aux_mean'][1])
 x=w['same_H_different_norm'];norm=lambda v,eps=1e-6,g=[1,1]:[y*z/math.sqrt(sum(t*t for t in v)/len(v)+eps)for y,z in zip(v,g)];ck('norm_identity_gates',all(abs(a-b)<1e-14 for a,b in zip(norm(x['input']),x['input_Z']))and all(abs(a-b)<1e-14 for a,b in zip(norm(x['terminal']),x['terminal_Z']))and x['input_Z']!=x['terminal_Z']and x['input_Z']!=x['same_input_different_gamma']and x['input_Z']!=x['same_input_different_epsilon'])
 x=w['final_demand_not_input_demand'];ck('demand_witness',sorted(set(x['ordered_head_I']+x['additional_final_J']))==x['terminal_U']and set(x['required_input_norm'])==set(x['required_cache_W'])and len(x['terminal_U'])<len(x['required_input_norm'])and len(x['ordered_head_I'])>len(set(x['ordered_head_I'])))
 x=w['raw_snapshot_independent'];ck('snapshot_cut',x['conditional_terminal_U']==x['conditional_collapse_norm_rows']and set(x['conditional_terminal_U'])<set(x['baseline_snapshot_raw_rows'])and x['actual_snapshot_consumed_rows']is None and not x['delete_raw_snapshot_authorized'])
 x=w['acceptance_graph'];reach=defaultdict(set)
 for a,z in x['edges']:reach[a].add(z)
 def path(a,z,seen=None):
  if a==z:return True
  s=set(seen or[])
  if a in s:return False
  return any(path(c,z,s|{a})for c in reach[a])
 a,z=x['illegal_future_edge'];ck('future_cycle',path(z,a)and not path(a,z)and not x['current_decoder_pruned_by_its_own_confidence'])
 comp=read(O/'norm-terminal-composition-v146.json');ck('eightpairs',comp['new_pairs']==[['F02',x]for x in ['F23','F26','F27','F29','F30','F31','F34','F35']]and not comp['all_runtime_gates_proved'])
 if args.preflight:
  rr=read(O/'candidate-relation-clauses-v145.json')
  for pair in comp['new_pairs']:ck('pending',not any({x['a'],x['b']}==set(pair)for x in rr['clauses']))
  for gid,m in comp['group_extensions']:ck('group_pending',m not in next(g for g in rr['higher_order_groups']if g['id']==gid)['members'])
  man=read(O/'checkpoint-manifest-v145.json')
 else:
  cp=read(O/'REVIEW-CHECKPOINT-v146.json');ck('progress',cp['status']=='PROGRESS'and cp['whole_modules_closed']==0 and not cp['overall_complete']);rr=read(O/'candidate-relation-clauses-v146.json');ck('186pairs',len(rr['clauses'])==rr['clause_count']==186 and len({tuple(sorted((x['a'],x['b'])))for x in rr['clauses']})==186 and len(rr['higher_order_groups'])==26)
  for c in comp['clauses']:
   r=next(x for x in rr['clauses']if x['id']==c['id']);ck('conditional',r['a']==c['a']and r['b']==c['b']and not r['all_runtime_gates_proved']and r['critical_path_gain_us']is None)
  for gid,m in comp['group_extensions']:
   g=next(x for x in rr['higher_order_groups']if x['id']==gid);ck('group_not_closed',m in g['members']and not g['all_subsets_analyzed']and not g['legal_joint_fusion_proved'])
  f2={x['b']if x['a']=='F02'else x['a']for x in rr['clauses']if'F02'in[x['a'],x['b']]};ck('F02star34_scoped',len(f2)==34)
  cv=read(O/'candidate-relation-coverage-v146.json');nv=read(O/'candidate-pair-navigation-v146.json');ck('409pending',cv['scoped_conditional_pairs']==186 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==409)
  a=read(O/'operator-analysis-coverage-ledger-v145.json');z=read(O/'operator-analysis-coverage-ledger-v146.json');ck('nativeglobal_unchanged',all(a[k]==z[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']));a=read(O/'candidate-profile-ledger-v145.json');z=read(O/'candidate-profile-ledger-v146.json');ck('broader_gaps_preserved',[x.get('conclusion_changing_gap')for x in a['candidates']]==[x.get('conclusion_changing_gap')for x in z['candidates']]);mm=read(O/'module-phase-branch-matrix-v146.json');ck('358_26',len(mm['modules'])==26 and sum(len(x['branches'])for x in mm['modules'])==358)
  for r in read(O/'candidate-record-index-v146.json')['model_candidates']:ck('master',hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
  ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');man=read(O/'checkpoint-manifest-v146.json')
 for x in man['artifacts']:ck('manifest:'+x['path'],pin(x))
 result=dict(status='PASS',checks=len(checks),preflight=args.preflight,unique_native_TASKs=3528,new_source_CID_bindings=248,target_tails=24,draft_head_norms=104,aux_frontiers=72,interlayer_bridges=936,MTP_collectives=48,demand_cases=12288,full_state_cases=full,doublecount_examples=over,new_scoped_pairs=8,new_native_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 if args.preflight:write(O/'norm-terminal-preflight-v146.json',result)
 print(json.dumps(result))


if __name__=='__main__':main()
