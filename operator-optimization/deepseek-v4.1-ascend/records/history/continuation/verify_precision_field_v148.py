"""Reconstruct full raw flag/regime scope and audit precision inference limits."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import argparse,ast,hashlib,html,json,re,sqlite3
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();count=0
 def ck(k,v):
  nonlocal count
  assert v,k
  count+=1
 def pin(x):
  b=Path(x['path']).read_bytes();return len(b)==x['bytes']and hashlib.sha256(b).hexdigest()==x['sha256']
 for x in read(O/'precision-field-inputs-v148.json')['inputs']:ck('input',pin(x))
 pop=read(O/'precision-field-population-v148.json');pd=defaultdict(dict);pe=defaultdict(dict)
 for x in pop['definitions']:ck('unique_def',(x[1]not in pd[x[0]]));pd[x[0]][x[1]]=x
 for x in pop['executions']:ck('unique_exec',(x[1]not in pe[x[0]]));pe[x[0]][x[1]]=x
 dh=Counter();eh=Counter();oh=Counter();un=Counter();ut=Counter();yeshosts=Counter();booldef=0;boolexec=0;bytask={}
 for ref in pop['databases']:
  di=ref['database_index'];ck('rawpin',pin(ref['raw_database']));db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};defs={};seen=set();idh=Counter()
  for d in db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO'):
   v=ss.get(d['hf32Eligible']);r=[di,d['compute_record_rowid'],d['globalTaskId'],d['hf32Eligible'],v,ss.get(d['opType']),ss.get(d['inputDataTypes']),ss.get(d['outputDataTypes']),ss.get(d['inputShapes']),ss.get(d['outputShapes']),d['blockNum'],d['mixBlockNum'],ss.get(d['taskType'])];ck('rawdef',pd[di][d['compute_record_rowid']]==r and v in ['YES','NO','N/A']and d['globalTaskId']not in defs);defs[d['globalTaskId']]=r;dh[v]+=1;idh[d['hf32Eligible'],v]+=1;booldef+=bool(d['hf32Eligible'])
   if v=='YES':yeshosts[r[12]]+=1
  got=Counter();unt=Counter()
  for t in db.execute('SELECT rowid AS TASK_rowid,*FROM TASK'):
   d=defs.get(t['globalTaskId'])
   if d is None:unt['NO_COMPUTE_DEFINITION']+=1;ut[ss.get(t['taskType'],'UNRESOLVED_DB'+str(di)+'_STRING_ID'+str(t['taskType']))]+=1;ck('outside_join_not_fabricated',t['TASK_rowid']not in pe[di]);continue
   r=[di,t['TASK_rowid'],t['globalTaskId'],d[1],t['modelId'],t['contextId'],t['streamId'],t['taskId']];ck('rawTASKjoin',pe[di][t['TASK_rowid']]==r);seen.add(t['TASK_rowid']);bytask[di,t['TASK_rowid']]=d;eh[d[4]]+=1;oh[d[5],d[6],d[4]]+=1;boolexec+=bool(d[3])
  ck('db_coverage',len(pd[di])==ref['definition_rows']==len(defs)and len(seen)==len(pe[di])==ref['joined_TASK_rows']and unt==ref['unmatched_TASK_counts']and not ref['missing_STRING_ID_compute_rows']and not ref['nonunique_globalTaskId_definitions']);ck('DBID_mapping',{(x['raw_id'],x['label']):x['definition_rows']for x in ref['raw_ID_mapping']}==idh);un.update(unt);db.close()
 ck('207750_286950',sum(dh.values())==pop['definition_rows']==207750 and sum(eh.values())==pop['joined_compute_TASK_rows']==286950 and dh==pop['definition_flag_counts']=={'NO':205926,'YES':1120,'N/A':704}and eh==pop['execution_flag_counts']=={'NO':284070,'YES':1120,'N/A':1760}and un==pop['unmatched_TASK_counts']=={'NO_COMPUTE_DEFINITION':821262});ck('OPdtypehist',oh=={(r['opType'],r['inputDataTypes'],r['label']):r['executions']for r in pop['op_dtype_flag_execution_counts']});ck('YES_vector_metadata',yeshosts=={'AI_VECTOR_CORE':1120})
 old147=read(O/'post-pre-native-frontiers-v147.json');old143=read(O/'hc-norm-projection-native-frontiers-v143.json');expected_roles=Counter()
 for r in old147['bindings']:
  for a in r['roles']:expected_roles[r['database_index'],r['TASK']['task_record_rowid'],'v147:'+a,'post-pre-native-frontiers-v147.json',json.dumps([r['database_index'],r['TASK']['task_record_rowid']])]+=1
 for r in old143['records']:
  for a,s in r['stages'].items():
   if s['TASK']is not None:expected_roles[r['database_index'],s['TASK']['task_record_rowid'],'v143:'+a,'hc-norm-projection-native-frontiers-v143.json',json.dumps([r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']])]+=1
 rh=Counter();actual_roles=Counter()
 for r in pop['source_roles']:
  d=bytask[r['database_index'],r['TASK_rowid']];ck('role_metadata',r['compute_record_rowid']==d[1]and r['decoded_hf32_flag']==d[4]and r['opType']==d[5]and r['inputDataTypes']==d[6]and not r['instruction_mode_proved']);actual_roles[r['database_index'],r['TASK_rowid'],r['source_role'],r['source_artifact'],json.dumps(r['source_key'])]+=1;rh[r['source_role'],r['decoded_hf32_flag']]+=1
 ck('28367roles',actual_roles==expected_roles and sum(actual_roles.values())==28367 and len({(r['database_index'],r['TASK_rowid'])for r in pop['source_roles']})==23975 and rh=={(x['role'],x['label']):x['role_bindings']for x in pop['source_role_flag_counts']});ck('APIonly_preserved',len(pop['API_only_roles_retained'])==len(old147['API_only'])==1 and pop['API_only_roles_retained'][0]['API_rowid']==old147['API_only'][0]['API']['api_rowid']and not pop['API_only_roles_retained'][0]['flag_inferred'])
 pr=read(O/'precision-regime-module-projection-v148.json');regs={r['regime_id']:r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']};assign_path=N/'original-profile-cache-v53/analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json';aa=read(assign_path)['execution_assignments'];byreg=defaultdict(Counter);mod=defaultdict(Counter);sets=defaultdict(Counter);seen=set();K=['blockNum','mixBlockNum','opType_text','inputShapes_text','inputDataTypes_text','outputShapes_text','outputDataTypes_text']
 for x,a in zip(pr['execution_projection'],aa):
  d=bytask[x[0],x[1]];r=regs[x[2]];sig=dict(zip(K,[d[10],d[11],d[5],d[8],d[6],d[9],d[7]]));ck('existing_regime_key',x[:2]==a[:2]and x[3]==d[4]and r['module_candidates']==a[3]and r['operator']==d[5]and r['role']==a[5]and r['origin']==a[2]and r['context']==a[6]and all(r['signature'][k]==sig[k]for k in K));ck('unique_regime_task',tuple(x[:2])not in seen);seen.add(tuple(x[:2]));byreg[x[2]][x[3]]+=1;sets[tuple(a[3])][x[3]]+=1
  for m in a[3]:mod[m][x[3]]+=1
 ck('full_regime_tasks',len(seen)==len(pr['execution_projection'])==len(aa)==pr['execution_count']==286950 and len(pr['regimes'])==pr['regime_count']==len(byreg)==2082 and seen==set(bytask))
 for r in pr['regimes']:ck('regime_flags',byreg[r['regime_id']]==r['exported_flag_counts']and sum(byreg[r['regime_id']].values())==r['executions']==regs[r['regime_id']]['executions']and r['existing_signature']==regs[r['regime_id']]['signature']and not r['actual_instruction_precision_or_native_operands_proved'])
 ck('candidate_sets',sets=={tuple(x['module_candidates']):x['flag_counts']for x in pr['module_candidate_set_counts']});ck('26moduleprojection',len(pr['modules'])==26 and all(r['candidate_inclusive_flag_counts']==mod[r['id']]and('NOT_ZERO_UTILIZATION'in r['scope']if not mod[r['id']]else'NOT_EXCLUSIVE_MFU'in r['scope'])for r in pr['modules']))
 co=read(O/'precision-field-source-and-primary-contract-v148.json');ck('outside_type_counts',co['unmatched_TASK_type_STRING_IDS_label_counts']==dict(sorted(ut.items()))and sum(ut.values())==821262 and not co['all_TASK_precision_fields_present'])
 for x in co['parts']:ck('sourceparts',pin(x)and Path(x['path']).read_text().splitlines()[x['first_line']-1:x['last_line']]==x['source'])
 fn=next(x for x in ast.parse(Path(co['parts'][0]['path']).read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='_build_dspark_swa_indices_kernel');calls={ast.unparse(x.func)for x in ast.walk(fn)if isinstance(x,ast.Call)};ck('index_source_only',sorted(calls)==co['index_kernel_AST_calls']and not any('dot'in x or'float'in x for x in calls)and all(not isinstance(x.value,float)for x in ast.walk(fn)if isinstance(x,ast.Constant)));ck('index_labels',co['index_kernel_definition_flag_counts']=={'YES':87,'NO':1}and co['YES_host_type_counts']=={'AI_VECTOR_CORE':1120});hc='\n'.join(co['parts'][-1]['source']);ck('HC_source_manual_modes',hc.index('SetHF32Mode(1)')<hc.index('SetHF32TransMode(1)')<hc.rindex('Mmad(')<hc.index('SetHF32Mode(0)')and not co['actual_executed_instruction_mode_question_closed']and not co['whole_module_or_physical_MFU_closed'])
 for r in co['documents']:
  ck('primary_snapshot',pin(r)and r['http_status']==200 and not r['exact_installed_version_match_proved']);s=html.unescape(re.sub('<[^>]+>',' ',Path(r['path']).read_text()))
  if 'schema'in r['path']:ck('official_STRINGID_contract','hf32Eligible'in s and'STRING_IDS(hf32Eligible)'in s and'YES'in s and'NO'in s)
  if 'legacy'in r['path']:ck('legacy_API_context','SetHF32TransMode'in s and'bool'in s and('zero'in s or'向零'in s))
 rc=read(O/'precision-gated-cycle-and-state-cases-v148.json');keys=set()
 for x in rc['cases']:
  if x['kind']=='HC_COEFFICIENT_ROUTER':
   m=x['rows'];w1=2*m*20480*24;w2=2*m*5120*384;a,b=x['HC_mode'],x['router_mode'];names=['HC_cycle_reference','router_cycle_reference']
  else:
   w1=w2=2*x['rows']*5120*512;a,b=x['KV_mode'],x['score_mode'];names=['KV_cycle_reference','score_cycle_reference']
  u1=24*{'OFF':2048,'ON':4096}[a];u2=24*{'OFF':2048,'ON':4096}[b];f1=float(Fraction(w1,u1));f2=float(Fraction(w2,u2));ck('cycle_formula',x[names[0]]==f1 and x[names[1]]==f2 and x['independent_cut_reference']==max(f1,f2)and x['same_mode_common_pool_cut_reference']==(float(Fraction(w1+w2,u1))if a==b else None)and not x['native_mode_proved']and x['actual_clock_or_time_lower_bound']is None);keys.add((x['kind'],x['rows'],a,b))
 ck('24cases',len(keys)==rc['count']==24 and not rc['new_numeric_physical_bound_or_speedup'])
 for x in rc['mode_automata']:ck('source_state_machine',x['events']==[dict(chunk=j,A2_mode=x['initial_mode']if j==0 else'OFF',AB_mode='ON',after_AB='OFF')for j in range(x['chunks'])]and x['final_mode']=='OFF'and not x['loaded_binary_or_kernel_context_persistence_proved'])
 w=read(O/'precision-flag-rounding-inference-witnesses-v148.json');z=w['wrong_bool_foreignkey'];ck('foreignkey_bool_counterexample',z['enabled_definitions']==booldef==207750 and z['enabled_TASK_joins']==boolexec==286950 and z['actual_exported_YES_definitions']==z['actual_exported_YES_TASK_joins']==1120 and not z['method_legal'])
 for x in w['rounding_cases']:
  v=Fraction(x['input']);u=Fraction(1,2**x['mantissa_fraction_bits']);sg=1 if v>0 else-1;a=abs(v)/u;rtz=sg*int(a)*u;rne=sg*round(a)*u;rna=sg*int(a+Fraction(1,2))*u;ck('roundmode_reference',x['RTZ']==float(rtz)and x['nearest_even']==float(rne)and x['nearest_away']==float(rna)and not x['actual_native_or_frozen_HC_values'])
 if args.preflight:man=read(O/'checkpoint-manifest-v147.json')
 else:
  cp=read(O/'REVIEW-CHECKPOINT-v148.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0);rel=read(O/'candidate-relation-clauses-v148.json');cv=read(O/'candidate-relation-coverage-v148.json');nv=read(O/'candidate-pair-navigation-v148.json');ck('192_26_403_unchanged',len(rel['clauses'])==rel['clause_count']==192 and len(rel['higher_order_groups'])==26 and cv['scoped_conditional_pairs']==192 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==403)
  a=read(O/'operator-analysis-coverage-ledger-v147.json');b=read(O/'operator-analysis-coverage-ledger-v148.json');ck('nativeglobal_unchanged',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']));old={r['regime_id']:r for r in a['regimes']}
  for r in b['regimes']:ck('regime_state_not_promoted',r['state']==old[r['regime_id']]['state']and r['exported_hf32_flag_counts_v148']==byreg[r['regime_id']]and r['instruction_mode_not_inferred_from_flag'])
  a=read(O/'candidate-profile-ledger-v147.json');b=read(O/'candidate-profile-ledger-v148.json');ck('broadergaps',[r.get('conclusion_changing_gap')for r in a['candidates']]==[r.get('conclusion_changing_gap')for r in b['candidates']]);mm=read(O/'module-phase-branch-matrix-v148.json');ck('358_26',len(mm['modules'])==26 and sum(len(x['branches'])for x in mm['modules'])==358)
  for r in read(O/'candidate-record-index-v148.json')['model_candidates']:ck('master',hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
  ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');man=read(O/'checkpoint-manifest-v148.json')
 for x in man['artifacts']:ck('manifest',pin(x))
 result=dict(status='PASS',checks=count,preflight=args.preflight,databases=32,COMPUTE_namespace_definitions=207750,TASK_joins=286950,exported_YES=1120,exported_NO=284070,exported_NA=1760,regimes_flag_reconciled=2082,module_projection=26,source_role_bindings=28367,source_role_TASK_union=23975,API_only_retained=1,mode_reference_cases=24,new_pair_clauses=0,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 if args.preflight:write(O/'precision-field-preflight-v148.json',result)
 print(json.dumps(result))


if __name__=='__main__':main()
