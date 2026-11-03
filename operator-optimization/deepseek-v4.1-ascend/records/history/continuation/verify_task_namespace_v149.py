"""Rebuild namespace/call/container relationships from immutable SQLite tables."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,sqlite3,statistics
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=0
 def ck(k,v):
  nonlocal checks
  assert v,k
  checks+=1
 def pin(x):
  b=Path(x['path']).read_bytes();return len(b)==x['bytes']and hashlib.sha256(b).hexdigest()==x['sha256']
 for x in read(O/'task-namespace-inputs-v149.json')['inputs']:ck('input',pin(x))
 c=read(O/'task-namespace-census-v149.json');b=read(O/'ta<REDACTED_TOKEN>.json');i=read(O/'ta<REDACTED_TOKEN>.json');e=read(O/'ta<REDACTED_TOKEN>.json');r=read(O/'ta<REDACTED_TOKEN>.json');wr=defaultdict(list);sr=defaultdict(list);ir=defaultdict(dict);cp=defaultdict(dict);er={(x['database_index'],x['outer_TASK_rowid']):x for x in e['records']};by_b={};allpart=Counter();alltypes=Counter();untypes=Counter();masks=Counter();rest=Counter();counts=Counter();inner_unique=set();copyn=0;rels=Counter();familyn=0;fallbackn=0;known_ids={ (x[0],x[1]):x[2]for x in read(O/'precision-regime-module-projection-v148.json')['execution_projection']};old=read(O/'communication-native-operator-population-v66.json');oldops={(x['database_index'],x['COMM_OP_rowid']):x for x in old['operations']};new=read(O/'communication-native-operator-population-v149.json')
 for x in c['wrapper_records']:wr[x['database_index']].append(x)
 for x in c['schedule_records']:sr[x['database_index']].append(x)
 for x in b['schedule_bindings']:ck('unique_COMM_binding',(x['database_index'],x['COMM_OP_rowid'])not in by_b);by_b[x['database_index'],x['COMM_OP_rowid']]=x
 for x in i['records']:ck('unique_internal',(x[1]not in ir[x[0]]));ir[x[0]][x[1]]=x
 for x in r['runtime_copy_records']:ck('unique_copy',x['TASK']['TASK_rowid']not in cp[x['database_index']]);cp[x['database_index']][x['TASK']['TASK_rowid']]=x
 for ref in c['databases']:
  di=ref['database_index'];ck('rawpin',pin(ref['raw_database']));db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};ts={x['TASK_rowid']:dict(x)for x in db.execute('SELECT rowid AS TASK_rowid,*FROM TASK')};cc={x['globalTaskId']:dict(x)for x in db.execute('SELECT rowid AS Compute_rowid,*FROM COMPUTE_TASK_INFO')};sc={x['globalTaskId']:dict(x)for x in db.execute('SELECT rowid AS Schedule_rowid,*FROM COMMUNICATION_SCHEDULE_TASK_INFO')};co={x['COMM_OP_rowid']:dict(x)for x in db.execute('SELECT rowid AS COMM_OP_rowid,*FROM COMMUNICATION_OP')};opid={x['opId']:x['COMM_OP_rowid']for x in co.values()};ci=defaultdict(set);pa=defaultdict(list);ca=defaultdict(list);bkey=defaultdict(list);stamp=defaultdict(list);im=defaultdict(list)
  for x in db.execute('SELECT *FROM CONNECTION_IDS'):ci[x['connectionId']].add(x['id'])
  for x in db.execute('SELECT rowid AS API_rowid,*FROM PYTORCH_API WHERE connectionId IS NOT NULL'):pa[x['connectionId']].append(dict(x,name_text=ss.get(x['name'])))
  ac=[dict(x,name_text=ss.get(x['name']))for x in db.execute('SELECT rowid AS CANN_API_rowid,*FROM CANN_API')]
  for x in ac:ca[x['connectionId']].append(x)
  for t in ts.values():
   stamp[t['globalTaskId'],t['startNs']].append(t)
   if t['globalTaskId']in cc:bkey[t['globalPid'],t['deviceId'],t['streamId'],t['taskId']].append(t)
  internal_tasks=set();info_task_count=Counter();ni=0
  for x in db.execute('SELECT rowid AS Internal_rowid,*FROM COMMUNICATION_TASK_INFO'):
   zz=stamp[x['globalTaskId'],x['timestampNs']];ck('internal_exact_start_unique',len(zz)==1);t=zz[0];expect=[di,x['Internal_rowid'],x['globalTaskId'],x['timestampNs'],t['TASK_rowid'],x['opId'],opid[x['opId']],x['globalTaskId']in cc];ck('internal_record',ir[di][x['Internal_rowid']]==expect);internal_tasks.add(t['TASK_rowid']);info_task_count[t['TASK_rowid']]+=1;im[x['globalTaskId']].append(x['Internal_rowid']);ni+=1
  ck('internal_bijection',ni==len(ir[di])==len(internal_tasks)and set(info_task_count.values())=={1});ws=set();sn=set()
  for x in wr[di]:
   t=x['TASK'];ck('outer_raw',t==ts[t['TASK_rowid']]and t['globalTaskId']not in cc and ss[t['taskType']]==x['task_type_text']);z=[q for q in bkey[t['globalPid'],t['deviceId'],t['streamId'],t['taskId']]if t['contextId']==4294967295 and q['contextId']!=4294967295 and(t['connectionId']==-1 or q['connectionId']==t['connectionId'])and t['startNs']<=q['startNs']<=q['endNs']<=t['endNs']];ck('single_structured_container',len(z)==1 and len(x['inner_candidates'])==1 and x['inner_candidates'][0]['TASK']==z[0]and not x['time_nearest_or_shape_join_used']);q=z[0];d=cc[q['globalTaskId']];ck('existing_inner_descriptor',x['inner_candidates'][0]['COMPUTE_rowid']==d['Compute_rowid']and x['inner_candidates'][0]['operator']==ss[d['opType']]and x['inner_candidates'][0]['existing_regime_id']==known_ids[di,q['TASK_rowid']]);ck('unique_inner',(di,q['TASK_rowid'])not in inner_unique);inner_unique.add((di,q['TASK_rowid']));ws.add(t['TASK_rowid']);a=er[di,t['TASK_rowid']];od=t['endNs']-t['startNs'];nd=q['endNs']-q['startNs'];ck('union_not_sum',a['inner_TASK_rowid']==q['TASK_rowid']and a['recorded_outer_duration_ns']==a['recorded_union_duration_ns']==od and a['recorded_inner_duration_ns']==nd and a['naive_sum_duration_ns']==od+nd and a['recorded_leading_margin_ns']==q['startNs']-t['startNs']and a['recorded_trailing_margin_ns']==t['endNs']-q['endNs']and not a['removable_overhead_or_instruction_count_inferred'])
  for x in sr[di]:
   t=x['TASK'];sn.add(t['TASK_rowid']);ck('schedule_raw',t==ts[t['TASK_rowid']]and x['SCHEDULE']==sc[t['globalTaskId']]and t['globalTaskId']not in cc and ss[t['taskType']]=='KERNEL_AICPU');candidates={q['COMM_OP_rowid']:q for q in co.values()if ci[t['connectionId']]&ci[q['connectionId']]};ck('family_candidates_all_ops',set(candidates)=={z['COMM_OP_rowid']for z in x['communication_candidates']})
   bb=next(z for z in b['schedule_bindings']if z['database_index']==di and z['TASK_rowid']==t['TASK_rowid']);oid=bb['COMM_OP_rowid'];comm=co[oid];ck('COMM_raw',bb['TASK']==t and bb['COMM_OP']==comm and bb['schedule_definition']==sc[t['globalTaskId']]and comm['deviceId']==t['deviceId']and oldops[di,oid]['node_namespace']=='NO_COMPUTEINFO_NODE'and bb['existing_v66_semantic_role']==oldops[di,oid]['semantic_role']and bb['existing_v66_source_grade']==oldops[di,oid]['source_grade'])
   if bb['grade']=='EXACT_SHARED_CONNECTION_IDS_FAMILY':
    familyn+=1;ck('direct_family_unique',len(candidates)==1 and oid in candidates and bb['proof']['family_IDs']==sorted(ci[t['connectionId']]&ci[comm['connectionId']])and bb['proof']['host_family_APIs']==x['host_family_APIs'])
    for fam in x['host_family_APIs']:ck('family_API_raw',fam['APIs']==pa[fam['family_id']]and fam['family_id']in ci[t['connectionId']])
   else:
    fallbackn+=1;ck('fallback_no_family',not candidates and di==8 and t['TASK_rowid']==22599 and oid==232);hits=[]
    for q in co.values():
     for par in ca[q['connectionId']]:
      for kid in ca[t['connectionId']]:
       typed=[z for z in ac if z['globalTid']==par['globalTid']and z['type']==20000 and z['name_text']==ss[q['opType']]and(z['startNs'],z['endNs'])==(par['startNs'],par['endNs'])]
       if typed and par['globalTid']==kid['globalTid']and par['connectionId']!=kid['connectionId']and int(par['startNs'])<=int(kid['startNs'])<=int(kid['endNs'])<=int(par['endNs']):hits.append(dict(COMM_OP_rowid=q['COMM_OP_rowid'],parent_CANN=par,kernel_CANN=kid,typed_same_interval_CANN=typed))
    ck('unique_native_CANN_nesting',len(hits)==1 and hits[0]==bb['proof']and bb['proof']['parent_CANN']['globalTid']>>32==t['globalPid']);q=read(O/'communication-role-contract-v16.json')['queue_source'];ck('prior_queue_grade_retained',q['database_index']==di and q['COMM_OP_rowid']==oid and bb['existing_v66_source_grade']=='REUSED_EXPLICIT_QUEUE_FLOW_PROOF')
   name=sc[t['globalTaskId']]['name'];ck('schedule_kind_compatible',{'allgatherAicpuKernel':'hcom_allGather_','alltoallAicpuKernel':'hcom_alltoall_','allreduceAicpuKernel':'hcom_allReduce_','reduce_scatterAicpuKernel':'hcom_reduceScatter_'}[ss[name]]==ss[comm['opType']]);a0,a1=t['startNs'],t['endNs'];b0,b1=comm['startNs'],comm['endNs'];rr='EQUAL'if(a0,a1)==(b0,b1)else'SCHEDULE_INSIDE_COMM'if b0<=a0<=a1<=b1 else'COMM_INSIDE_SCHEDULE'if a0<=b0<=b1<=a1 else'DISJOINT'if a1<b0 or b1<a0 else'OVERLAP';ck('recorded_intervals',bb['recorded_interval_relation']==rr and not bb['task_precision_or_new_dataflow_or_crossrank_clock_inferred']);rels[rr]+=1
  picks=[];rawtype=Counter();rawmask=Counter();restthis=Counter()
  for t in ts.values():
   tid=t['TASK_rowid'];gt=t['globalTaskId'];ty=ss.get(t['taskType']);mask=('C'if gt in cc else'')+('S'if gt in sc else'')+('I'if gt in im else'');masks[mask]+=1;rawtype[ty]+=1;rawmask[ty,mask]+=1;alltypes[ty]+=1;tags=[]
   if gt in cc:tags.append('COMPUTE')
   if tid in internal_tasks:tags.append('COMM_INTERNAL')
   if gt in sc:tags.append('COMM_SCHEDULE')
   if tid in ws:tags.append('RUNTIME_CONTAINER')
   allpart['+'.join(tags)if tags else'OTHER_RUNTIME_RECORD']+=1
   if gt not in cc:
    untypes[ty]+=1
    if ty in ['AI_CORE','KERNEL_MIX_AIC','KERNEL_MIX_AIV','KERNEL_AICPU']:picks.append(tid);ck('selected_covered',tid in ws or tid in sn)
   if not tags:
    rest[ty]+=1;restthis[ty]+=1
    if ty=='MEMCPY_ASYNC':
     copyn+=1;z=cp[di][tid];ck('runtime_copy_raw',z['TASK']==t and z['connection_family_IDs']==sorted(ci[t['connectionId']])and z['CANN_API_same_connection']==ca[t['connectionId']]and z['PYTORCH_API_candidates']==[dict(family_ID=h,API=a)for h in sorted(ci[t['connectionId']])for a in pa[h]]and not z['byte_count_direction_address_or_phase_inferred'])
  ck('whole_db_partition',len(ts)==ref['TASK_rows']and picks==ref['selected_TASK_rowids']and {(x['task_type'],x['namespace_membership']):x['rows']for x in ref['raw_TASK_type_membership_counts']}==rawmask);ck('def_namespace_overlap',ref['globalTaskId_namespace_overlap']==dict(C_S=len(set(cc)&set(sc)),C_I=len(set(cc)&set(im)),S_I=len(set(sc)&set(im))));db.close()
 ck('full_raw_partition',sum(allpart.values())==c['raw_TASK_rows']==r['raw_TASK_total']==1108212 and allpart==r['raw_TASK_partition']=={'OTHER_RUNTIME_RECORD':240283,'COMPUTE':275422,'COMPUTE+COMM_INTERNAL':11528,'RUNTIME_CONTAINER':31627,'COMM_SCHEDULE':3192,'COMM_INTERNAL':546160});ck('whole_type_census',alltypes==c['raw_TASK_type_counts']and untypes==c['outside_COMPUTE_type_counts']and masks==c['raw_namespace_membership_counts']);ck('all_34819',len(inner_unique)==e['count']==31627 and sum(len(v)for v in sr.values())==3192 and c['selected_TASK_rows']==34819 and familyn==3191 and fallbackn==1)
 ck('all_557688',i['count']==sum(map(len,ir.values()))==557688 and i['exact_globalTaskId_start_timestamp_hist']=={'1':557688});ck('all_schedule_population',set(by_b)=={k for k,z in oldops.items()if z['node_namespace']=='NO_COMPUTEINFO_NODE'}and len(by_b)==b['schedule_count']==3192 and rels==b['recorded_interval_relations']=={'DISJOINT':2922,'SCHEDULE_INSIDE_COMM':247,'OVERLAP':23})
 ck('14720_ops_preserved',len(new['operations'])==len(old['operations'])==new['operation_count']==14720 and new['schedule_control_associations']==3192 and new['noncompute_ops_to_schedule_one_to_one_join_proved'])
 for z in new['operations']:
  q=dict(z);extra=q.pop('schedule_control_association_v149',None);ck('old_role_not_promoted',q==oldops[q['database_index'],q['COMM_OP_rowid']]and((extra is not None)==((q['database_index'],q['COMM_OP_rowid'])in by_b)))
 ck('runtime_still_open',rest==r['remaining_runtime_type_counts']and sum(rest.values())==r['remaining_runtime_records']==240283 and copyn==len(r['runtime_copy_records'])==r['runtime_copy_count']==16342 and not r['bytes_direction_or_allsourceowners_inferred']);ck('copy_payload_column_scope',all(not({'size','bytes','src','dst','memcpyKind'}&set(z['TASK_columns']+z['CANN_API_columns']))for z in r['schemas']))
 co=read(O/'ta<REDACTED_TOKEN>.json');ck('boundary_not_semantics',co['raw_TASK_partition']==dict(allpart)and not co['semantic_dataflow_or_all_task_ownership_closed']and not co['new_device_work']);w=read(O/'ta<REDACTED_TOKEN>.json')
 for x in w['same_observed_times_different_bank_readers']:ck('staging_nonidentifiability',x['early_staging_at1']['output_at8']==x['old_value']==x['reference_expected']and x['late_source_read_at6']['output_at8']==(x['new_value']if x['reuse_source_bank_at3']else x['old_value'])and not x['actual_native_reader_model_bound'])
 x=w['namespace_overlap'];ck('namespace_union',x['correct_union_rows']==x['COMPUTE_TASK_rows']+x['COMM_INTERNAL_TASK_rows']-x['intersection_same_TASK_rows']==833110 and x['wrong_naive_sum_rows']==844638);x=w['contained_record_time'];ck('time_union',x['recorded_union']==10 and x['raw_duration_sum']==16 and not x['gain_or_actual_overhead_inferred'])
 if args.preflight:man=read(O/'checkpoint-manifest-v148.json')
 else:
  ck('progress',read(O/'REVIEW-CHECKPOINT-v149.json')['status']=='PROGRESS'and not read(O/'REVIEW-CHECKPOINT-v149.json')['overall_complete']);a=read(O/'operator-analysis-coverage-ledger-v148.json');z=read(O/'operator-analysis-coverage-ledger-v149.json');ck('old_native_global',all(a[k]==z[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']));ck('old_regime_states',a['regimes']==z['regimes']);a=read(O/'candidate-profile-ledger-v148.json');z=read(O/'candidate-profile-ledger-v149.json');ck('old_broad_gaps',[x.get('conclusion_changing_gap')for x in a['candidates']]==[x.get('conclusion_changing_gap')for x in z['candidates']]);rel=read(O/'candidate-relation-clauses-v149.json');cv=read(O/'candidate-relation-coverage-v149.json');ck('192_26_403',len(rel['clauses'])==192 and len(rel['higher_order_groups'])==26 and cv['remaining_pair_relations']==403);ob=read(O/'coverage-current-obligations-v149.json');ck('runtime_obligation',any(x['id']=='G14'and'16342'in x['remaining']for x in ob['obligations']));mx=read(O/'module-phase-branch-matrix-v149.json');ck('358_26',len(mx['modules'])==26 and sum(len(x['branches'])for x in mx['modules'])==358)
  for z in read(O/'candidate-record-index-v149.json')['model_candidates']:ck('master',hashlib.sha256(Path(z['master_record']).read_bytes()).hexdigest()==z['master_sha256'])
  ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');man=read(O/'checkpoint-manifest-v149.json')
 for x in man['artifacts']:ck('manifest',pin(x))
 result=dict(status='PASS',checks=checks,preflight=args.preflight,raw_TASK_rows=1108212,accelerator_labeled_records=34819,containers=31627,HCCL_schedule_control_associations=3192,internal_record_associations=557688,C_I_same_TASK_overlap=11528,HCCL_calls=14720,runtime_copy_pending=16342,other_runtime_records=240283,new_pair_clauses=0,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 if args.preflight:write(O/'task-namespace-preflight-v149.json',result)
 print(json.dumps(result))


if __name__=='__main__':main()
