"""Exact comm/internal and single-thread call relationships; not tensor edges."""
from pathlib import Path
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib,json,sqlite3,statistics
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def interval(a,b):
 x,y=a;u,v=b
 return 'EQUAL'if(x,y)==(u,v)else'SCHEDULE_INSIDE_COMM'if u<=x<=y<=v else'COMM_INSIDE_SCHEDULE'if x<=u<=v<=y else'DISJOINT'if y<u or v<x else'OVERLAP'


def main():
 assert not(O/'REVIEW-CHECKPOINT-v149.json').exists()
 c=read(O/'task-namespace-census-v149.json');old=read(O/'communication-native-operator-population-v66.json');oldmap={(r['database_index'],r['COMM_OP_rowid']):r for r in old['operations']};sb=defaultdict(list);wr=defaultdict(list)
 for r in c['schedule_records']:sb[r['database_index']].append(r)
 for r in c['wrapper_records']:wr[r['database_index']].append(r)
 binds=[];internals=[];copy=[];rest=Counter();part=Counter();rel=Counter();wstats=[];internal_hist=Counter();schemas=[];fallback=[];periods=defaultdict(list);imeta=[]
 for ref in c['databases']:
  di=ref['database_index'];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};tasks=[dict(x)for x in db.execute('SELECT rowid AS TASK_rowid,*FROM TASK')];tt={x['TASK_rowid']:x for x in tasks};ct={x[0]for x in db.execute('SELECT globalTaskId FROM COMPUTE_TASK_INFO')};sg={x[0]for x in db.execute('SELECT globalTaskId FROM COMMUNICATION_SCHEDULE_TASK_INFO')};co={x['COMM_OP_rowid']:dict(x)for x in db.execute('SELECT rowid AS COMM_OP_rowid,*FROM COMMUNICATION_OP')};opids={x['opId']:x['COMM_OP_rowid']for x in co.values()};by_stamp=defaultdict(list);ci=defaultdict(set);pa=defaultdict(list);ca=defaultdict(list)
  for t in tasks:by_stamp[t['globalTaskId'],t['startNs']].append(t)
  for x in db.execute('SELECT *FROM CONNECTION_IDS'):ci[x['connectionId']].add(x['id'])
  for x in db.execute('SELECT rowid AS API_rowid,*FROM PYTORCH_API WHERE connectionId IS NOT NULL'):pa[x['connectionId']].append(dict(x,name_text=ss.get(x['name'])))
  allcann=[dict(x,name_text=ss.get(x['name']))for x in db.execute('SELECT rowid AS CANN_API_rowid,*FROM CANN_API')]
  for x in allcann:ca[x['connectionId']].append(x)
  for r in sb[di]:
   t=r['TASK'];cand=r['communication_candidates'];proof={};grade='EXACT_SHARED_CONNECTION_IDS_FAMILY'
   if len(cand)==1:
    oid=cand[0]['COMM_OP_rowid'];proof=dict(family_IDs=cand[0]['shared_connection_family_IDs'],host_family_APIs=r['host_family_APIs'])
   else:
    assert not cand;hits=[]
    for comm in co.values():
     for par in ca[comm['connectionId']]:
      for kid in ca[t['connectionId']]:
       if par['globalTid']!=kid['globalTid']or par['connectionId']==kid['connectionId']:continue
       if not(int(par['startNs'])<=int(kid['startNs'])<=int(kid['endNs'])<=int(par['endNs'])):continue
       typed=[x for x in allcann if x['globalTid']==par['globalTid']and x['type']==20000 and x['name_text']==ss[comm['opType']]and(x['startNs'],x['endNs'])==(par['startNs'],par['endNs'])]
       if typed:hits.append(dict(COMM_OP_rowid=comm['COMM_OP_rowid'],parent_CANN=par,kernel_CANN=kid,typed_same_interval_CANN=typed))
    assert len(hits)==1,(di,t['TASK_rowid'],len(hits));proof=hits[0];oid=proof['COMM_OP_rowid'];grade='EXACT_NATIVE_CID_SAME_THREAD_UNIQUE_NESTED_CANN_CALL';fallback.append(dict(database_index=di,TASK_rowid=t['TASK_rowid'],**proof))
   comm=co[oid];prev=oldmap[di,oid];assert prev['node_namespace']=='NO_COMPUTEINFO_NODE'and comm['deviceId']==t['deviceId'];ri=interval((t['startNs'],t['endNs']),(comm['startNs'],comm['endNs']));rel[ri]+=1
   b=dict(database_index=di,COMM_OP_rowid=oid,opId=comm['opId'],TASK_rowid=t['TASK_rowid'],SCHEDULE_rowid=r['SCHEDULE_rowid'],schedule_definition=r['SCHEDULE'],TASK=t,COMM_OP=comm,schedule_name=r['schedule_name_text'],COMM_operator=ss[comm['opType']],existing_v66_semantic_role=prev['semantic_role'],existing_v66_source_grade=prev['source_grade'],grade=grade,proof=proof,recorded_interval_relation=ri,recorded_schedule_duration_ns=t['endNs']-t['startNs'],recorded_COMM_duration_ns=comm['endNs']-comm['startNs'],task_precision_or_new_dataflow_or_crossrank_clock_inferred=False);binds.append(b);periods[b['COMM_operator']].append([b['recorded_schedule_duration_ns'],b['recorded_COMM_duration_ns']])
  used=set();byinfo_task=Counter();ig=set();ni=0
  for x in db.execute('SELECT rowid AS Internal_rowid,*FROM COMMUNICATION_TASK_INFO'):
   z=by_stamp[x['globalTaskId'],x['timestampNs']];internal_hist[len(z)]+=1;assert len(z)==1,(di,x['Internal_rowid'],len(z));t=z[0];oi=opids[x['opId']];internals.append([di,x['Internal_rowid'],x['globalTaskId'],x['timestampNs'],t['TASK_rowid'],x['opId'],oi,x['globalTaskId']in ct]);used.add(t['TASK_rowid']);byinfo_task[t['TASK_rowid']]+=1;ig.add(x['globalTaskId']);ni+=1
  imeta.append(dict(database_index=di,internal_records=ni,matched_TASK_rows=len(used),TASK_record_multiplicity=dict(Counter(byinfo_task.values())),metadata_globalTaskId_count=len(ig)))
  ws={r['TASK']['TASK_rowid']for r in wr[di]}
  for r in wr[di]:
   assert len(r['inner_candidates'])==1;z=r['inner_candidates'][0];t=r['TASK'];a=z['TASK'];wstats.append(dict(database_index=di,outer_TASK_rowid=t['TASK_rowid'],inner_TASK_rowid=a['TASK_rowid'],existing_regime_id=z['existing_regime_id'],operator=z['operator'],outer_type=r['task_type_text'],recorded_outer_duration_ns=t['endNs']-t['startNs'],recorded_inner_duration_ns=a['endNs']-a['startNs'],recorded_leading_margin_ns=a['startNs']-t['startNs'],recorded_trailing_margin_ns=t['endNs']-a['endNs'],recorded_union_duration_ns=t['endNs']-t['startNs'],naive_sum_duration_ns=t['endNs']-t['startNs']+a['endNs']-a['startNs'],removable_overhead_or_instruction_count_inferred=False))
  for t in tasks:
   tid=t['TASK_rowid'];gt=t['globalTaskId'];label=ss.get(t['taskType']);tags=[]
   if gt in ct:tags.append('COMPUTE')
   if tid in used:tags.append('COMM_INTERNAL')
   if gt in sg:tags.append('COMM_SCHEDULE')
   if tid in ws:tags.append('RUNTIME_CONTAINER')
   part['+'.join(tags)if tags else'OTHER_RUNTIME_RECORD']+=1
   if not tags:
    rest[label]+=1
    if label=='MEMCPY_ASYNC':copy.append(dict(database_index=di,TASK=t,connection_family_IDs=sorted(ci[t['connectionId']]),PYTORCH_API_candidates=[dict(family_ID=h,API=a)for h in sorted(ci[t['connectionId']])for a in pa[h]],CANN_API_same_connection=ca[t['connectionId']],byte_count_direction_address_or_phase_inferred=False))
  schemas.append(dict(database_index=di,all_tables=[x[0]for x in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")],TASK_columns=[x['name']for x in db.execute('PRAGMA table_info(TASK)')],CANN_API_columns=[x['name']for x in db.execute('PRAGMA table_info(CANN_API)')]));db.close()
 assert len(binds)==len({(r['database_index'],r['COMM_OP_rowid'])for r in binds})==len({(r['database_index'],r['TASK_rowid'])for r in binds})==3192
 assert {(r['database_index'],r['COMM_OP_rowid'])for r in binds}=={k for k,r in oldmap.items()if r['node_namespace']=='NO_COMPUTEINFO_NODE'}
 current=deepcopy(old);bm={(r['database_index'],r['COMM_OP_rowid']):r for r in binds}
 for r in current['operations']:
  z=bm.get((r['database_index'],r['COMM_OP_rowid']))
  if z:r['schedule_control_association_v149']=dict(TASK_rowid=z['TASK_rowid'],SCHEDULE_rowid=z['SCHEDULE_rowid'],grade=z['grade'],same_native_connection_and_identical_interval_inferred=False)
 current.update(parent='communication-native-operator-population-v66.json',schedule_control_associations=3192,noncompute_ops_to_schedule_one_to_one_join_proved=True,join_criterion_v149='Recordedcall-controlassociation:3191sharedCONNECTION_IDSfamilies+1same-threadnestedCANNlaunchunderexactCOMMnativeCID/typedHCCLcall. NotTensor/actualclock/precisionproof.',prior_v66_same_connection_equal_interval_rule_did_not_prove_join=True,new_operator_calls_added=0,overall_complete=False);write(O/'communication-native-operator-population-v149.json',current)
 st=lambda v:dict(count=len(v),min=min(v),median=statistics.median(v),max=max(v))
 write(O/'ta<REDACTED_TOKEN>.json',dict(schedule_bindings=binds,schedule_count=3192,connection_family_count=3191,CANN_nesting_count=1,CANN_fallback=fallback,recorded_interval_relations=dict(rel),operator_recorded_time_statistics={k:dict(schedule_ns=st([x[0]for x in v]),COMM_ns=st([x[1]for x in v]))for k,v in periods.items()},source_or_precision_or_global_phase_promotions=0,new_device_work=False,overall_complete=False))
 write(O/'ta<REDACTED_TOKEN>.json',dict(columns=['database_index','Internal_rowid','globalTaskId','timestampNs','TASK_rowid','opId','COMM_OP_rowid','also_COMPUTE'],records=internals,count=len(internals),exact_globalTaskId_start_timestamp_hist=dict(internal_hist),per_database=imeta,byte_count_not_universal_physical_wire_lowerbound=True,overall_complete=False))
 write(O/'ta<REDACTED_TOKEN>.json',dict(records=wstats,count=len(wstats),scope='Runtimekey/context/connection/recordedcontainmentassociationonly.Notnewarithmeticoperators;outerspanstillhasunknownoverhead/work;do notsumenclosingintervalsandcallitwalltimegain.No tensor/epoch/binaryproof.',new_device_work=False,overall_complete=False))
 write(O/'ta<REDACTED_TOKEN>.json',dict(raw_TASK_partition=dict(part),raw_TASK_total=sum(part.values()),remaining_runtime_type_counts=dict(rest),remaining_runtime_records=sum(rest.values()),runtime_copy_records=copy,runtime_copy_count=len(copy),schemas=schemas,remaining_runtime_is_not_zero_work=True,copy_payload_fields_not_present_in_TASK_CANN_tables=True,bytes_direction_or_allsourceowners_inferred=False,next_question='Reconcile16342MEMCPY_ASYNCsource/controlfamiliesandavailablepayload/phaseevidencebeforereturningtocandidatepairs;runtimeevent/controlrecordownershipandallreader/queuegatesremain.',new_device_work=False,overall_complete=False))
 print(json.dumps(dict(raw_TASK_rows=sum(part.values()),partition=dict(part),runtime_container_records=len(wstats),schedule_associations=len(binds),schedule_grades=dict(Counter(r['grade']for r in binds)),internal_records=len(internals),internal_exact_stamp_hist=dict(internal_hist),remaining_runtime_rows=sum(rest.values()),remaining_runtime_types=dict(rest),runtime_copy_records=len(copy),new_operator_calls=0,overall_complete=False)))


if __name__=='__main__':main()
