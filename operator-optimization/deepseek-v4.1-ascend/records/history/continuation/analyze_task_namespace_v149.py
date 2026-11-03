"""Raw TASK namespace census and structural wrapper/communication family joins."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
 assert not(O/'REVIEW-CHECKPOINT-v149.json').exists()
 refs=read(O/'precision-field-population-v148.json')['databases'];old=read(O/'communication-native-operator-population-v66.json');co_old={(r['database_index'],r['COMM_OP_rowid']):r for r in old['operations']};out=[];wr=[];sj=[];tables=[];counts=Counter();alltypes=Counter();combined=Counter();unmatched=Counter();schemas=[];source=read(O/'precision-regime-module-projection-v148.json');rid={(x[0],x[1]):x[2]for x in source['execution_projection']};ASSOC_KEYS=['globalPid','deviceId','streamId','taskId'];PICK={'AI_CORE','KERNEL_MIX_AIC','KERNEL_MIX_AIV','KERNEL_AICPU'}
 for ref in refs:
  di=ref['database_index'];path=Path(ref['raw_database']['path']);db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};tt=[dict(x)for x in db.execute('SELECT rowid AS TASK_rowid,*FROM TASK')];cc={x['globalTaskId']:dict(x)for x in db.execute('SELECT rowid AS Compute_rowid,*FROM COMPUTE_TASK_INFO')};sc={x['globalTaskId']:dict(x)for x in db.execute('SELECT rowid AS Schedule_rowid,*FROM COMMUNICATION_SCHEDULE_TASK_INFO')};im=defaultdict(list)
  for x in db.execute('SELECT rowid AS Internal_rowid,*FROM COMMUNICATION_TASK_INFO'):im[x['globalTaskId']].append(x['Internal_rowid'])
  co=[dict(x)for x in db.execute('SELECT rowid AS COMM_OP_rowid,*FROM COMMUNICATION_OP')];ci=defaultdict(set)
  for x in db.execute('SELECT *FROM CONNECTION_IDS'):ci[x['connectionId']].add(x['id'])
  op_by_family=defaultdict(list)
  for x in co:
   for h in ci[x['connectionId']]:op_by_family[h].append(x)
  pyr=defaultdict(list)
  for x in db.execute('SELECT rowid AS API_rowid,*FROM PYTORCH_API WHERE connectionId IS NOT NULL'):pyr[x['connectionId']].append(dict(x,name_text=ss.get(x['name'])))
  bykey=defaultdict(list)
  for t in tt:
   if t['globalTaskId']in cc:bykey[tuple(t[k]for k in ASSOC_KEYS)].append(t)
  rawcounts=Counter();dbcounts=Counter();picked=[]
  for t in tt:
   gt=t['globalTaskId'];name=ss.get(t['taskType']);mask=('C'if gt in cc else'')+('S'if gt in sc else'')+('I'if gt in im else'');rawcounts[name,mask]+=1;counts['raw_TASKs']+=1;alltypes[name]+=1;combined[mask]+=1
   if gt in cc:continue
   unmatched[name]+=1
   if name not in PICK:continue
   counts['picked_TASKs']+=1;picked.append(t['TASK_rowid']);r=dict(database_index=di,TASK=t,task_type_text=name,namespace_memberships=mask,COMPUTE_rowid=None,SCHEDULE_rowid=sc[gt]['Schedule_rowid']if gt in sc else None,COMM_INTERNAL_rowids=im.get(gt,[]),existing_operator_or_phase_promotion=False)
   if gt in sc:
    cand={x['COMM_OP_rowid']:x for h in ci[t['connectionId']]for x in op_by_family[h]};rows=[]
    for j,x in sorted(cand.items()):rows.append(dict(COMM_OP_rowid=j,opId=x['opId'],COMM_connectionId=x['connectionId'],shared_connection_family_IDs=sorted(ci[t['connectionId']]&ci[x['connectionId']]),COMM_opType=ss.get(x['opType']),existing_v66_node_namespace=co_old[di,j]['node_namespace'],existing_v66_semantic_role=co_old[di,j]['semantic_role']))
    r.update(SCHEDULE=sc[gt],schedule_name_text=ss.get(sc[gt]['name']),schedule_opType_text=ss.get(sc[gt]['opType']),connection_family_IDs=sorted(ci[t['connectionId']]),communication_candidates=rows,host_family_APIs=[dict(family_id=h,APIs=pyr[h])for h in sorted(ci[t['connectionId']])]);sj.append(r);dbcounts['schedule_candidates_'+str(len(rows))]+=1
   else:
    cand=[z for z in bykey[tuple(t[k]for k in ASSOC_KEYS)]if t['contextId']==4294967295 and z['contextId']!=4294967295 and(t['connectionId']==-1 or z['connectionId']==t['connectionId'])and t['startNs']<=z['startNs']<=z['endNs']<=t['endNs']];rows=[]
    for z in cand:
     de=cc[z['globalTaskId']];rows.append(dict(TASK=z,COMPUTE_rowid=de['Compute_rowid'],operator=ss.get(de['opType']),host_accelerator_type=ss.get(de['taskType']),task_type_text=ss.get(z['taskType']),existing_regime_id=rid.get((di,z['TASK_rowid'])),same_connection=t['connectionId']==z['connectionId']))
    r.update(runtime_key_fields=ASSOC_KEYS,containing_record_context_is_sentinel=t['contextId']==4294967295,inner_candidates=rows,identity_grade='SAME_RUNTIME_KEY_CONNECTION_GUARD_CONTEXT_AND_UNIQUE_CONTAINMENT'if len(rows)==1 else'UNRESOLVED_STRUCTURAL_CONTAINER',time_nearest_or_shape_join_used=False,contains_not_new_independent_operator_proof=True);wr.append(r);dbcounts['wrapper_candidates_'+str(len(rows))]+=1
   out.append(r)
  overlap=dict(C_S=len(set(cc)&set(sc)),C_I=len(set(cc)&set(im)),S_I=len(set(sc)&set(im)))
  tables.append(dict(database_index=di,raw_database=pin(path),TASK_rows=len(tt),COMPUTE_definitions=len(cc),SCHEDULE_definitions=len(sc),COMM_INTERNAL_distinct_globalTaskId=len(im),COMM_INTERNAL_rows=sum(map(len,im.values())),COMM_OP_rows=len(co),globalTaskId_namespace_overlap=overlap,raw_TASK_type_membership_counts=[dict(task_type=k[0],namespace_membership=k[1],rows=v)for k,v in sorted(rawcounts.items())],selected_TASK_rowids=picked,selected_join_counts=dict(dbcounts)));schemas.append(dict(database_index=di,tables={name:[dict(x)for x in db.execute('PRAGMA table_info('+name+')')]for name in ['TASK','COMPUTE_TASK_INFO','COMMUNICATION_TASK_INFO','COMMUNICATION_SCHEDULE_TASK_INFO','CONNECTION_IDS','PYTORCH_API']}));db.close()
 data=dict(databases=tables,schemas=schemas,accelerator_labeled_records=out,wrapper_records=wr,schedule_records=sj,raw_TASK_rows=counts['raw_TASKs'],selected_TASK_rows=counts['picked_TASKs'],wrapper_rows=len(wr),schedule_rows=len(sj),raw_namespace_membership_counts=dict(combined),raw_TASK_type_counts=dict(alltypes),outside_COMPUTE_type_counts=dict(unmatched),wrapper_candidate_hist=dict(Counter(len(r['inner_candidates'])for r in wr)),schedule_candidate_hist=dict(Counter(len(r['communication_candidates'])for r in sj)),new_native_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 write(O/'task-namespace-census-v149.json',data);print(json.dumps({k:data[k]for k in ['raw_TASK_rows','selected_TASK_rows','wrapper_rows','schedule_rows','raw_namespace_membership_counts','wrapper_candidate_hist','schedule_candidate_hist','new_device_work','overall_complete']}))


if __name__=='__main__':main()
