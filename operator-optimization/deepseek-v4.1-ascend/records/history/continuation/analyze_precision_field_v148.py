"""Decode all saved COMPUTE hf32 flags through local STRING_IDS, not bool casts."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
 assert not(O/'REVIEW-CHECKPOINT-v148.json').exists()
 refs=read(O/'post-pre-native-frontiers-v147.json')['databases'];defs=[];execs=[];dbs=[];dh=Counter();eh=Counter();oh=Counter();sample={};unmatched=Counter();dcols=['database_index','compute_record_rowid','globalTaskId','raw_hf32Eligible','decoded_label','opType','inputDataTypes','outputDataTypes','inputShapes','outputShapes','blockNum','mixBlockNum','host_task_type'];ecols=['database_index','TASK_rowid','globalTaskId','compute_record_rowid','modelId','contextId','streamId','taskId'];bytask={}
 for ref in refs:
  di=ref['database_index'];path=Path(ref['raw_database']['path']);db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};dr=list(db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO'));bygid=defaultdict(list);h=Counter();missing=[];dups=[]
  for d in dr:
   v=ss.get(d['hf32Eligible']);op=ss.get(d['opType']);dt=ss.get(d['inputDataTypes']);bygid[d['globalTaskId']].append(d);dh[v]+=1;h[d['hf32Eligible'],v]+=1
   if v is None:missing.append(d['compute_record_rowid'])
   defs.append([di,d['compute_record_rowid'],d['globalTaskId'],d['hf32Eligible'],v,op,dt,ss.get(d['outputDataTypes']),ss.get(d['inputShapes']),ss.get(d['outputShapes']),d['blockNum'],d['mixBlockNum'],ss.get(d['taskType'])])
  du=[dict(globalTaskId=k,compute_rowids=[x['compute_record_rowid']for x in v])for k,v in bygid.items()if len(v)!=1];uc=Counter();cnt=0
  for t in db.execute('SELECT rowid AS TASK_rowid,*FROM TASK'):
   hits=bygid.get(t['globalTaskId'],[])
   if len(hits)!=1:uc['NO_COMPUTE_DEFINITION'if not hits else'NONUNIQUE_COMPUTE_DEFINITION']+=1;continue
   d=hits[0];v=ss.get(d['hf32Eligible']);op=ss.get(d['opType']);dt=ss.get(d['inputDataTypes']);row=[di,t['TASK_rowid'],t['globalTaskId'],d['compute_record_rowid'],t['modelId'],t['contextId'],t['streamId'],t['taskId']];execs.append(row);bytask[di,t['TASK_rowid']]=(d['compute_record_rowid'],v,op,dt);eh[v]+=1;oh[op,dt,v]+=1;cnt+=1
  unmatched.update(uc);dbs.append(dict(database_index=di,raw_database=pin(path),definition_rows=len(dr),joined_TASK_rows=cnt,unmatched_TASK_counts=dict(uc),raw_ID_mapping=[dict(raw_id=i,label=v,definition_rows=c)for(i,v),c in sorted(h.items())],missing_STRING_ID_compute_rows=missing,nonunique_globalTaskId_definitions=du));db.close()
 roles=[];rhist=Counter();api=[]
 def role(di,tid,name,source,key):
  z=bytask[di,tid];roles.append(dict(database_index=di,TASK_rowid=tid,source_role=name,source_artifact=source,source_key=key,compute_record_rowid=z[0],decoded_hf32_flag=z[1],opType=z[2],inputDataTypes=z[3],instruction_mode_proved=False));rhist[name,z[1]]+=1
 name='post-pre-native-frontiers-v147.json'
 for r in read(O/name)['API_only']:
  api.append(dict(source=name,key=[r['database_index'],r['API']['api_rowid']],role=r['role'],API_rowid=r['API']['api_rowid'],flag_inferred=False))
 for r in read(O/name)['bindings']:
  for a in r['roles']:role(r['database_index'],r['TASK']['task_record_rowid'],'v147:'+a,name,[r['database_index'],r['TASK']['task_record_rowid']])
 name='hc-norm-projection-native-frontiers-v143.json'
 for r in read(O/name)['records']:
  key=[r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']]
  for a,s in r['stages'].items():
   if s['TASK']is None:api.append(dict(source=name,key=key,role=a,API_rowid=s['API']['api_rowid'],flag_inferred=False));continue
   role(r['database_index'],s['TASK']['task_record_rowid'],'v143:'+a,name,key)
 out=dict(databases=dbs,definition_columns=dcols,definitions=defs,execution_columns=ecols,executions=execs,definition_rows=len(defs),joined_compute_TASK_rows=len(execs),definition_flag_counts=dict(dh),execution_flag_counts=dict(eh),unmatched_TASK_counts=dict(unmatched),op_dtype_flag_execution_counts=[dict(opType=o,inputDataTypes=d,label=f,executions=n)for(o,d,f),n in sorted(oh.items())],source_roles=roles,source_role_flag_counts=[dict(role=r,label=v,role_bindings=n)for(r,v),n in sorted(rhist.items())],API_only_roles_retained=api,field_is_exported_flag_not_instruction_inventory=True,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 write(O/'precision-field-population-v148.json',out);print(json.dumps({k:out[k]for k in ['definition_rows','joined_compute_TASK_rows','definition_flag_counts','execution_flag_counts','unmatched_TASK_counts']}));print(json.dumps(dict(databases=len(dbs),source_role_bindings=len(roles),source_role_TASK_union=len({(x['database_index'],x['TASK_rowid'])for x in roles}),API_only=len(api),nonunique_definition_groups=sum(len(x['nonunique_globalTaskId_definitions'])for x in dbs),missing_strings=sum(len(x['missing_STRING_ID_compute_rows'])for x in dbs),new_device_work=False,overall_complete=False)))


if __name__=='__main__':main()
