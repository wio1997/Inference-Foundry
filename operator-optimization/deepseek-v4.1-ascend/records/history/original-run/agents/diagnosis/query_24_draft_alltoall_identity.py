#!/usr/bin/env python3
"""Fixed 24 stage135 alltoalls: per-DB operation identities and child metadata."""
import argparse,hashlib,json,sqlite3
from collections import Counter
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);prior=T/'agents/diagnosis/16rank-draft-alltoall-distribution.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(prior)!='633011028e6d5953f03f1493c197085bdd1a810c095cc646e4f5b0e115f61ec6':raise RuntimeError('stage135 input hash mismatch')
d=json.loads(prior.read_text());samples=[]
if d['issues']:raise RuntimeError('prior contains issues')
for b in d['brackets']:
 for h in b['alltoall']:
  ops=h['exact_communication_op_matches']
  if b['gap_ordinal']==1 and b['hc_input_shape'].startswith('"1,4,5120;') and len(ops)==1 and ops[0]['count']==4096 and ops[0]['rankSize']==8:
   if len(h['exact_task_matches'])!=1:raise RuntimeError('unmatched target task')
   samples.append((b,h,ops[0]))
if len(samples)!=24 or len({b['rank'] for b,_,_ in samples})!=8:raise RuntimeError('24-target selection mismatch')
def digest(x):
 if x is None:return None
 return hashlib.sha256(str(x).encode()).hexdigest()
byrank={}
for b,h,o in samples:byrank.setdefault(b['rank'],[]).append((b,h,o))
out=[]
for rank,rs in sorted(byrank.items()):
 root=R/rank/'ASCEND_PROFILER_OUTPUT';dbs=list(root.glob('ascend_pytorch_profiler_*.db'))
 if len(dbs)!=1 or sha(dbs[0])!=d['file_sha256_by_rank'][rank]['db']:raise RuntimeError('database identity mismatch: '+rank)
 con=sqlite3.connect(dbs[0].resolve().as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row;con.execute('PRAGMA query_only=ON');con.execute('PRAGMA temp_store=MEMORY')
 def string(ident):
  if ident is None:return None
  q=con.execute('SELECT value FROM STRING_IDS WHERE id=?',(ident,)).fetchone()
  return q[0] if q else None
 def ename(table,ident):
  q=con.execute('SELECT name FROM '+table+' WHERE id=?',(ident,)).fetchone()
  return q[0] if q else None
 for b,h,op in rs:
  target=[dict(x) for x in con.execute('''SELECT * FROM COMMUNICATION_OP WHERE opId=? AND connectionId=? AND startNs=? AND endNs=? LIMIT 2''',
    (op['opId'],op['connectionId'],h['alltoall_start_ns'],h['alltoall_end_ns']))]
  if len(target)!=1:raise RuntimeError('operation not unique')
  v=target[0];children=[dict(x) for x in con.execute('SELECT * FROM COMMUNICATION_TASK_INFO WHERE opId=? ORDER BY timestampNs,globalTaskId LIMIT 2001',(op['opId'],))]
  if len(children)>2000:raise RuntimeError('too many local communication subtask rows')
  roles=Counter();child_rows=[]
  for x in children:
   roles[str(x['isMaster'])]+=1
   child_rows.append({'globalTaskId':x['globalTaskId'],'name_sha256':digest(string(x['name'])),
    'isMaster':x['isMaster'],'planeId':x['planeId'],'group_name_sha256':digest(string(x['groupName'])),
    'notify_id_sha256':digest(x['notifyId']),'srcRank':x['srcRank'],'dstRank':x['dstRank'],
    'transportType':x['transportType'],'linkType':x['linkType'],'rdmaType':x['rdmaType'],
    'dataType':x['dataType'],'size_field':x['size']})
  out.append({'rank':rank,'gap_ordinal':b['gap_ordinal'],'status':b['status'],'hc_input_shape':b['hc_input_shape'],
   'mc2_input_shape':b['mc2_input_shape'],'duration_ns':h['duration_ns'],'opId_local':v['opId'],
   'op_name_sha256':digest(string(v['opName'])),'group_name_sha256':digest(string(v['groupName'])),
   'algorithm_sha256':digest(string(v['algType'])),'opType_raw':v['opType'],
   'dataType_raw':v['dataType'],'dataType_enum':ename('ENUM_HCCL_DATA_TYPE',v['dataType']),
   'count_field':v['count'],'rankSize':v['rankSize'],'relay':v['relay'],'retry':v['retry'],
   'selected_task_global_id':h['exact_task_matches'][0]['globalTaskId'],
   'children_by_local_op_id':child_rows,'child_roles_counts':dict(roles)})
 con.close()
print(json.dumps({'scope':'Exactly 24 already selected stage135 per-rank records; no cross-rank timestamp joins',
 'source_sha256':sha(prior),'samples':out,
 'limitations':['opId is local to each database; equal values across ranks do not establish one collective invocation.',
 'Equal hashes indicate equal exported strings or identifiers only, not verified globally unique group or request identity.',
 'Children joined by local opId; mapping of notify, rank, size, transport and algorithm fields requires matching version documentation.',
 'No network bytes, causality, peer lateness or latency decomposition is inferred from these fields.',
 'No cross-rank absolute timestamps or request/token data are returned; no NPU workload.']},ensure_ascii=False,indent=2))
