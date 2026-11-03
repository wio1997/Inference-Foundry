#!/usr/bin/env python3
"""EP0 fixed gating and MC2 anchors: read-only local TASK/API connection candidates."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root)
prior=T/'agents/diagnosis/draft-six-mc2-preceding-csv.json'
rank='dp0_pp0_tp0_dcp0_ep0_rank0_728_20261028020238584_ascend_pt'
db=R/rank/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
expected_prior='5de5efc398e82f21c28f1aabde8bd88141afb3343bf42d52289eb2126cdb7c42';expected_db='a6c626753b5e21cd066ede0cceffe104e8603c01a79918eb79a270dcb4ad0a09'
if sha(prior)!=expected_prior or sha(db)!=expected_db:raise RuntimeError('input SHA mismatch')
x=json.loads(prior.read_text());samples=[s for s in x['samples'] if s['rank']==rank]
if len(samples)!=3 or {s['local_op_id'] for s in samples}!={402,405,408}:raise RuntimeError('unexpected samples')
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
def bounded(sql,args,cap=40):
 rows=[dict(z) for z in c.execute(sql,args).fetchmany(cap+1)]
 if len(rows)>cap:raise RuntimeError('query cap exceeded')
 return rows
def safe(z):
 name=str(z.pop('name',None));z['name_sha256']=hashlib.sha256(name.encode()).hexdigest();z['name_class']=[v for v in ('launch','enqueue','dequeue','moe','gating','dispatch','combine','forward') if v in name.lower()];return z
out=[]
for s in samples:
 for role,anchors in [('gating',s['db_latest_end_rows']),('mc2',s['db_anchor_rows'])]:
  connections={z['connectionId'] for z in anchors}
  if len(connections)!=1:raise RuntimeError('anchor connection mismatch')
  conn=next(iter(connections)); checked=[]
  for z in anchors:
   rows=bounded('SELECT startNs,endNs,streamId,modelId,taskId,globalTaskId,connectionId FROM TASK WHERE globalTaskId=? LIMIT 41',(z['globalTaskId'],))
   exact=[r for r in rows if all(r[k]==z[k] for k in ('startNs','endNs','streamId','modelId','taskId','globalTaskId','connectionId'))]
   if len(exact)!=1:raise RuntimeError('exact TASK key not unique')
   checked.append({'anchor':exact[0],'globalTaskId_count':len(rows)})
  cann=[safe(z) for z in bounded('SELECT a.connectionId,a.startNs,a.endNs,a.globalTid,s.value AS name FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? LIMIT 41',(conn,))]
  sets=bounded('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId=? LIMIT 41',(conn,));pyt=[]
  for st in sets:
   rows=bounded('SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,a.sequenceNumber,a.callchainId,s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? LIMIT 41',(st['id'],))
   pyt.append({'connection_set_id':st['id'],'rows':[safe(z) for z in rows]})
  out.append({'opId_local':s['local_op_id'],'role':role,'task_matches':checked,'cann_same_connection':cann,'connection_sets':sets,'pytorch_via_sets':pyt})
c.close()
print(json.dumps({'scope':'EP0 stage189 gating and MC2 anchors: exact TASK fields and CANN/PYTORCH same connection candidates','input_sha256':{'prior':expected_prior,'db':expected_db},'targets':out,'limits':['Only local EP0 DB and fixed anchors, no cross-rank clocks or timestamp subtraction.','connection sets may be reused; matches are candidates not proof of unique parent, data dependence or causation.','Negative match does not prove missing submission or event; profiler coverage, graph and exporter behavior unknown.','No new NPU activity.']},ensure_ascii=False,indent=2))
