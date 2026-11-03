#!/usr/bin/env python3
"""Six existing EP1 gating/MC2 anchors: bounded queue-mark candidates, read-only."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root); R=Path(a.trace_root)
p=T/'agents/diagnosis/draft-mc2-gap-host-links.json'
db=R/'dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_1.db'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(p)!='2b07456e853931f2764b494ccc5e315ad6ed9af5745cfa87f0b2da19940b157e' or sha(db)!='1d823cb13f49703ebd0a86ade38030ac8d6ef66b7085b23acf06f19ea42fa38b':raise RuntimeError('input mismatch')
x=json.loads(p.read_text()); c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
def bounded(sql,args,cap=100):
 rows=[dict(z) for z in c.execute(sql,args).fetchmany(cap+1)]
 if len(rows)>cap:raise RuntimeError('query cap exceeded')
 return rows
def safe(z):
 name=str(z.pop('name',None));z['name_sha256']=hashlib.sha256(name.encode()).hexdigest();z['name_class']=[k for k in ('enqueue','dequeue','queue','launch') if k in name.lower()];return z
results=[]
if len(x['targets'])!=6:raise RuntimeError('expected six targets')
for target in x['targets']:
 ps=[z for s in target['pytorch_via_sets'] for z in s['rows']]
 cs=target['cann_same_connection']
 if len(ps)!=1 or len(cs)!=1:raise RuntimeError('ambiguous anchor')
 p0,ca=ps[0],cs[0];start=int(p0['startNs']);end=int(p0['endNs'])
 anchor=bounded('SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? AND a.globalTid=? AND a.startNs=? AND a.endNs=? AND a.type=? LIMIT 101',(p0['connectionId'],p0['globalTid'],p0['startNs'],p0['endNs'],p0['type']))
 if len(anchor)!=1 or safe(anchor[0])['name_sha256']!=p0['name_sha256']:raise RuntimeError('pytorch anchor mismatch')
 cann=bounded('SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,s.value AS name FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? AND a.globalTid=? AND a.startNs=? AND a.endNs=? LIMIT 101',(ca['connectionId'],ca['globalTid'],ca['startNs'],ca['endNs']))
 if len(cann)!=1 or safe(cann[0])['name_sha256']!=ca['name_sha256']:raise RuntimeError('cann anchor mismatch')
 children=bounded('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.type=50002 AND a.globalTid=? AND a.startNs NOT GLOB '*[^0-9]*' AND a.endNs NOT GLOB '*[^0-9]*' AND CAST(a.startNs AS INTEGER)>=? AND CAST(a.endNs AS INTEGER)<=? LIMIT 101''',(p0['globalTid'],start,end))
 pairs=[]
 for child in children:
  if child['connectionId'] is None:continue
  corrs=bounded('SELECT connectionId FROM CONNECTION_IDS WHERE id=? LIMIT 101',(child['connectionId'],))
  for corr in corrs:
   rel=bounded('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value AS name FROM CONNECTION_IDS ids JOIN PYTORCH_API a ON a.connectionId=ids.id LEFT JOIN STRING_IDS s ON s.id=a.name WHERE ids.connectionId=? AND a.type=50002 AND a.globalTid=? AND a.startNs NOT GLOB '*[^0-9]*' AND a.endNs NOT GLOB '*[^0-9]*' AND CAST(a.startNs AS INTEGER)>=? AND CAST(a.endNs AS INTEGER)<=? LIMIT 101''',(corr['connectionId'],ca['globalTid'],start-2000000,end+2000000))
   for row in rel:pairs.append({'child_connection':child['connectionId'],'shared_correlation':corr['connectionId'],'worker':safe(row)})
 if len(pairs)>500:raise RuntimeError('pair cap')
 results.append({'opId_local':target['opId_local'],'role':target['role'],'anchor_pytorch':p0,'anchor_cann':ca,'contained_queue_children':[safe(z) for z in children],'related_queue_workers':pairs})
c.close()
print(json.dumps({'scope':'six stage190 EP1 anchors; same-PYTORCH-API nesting and shared correlation; worker bounded to anchor +/-2ms','inputs':{'stage190_sha256':sha(p),'db_sha256':sha(db)},'targets':results,'limits':['Containment and correlation are exported-record candidates, not proof of real dependency or complete capture.','Same-PYTORCH table endpoint arithmetic conditional on checked version/configuration and uncertainty in EP1-GAP-HOST-RECOVERY.','CANN/PyTorch containment follows parser matching logic as an association test only; cross-table clock accuracy and error unverified.','Search bounds and absent records prevent global claims; no NPU activity.']},ensure_ascii=False,indent=2))
