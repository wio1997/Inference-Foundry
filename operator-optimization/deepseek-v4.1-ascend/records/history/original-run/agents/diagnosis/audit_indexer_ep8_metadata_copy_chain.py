#!/usr/bin/env python3
"""Bounded 6-call query: host copy scope→queue→device copy, plus trace keys only."""
import pathlib,sqlite3,json,hashlib,ijson
from decimal import Decimal
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt'); db=p/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db';trace=p/'ASCEND_PROFILER_OUTPUT/trace_view.json';info=p/'profiler_info_8.json'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert digest(db)=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
assert digest(info)=='a249b01a52dc6d9cbbd34953f2573b52d1d9f7280ff2d561b228b4a9417eb544'
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA automatic_index=OFF')
selected=[(1,1800,119274,93962),(1,1801,119275,93965),(2,13298,129219,111683),(2,13299,129220,111686),(3,14477,130133,113016),(3,14478,130134,113019)]
rows=[]
for run,gid,host_meta_id,enqueue_id in selected:
 prod=[dict(r) for r in c.execute('SELECT * FROM TASK WHERE globalTaskId=?',(gid,))];assert len(prod)==1
 meta=[dict(r) for r in c.execute('SELECT * FROM PYTORCH_API WHERE connectionId=?',(host_meta_id,))];assert len(meta)==1
 eq=[dict(r) for r in c.execute('SELECT * FROM PYTORCH_API WHERE connectionId=?',(enqueue_id,))];assert len(eq)==1
 meta=meta[0];eq=eq[0];assert meta['globalTid']==eq['globalTid'] and int(eq['startNs'])>=int(meta['endNs'])
 q='SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,p.inputShapes,p.inputDtypes,s.value name FROM PYTORCH_API p JOIN STRING_IDS s ON p.name=s.id WHERE p.globalTid=? AND CAST(p.startNs AS INTEGER)<=? AND CAST(p.endNs AS INTEGER)>=? ORDER BY CAST(p.startNs AS INTEGER) DESC'
 host=[dict(r) for r in c.execute(q,(eq['globalTid'],int(eq['startNs']),int(eq['endNs'])))];assert [x['name'] for x in host]==['Enqueue@aclnnInplaceCopy','aclnnInplaceCopy','aten::copy_']
 shared=[x[0] for x in c.execute('SELECT connectionId FROM CONNECTION_IDS WHERE id=?',(enqueue_id,))];assert len(shared)==1
 ids=[x[0] for x in c.execute('SELECT id FROM CONNECTION_IDS WHERE connectionId=?',(shared[0],))];dequeues=[]
 for id in ids:
  dequeues += [dict(r) for r in c.execute('SELECT p.*,s.value name FROM PYTORCH_API p JOIN STRING_IDS s ON p.name=s.id WHERE p.connectionId=? AND s.value=?',(id,'Dequeue@aclnnInplaceCopy'))]
 assert len(dequeues)==1;dq=dequeues[0]
 q='SELECT a.startNs,a.endNs,a.connectionId,s.value name FROM CANN_API a JOIN STRING_IDS s ON s.id=a.name WHERE a.globalTid=? AND a.startNs>=? AND a.endNs<=? AND s.value=?'
 api=[dict(r) for r in c.execute(q,(dq['globalTid'],int(dq['startNs']),int(dq['endNs']),'aclnnInplaceCopy'))];assert len(api)==1
 dev=[dict(r) for r in c.execute('SELECT t.*,s.value type FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType WHERE t.connectionId=?',(api[0]['connectionId'],))];assert len(dev)==1 and dev[0]['type']=='MEMCPY_ASYNC' and dev[0]['streamId']==prod[0]['streamId']
 rows.append({'run':run,'producer':prod[0],'host_metadata':{'connectionId':host_meta_id,'endNs':meta['endNs'],'globalTid':meta['globalTid']},'enqueue':{'connectionId':enqueue_id,'startNs':eq['startNs'],'globalTid':eq['globalTid']},'host_containing_scopes':host,'queue_association':shared[0],'dequeue':{'connectionId':dq['connectionId'],'globalTid':dq['globalTid'],'startNs':dq['startNs'],'endNs':dq['endNs']},'cann_copy_api':api[0],'device_copy':dev[0],'device_copy_start_minus_producer_end_ns':dev[0]['startNs']-prod[0]['endNs']})
wanted={r['device_copy']['startNs'] for r in rows};matches=[]
with trace.open('rb') as f:
 for e in ijson.items(f,'item'):
  if e.get('ph')!='X' or e.get('ts') is None:continue
  try:ns=int(Decimal(str(e['ts']))*1000)
  except Exception:continue
  if ns in wanted:
   args=e.get('args') or {};matches.append({'startNs':ns,'name':e.get('name'),'event_keys':sorted(e.keys()),'arg_keys':sorted(args.keys()) if isinstance(args,dict) else []})
assert len(matches)==6 and {x['startNs'] for x in matches}==wanted and all(x['name']=='MEMCPY_ASYNC' for x in matches)
config=json.loads(info.read_text())['config']['common_config']
print(json.dumps({'db_sha256':digest(db),'trace_sha256':digest(trace),'info_sha256':digest(info),'profile_common_whitelist':{k:config.get(k) for k in ['record_shapes','profile_memory','with_stack']},'matches':rows,'device_trace_keys':matches,'limits':'Six fixed candidates; host containment and queue link do not prove metadata-output identity. Same-stream device proximity does not prove copy reads producer output. Trace keys are six selected events only. No replay input binding.'},indent=2))
