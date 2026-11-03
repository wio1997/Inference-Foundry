#!/usr/bin/env python3
"""Exact rank-local selected op + TASK/COMMUNICATION_TASK_INFO interval audit."""
import pathlib,sqlite3,hashlib,json,collections
spec=json.loads(SPEC_JSON)
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def union(xs):
 z=0;cur=None
 for a,b in sorted(xs):
  if cur is None:cur=(a,b)
  elif a<=cur[1]:cur=(cur[0],max(cur[1],b))
  else:z+=cur[1]-cur[0];cur=(a,b)
 return z+(cur[1]-cur[0] if cur else 0)
out=[]
for r in spec:
 ep=r['ep'];p=base/r['rank']/'ASCEND_PROFILER_OUTPUT'/('ascend_pytorch_profiler_%d.db'%ep)
 assert not pathlib.Path(str(p)+'-wal').exists() and sha(p)==r['db_sha']
 c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('pragma query_only=ON');c.execute('pragma automatic_index=OFF')
 def q(s,args=()):return [dict(x) for x in c.execute(s,args)]
 a=r['anchor']; anchors=q('SELECT * FROM TASK WHERE deviceId=? AND globalPid=? AND streamId=? AND taskId=? AND startNs=?',(ep,r['pid'],a['streamId'],a['taskId'],a['startNs']));assert len(anchors)==1 and anchors[0]['endNs']==a['endNs']
 opid=r['op']['opId'];ops=q('SELECT * FROM COMMUNICATION_OP WHERE opId=?',(opid,));assert len(ops)==1
 op=ops[0];assert op['connectionId']==r['op']['connectionId'] and op['startNs']==r['op']['startNs'] and op['endNs']==r['op']['endNs']
 cc=q('SELECT * FROM COMMUNICATION_OP WHERE connectionId=?',(op['connectionId'],)); assert sum(z['opId']==opid for z in cc)==1
 children=q('SELECT * FROM COMMUNICATION_TASK_INFO WHERE opId=? ORDER BY globalTaskId',(opid,));ids=[z['globalTaskId'] for z in children];unique_ids=sorted({z for z in ids if z is not None}); byid=collections.defaultdict(list)
 for i in range(0,len(unique_ids),400):
  batch=unique_ids[i:i+400];marks=','.join('?' for x in batch)
  for z in q('SELECT * FROM TASK WHERE globalTaskId IN ('+marks+')',batch):byid[z['globalTaskId']].append(z)
 refs=[]
 for ch in children:
  m=byid[ch['globalTaskId']];ok=len(m)==1;task=m[0] if ok else None
  refs.append({'child':ch,'task_matches':len(m),'task':task,'timestamp_matches':ok and task['startNs']==ch['timestampNs']})
 valid=[x['task'] for x in refs if x['task_matches']==1 and x['timestamp_matches']]
 overlaps=[x for x in valid if x['startNs']<a['endNs'] and x['endNs']>a['startNs'] and x['endNs']>x['startNs']]
 intervals=[(max(x['startNs'],a['startNs'])-a['startNs'],min(x['endNs'],a['endNs'])-a['startNs']) for x in overlaps]
 reused=q('SELECT * FROM CONNECTION_IDS WHERE connectionId=? ORDER BY id',(op['connectionId'],))
 extra=q('SELECT * FROM COMMUNICATION_TASK_INFO WHERE globalTaskId IN (?,?) ORDER BY opId,globalTaskId',(a['globalTaskId'],r['aicpu']['globalTaskId']))
 out.append({'ep':ep,'db_sha256':r['db_sha'],'anchor':a,'aicpu':r['aicpu'],'op':op,'opid_row_count':len(ops),'op_connection_comm_row_count':len(cc),'op_connection_id_links':reused,'op_start_from_anchor_end_ns':op['startNs']-a['endNs'],'op_overlap_ns':max(0,min(op['endNs'],a['endNs'])-max(op['startNs'],a['startNs'])),'children_count':len(children),'children_distinct_nonnull_globalTaskIds':len(unique_ids),'task_unique_match_count':sum(x['task_matches']==1 for x in refs),'timestamp_exact_match_count':sum(x['timestamp_matches'] for x in refs),'overlapping_child_count':len(overlaps),'overlap_union_ns':union(intervals),'earliest_child_start_after_anchor_end_ns':min((x['startNs']-a['endNs'] for x in valid),default=None),'matched_child_aicpu_connection_count':sum(x['connectionId']==r['aicpu']['connectionId'] for x in valid),'anchor_or_aicpu_global_id_child_refs':extra,'child_refs':refs})
 c.close();assert sha(p)==r['db_sha'] and not pathlib.Path(str(p)+'-wal').exists()
print(json.dumps({'scope':'per-rank same DB coordinates only; no cross-rank time differences, same transaction or causation claims','ranks':out},indent=2))
