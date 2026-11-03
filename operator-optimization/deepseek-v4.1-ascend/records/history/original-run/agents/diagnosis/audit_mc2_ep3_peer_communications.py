#!/usr/bin/env python3
"""Offline DB overlap and exact communication-op linkage, not dependency proof."""
import sqlite3,pathlib,json
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
samples={'L15':(2380,1793241329339782435,1793241329344803935),'L36':(3198,1793241329814910935,1793241329818413195)}
def union_ns(xs):
 xs=sorted(xs); out=0;end=None
 for a,b in xs:
  if b<=a: continue
  if end is None: start,end=a,b
  elif a<=end:end=max(b,end)
  else:out+=end-start;start,end=a,b
 return out+(end-start if end is not None else 0)
result={}
for label,(gid,lo,hi) in samples.items():
 target=[dict(r) for r in c.execute('SELECT * FROM TASK WHERE globalTaskId=? AND startNs=? AND endNs=?',(gid,lo,hi))];assert len(target)==1
 q='''SELECT t.startNs,t.endNs,t.globalTaskId,t.streamId,t.taskId,t.taskType,t.connectionId,ct.name label,ct.opId,ct.srcRank,ct.dstRank,ct.notifyId FROM TASK t LEFT JOIN COMMUNICATION_TASK_INFO ct ON ct.globalTaskId=t.globalTaskId WHERE t.deviceId=3 AND t.streamId<>47 AND t.startNs<? AND t.endNs>? ORDER BY t.startNs'''
 other=[dict(r) for r in c.execute(q,(hi,lo))]
 for row in other:
  if row['label'] is not None:
   v=c.execute('SELECT value FROM STRING_IDS WHERE id=?',(row['label'],)).fetchone();row['label']=v[0] if v else None
 comm=[dict(r) for r in c.execute('SELECT * FROM COMMUNICATION_OP WHERE deviceId=3 AND startNs<? AND endNs>?',(hi,lo))]
 ids=sorted({x['opId'] for x in other if x['opId'] is not None})
 linked=[]
 for opid in ids:
  rs=[dict(r) for r in c.execute('SELECT o.opId,o.opName,o.startNs,o.endNs,o.connectionId,o.rankSize FROM COMMUNICATION_OP o WHERE o.opId=?',(opid,))]
  for row in rs:
   v=c.execute('SELECT value FROM STRING_IDS WHERE id=?',(row['opName'],)).fetchone();row['name']=v[0] if v else None;row['start_minus_target_end_ns']=row['startNs']-hi
  linked.extend(rs)
 waits=[r for r in other if r['label']=='Notify_Wait']
 result[label]={'duration_ns':hi-lo,'other_track_task_count':len(other),'notify_wait_rows':waits,'notify_wait_clipped_union_ns':union_ns([(max(lo,r['startNs']),min(hi,r['endNs'])) for r in waits]),'overlapping_communication_op_count':len(comm),'linked_ops':linked,'other_track_rows':other}
print(json.dumps({'samples':result,'limits':'Two selected tasks from same profiled EP3 capture. Overlap is exported-clock interval association only. Same opId joins DB records; no proof of relation to MC2, complete capture, communication internals, clock calibration, physical idle, waiting cause, or attainable gain.'},indent=2))
