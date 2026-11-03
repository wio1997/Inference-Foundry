#!/usr/bin/env python3
"""Stage-only exact selected op child start coordinates, not causal events."""
import pathlib,sqlite3,json,hashlib
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
out=[]
for rank in SPEC:
 p=base/rank['rank']/'ASCEND_PROFILER_OUTPUT'/('ascend_pytorch_profiler_%d.db'%rank['ep']);wal=pathlib.Path(str(p)+'-wal')
 assert not wal.exists() and sha(p)==rank['dbsha']
 c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA automatic_index=OFF');items=[]
 for it in rank['items']:
  op=c.execute('SELECT startNs,endNs FROM COMMUNICATION_OP WHERE opId=? AND connectionId=?',(it['opid'],it['opconn'])).fetchall();assert len(op)==1 and tuple(op[0])==(it['opstart'],it['opend'])
  child=[dict(x) for x in c.execute('SELECT globalTaskId,timestampNs FROM COMMUNICATION_TASK_INFO WHERE opId=? LIMIT 2001',(it['opid'],))];assert len(child)<=2000
  ids=[x['globalTaskId'] for x in child];ts=[x['timestampNs'] for x in child if x['timestampNs'] is not None];assert len(ts)==len(ids) and len(set(ids))==len(ids)
  items.append({'digest':it['digest'],'opid':it['opid'],'child_count':len(child),'selected_aicpu_id_in_child_count':sum(x==it['taskid'] for x in ids),'child_start_before_aicpu_end_count':sum(t<it['aicpuend'] for t in ts),'child_start_inside_aicpu_count':sum(it['aicpustart']<=t<it['aicpuend'] for t in ts),'first_child_minus_aicpu_end_ns':min(ts)-it['aicpuend'] if ts else None,'first_child_minus_op_start_ns':min(ts)-it['opstart'] if ts else None})
 c.close();assert not wal.exists() and sha(p)==rank['dbsha'];out.append({'ep':rank['ep'],'db_sha':rank['dbsha'],'items':items})
print(json.dumps({'scope':'8 rank-local DBs, two selected opIds each, child start coordinates only','results':out,'limits':['cross-track clock accuracy and error unverified','start in interval does not prove overlap of duration, dependencies or physical execution','local opId and event associations do not prove same replay across ranks','no new device workload']},indent=2))
