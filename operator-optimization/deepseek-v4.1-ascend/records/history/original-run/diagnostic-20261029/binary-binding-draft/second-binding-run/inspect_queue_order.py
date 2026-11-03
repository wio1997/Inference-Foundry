#!/usr/bin/env python3
"""Check observed queue order using complete-DB exact correlation, without FIFO assumption."""
import json,sqlite3
from pathlib import Path
from collections import defaultdict
root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run');job=root/'offline-ep8-20261029';status=json.loads((job/'status.json').read_text());item=next(x for x in status['dbs'] if x['path'].endswith('/ascend_pytorch_profiler_8.db'));db=job/item['path'];assert not Path(str(db)+'-wal').exists() and not Path(str(db)+'-journal').exists()
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);c.execute('PRAGMA query_only=ON');c.row_factory=sqlite3.Row
conn=defaultdict(set)
for x in c.execute('SELECT id,connectionId FROM CONNECTION_IDS'):conn[x['id']].add(x['connectionId'])
enq=defaultdict(list);deq=defaultdict(list)
for r in c.execute('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,s.value AS name FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name WHERE a.type=50002 AND (s.value LIKE 'Enqueue@%' OR s.value LIKE 'Dequeue@%')'''):
 name=r['name']; typ,base=name.split('@',1);entry={'startNs':int(r['startNs']),'endNs':int(r['endNs']),'globalTid':r['globalTid'],'connectionId':r['connectionId'],'name':name}
 for cor in conn.get(r['connectionId'],()): (enq if typ=='Enqueue' else deq)[(base,cor)].append(entry)
audit=json.loads((job/'ep8-layer14-queue-audit.json').read_text())
def union(xs):
 z=0;lo=hi=None
 for a,b in sorted(xs):
  if hi is None:lo,hi=a,b
  elif a<=hi:hi=max(hi,b)
  else:z+=hi-lo;lo,hi=a,b
 return z+(hi-lo if hi is not None else 0)
out=[]
for s in audit['results']:
 q=s['operations'][0]['queues'][0]; d=q['correlations'][0]['matching_dequeues'][0]['dequeue'];a=int(q['enqueue']['endNs']);b=int(d['startNs']);tid=d['globalTid'];classes=defaultdict(list);counts=defaultdict(int)
 for key,rs in deq.items():
  for x in rs:
   if x['globalTid']!=tid or x['endNs']<=a or x['startNs']>=b:continue
   v=(max(a,x['startNs']),min(b,x['endNs'])); matches=enq.get(key,[])
   if len(conn.get(x['connectionId'],()))!=1 or len(rs)!=1 or len(matches)!=1 or len(conn.get(matches[0]['connectionId'],()))!=1:cl='ambiguous_or_missing'
   elif matches[0]['endNs']<=a:cl='enqueue_completed_before_target'
   elif matches[0]['startNs']>=a:cl='enqueue_started_after_target'
   else:cl='enqueue_straddles_target'
   if len(matches)==1 and matches[0]['startNs']>x['startNs']:cl='invalid_recorded_order'
   counts[cl]+=1;classes[cl].append(v)
 out.append({'call_id':s['call_id'],'interval_ms':(b-a)/1e6,'counts':dict(counts),'class_union_ms':{k:union(v)/1e6 for k,v in classes.items()},'all_class_union_ms':union([x for values in classes.values() for x in values])/1e6})
print(json.dumps({'db_sha256':item['sha256'],'rules':'all observed same-name correlation enqueues and dequeues in full DB must each be unique and each row resolve to one correlation; target boundary is target enqueue end; interval times clipped','limits':['Profiler I/O error, dropped events, ID reuse outside captured DB cannot be excluded.','Enqueue start/end vs target does not establish FIFO, causality, critical path or that same physical queue serviced all operations.','An unrecorded enqueue/dequeue invalidates uniqueness assumptions.','The DB contains profiler overhead, cannot establish online scheduling latency.'],'results':out},indent=2))
