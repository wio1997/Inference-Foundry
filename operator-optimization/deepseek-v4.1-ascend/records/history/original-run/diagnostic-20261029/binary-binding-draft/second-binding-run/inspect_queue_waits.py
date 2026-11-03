#!/usr/bin/env python3
"""Read-only profiler DB worker activity within two identified queue intervals."""
import json,sqlite3,hashlib
from collections import defaultdict
from pathlib import Path
root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/offline-ep8-20261029')
status=json.loads((root/'status.json').read_text()); item=next(x for x in status['dbs'] if x['path'].endswith('/ascend_pytorch_profiler_8.db')); db=root/item['path']
assert not Path(str(db)+'-wal').exists() and not Path(str(db)+'-journal').exists()
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);c.execute('PRAGMA query_only=ON');c.row_factory=sqlite3.Row
qdata=json.loads((root/'ep8-layer14-queue-audit.json').read_text())
def union(xs):
 xs=sorted(xs); z=0;lo=hi=None
 for a,b in xs:
  if hi is None:lo,hi=a,b
  elif a<=hi:hi=max(hi,b)
  else:z+=hi-lo;lo,hi=a,b
 return z+(hi-lo if hi is not None else 0)
results=[]
for sample in qdata['results']:
 q=sample['operations'][0]['queues'][0]['correlations'][0]; e=sample['operations'][0]['queues'][0]['enqueue']; d=q['matching_dequeues'][0]['dequeue'];a=int(e['endNs']);b=int(d['startNs']); tid=d['globalTid']; assert b>a
 rows=c.execute('''SELECT a.startNs,a.endNs,s.value AS name FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name WHERE a.type=50002 AND a.globalTid=? AND CAST(a.endNs AS INTEGER)>? AND CAST(a.startNs AS INTEGER)<? AND s.value LIKE 'Dequeue@%' ''',(tid,a,b)).fetchall()
 intervals=[];by=defaultdict(lambda:[0,0]);long=[]
 for row in rows:
  x=max(a,int(row['startNs']));y=min(b,int(row['endNs']));
  if y<=x:continue
  intervals.append((x,y));by[row['name']][0]+=1;by[row['name']][1]+=y-x;long.append((y-x,row['name'],x,y))
 cann=c.execute('''SELECT a.startNs,a.endNs,s.value AS name,a.type FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.globalTid=? AND CAST(a.endNs AS INTEGER)>? AND CAST(a.startNs AS INTEGER)<?''',(tid,a,b)).fetchall()
 cans=[]
 for row in cann:
  x=max(a,int(row['startNs']));y=min(b,int(row['endNs']));
  if y>x:cans.append((x,y))
 copy_details=[]
 for n,name,x,y in sorted(long,reverse=True):
  if name!='Dequeue@aclnnInplaceCopy': continue
  inside=c.execute('''SELECT a.startNs,a.endNs,s.value AS name,a.type,a.connectionId FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.globalTid=? AND CAST(a.startNs AS INTEGER)>=? AND CAST(a.endNs AS INTEGER)<=?''',(tid,x,y)).fetchall()
  calls=sorted([{'name':r['name'],'type':r['type'],'duration_ms':(int(r['endNs'])-int(r['startNs']))/1e6,'connectionId':r['connectionId']} for r in inside],key=lambda r:-r['duration_ms'])
  copy_details.append({'dequeue_ms':n/1e6,'startNs':x,'endNs':y,'cann_count':len(calls),'longest_cann':calls[:12]})
  if len(copy_details)==3:break
 results.append({'copy_details':copy_details,'call_id':sample['call_id'],'interval_ns':[a,b],'gap_ms':(b-a)/1e6,'worker_tid_global':tid,'dequeue_count':len(intervals),'dequeue_union_ms':union(intervals)/1e6,'dequeue_top_by_clipped_sum_ms':[{'name':name,'count':v[0],'clipped_sum_ms':v[1]/1e6} for name,v in sorted(by.items(),key=lambda t:-t[1][1])[:12]],'longest_clipped_dequeues':[{'name':name,'duration_ms':n/1e6,'startNs':x,'endNs':y} for n,name,x,y in sorted(long,reverse=True)[:10]],'cann_record_count':len(cans),'cann_union_ms':union(cans)/1e6})
res={'db_sha256':item['sha256'],'assumptions':['EP8 selective2 profiler, same DB clock; intervals enqueue end to matched dequeue start','dequeue and CANN clipped union is observed record coverage, not complete worker CPU utilization or proof of idle','profiling perturbation, missing records, SQLite I/O error and queue ID semantics apply'],'results':results}
print(json.dumps(res,ensure_ascii=False,indent=2))
