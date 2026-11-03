#!/usr/bin/env python3
"""Observed profiler task coverage during selected host queue windows; read-only."""
import json,sqlite3
from pathlib import Path
root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run')
job=root/'offline-ep8-20261029';status=json.loads((job/'status.json').read_text());item=next(x for x in status['dbs'] if x['path'].endswith('/ascend_pytorch_profiler_8.db'));db=job/item['path'];assert not Path(str(db)+'-wal').exists() and not Path(str(db)+'-journal').exists()
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);c.execute('PRAGMA query_only=ON');c.row_factory=sqlite3.Row
r=json.loads((root/'inspect_queue_waits.json').read_text())
def union(xs):
 xs=sorted(xs);z=0;lo=hi=None
 for a,b in xs:
  if hi is None:lo,hi=a,b
  elif a<=hi:hi=max(hi,b)
  else:z+=hi-lo;lo,hi=a,b
 return z+(hi-lo if hi is not None else 0)
res=[]
for window in r['results']:
 a,b=window['interval_ns']; rows=c.execute('''SELECT startNs,endNs,globalTaskId,streamId,taskId,modelId FROM TASK WHERE deviceId=8 AND streamId=47 AND endNs>? AND startNs<?''',(a,b)).fetchall()
 intervals=[(max(a,int(x['startNs'])),min(b,int(x['endNs']))) for x in rows if int(x['endNs'])>int(x['startNs'])]
 copies=[]
 for x in window['copy_details']:
  lo,hi=x['startNs'],x['endNs']; ins=[(max(lo,v),min(hi,w)) for v,w in intervals if w>lo and v<hi];copies.append({'startNs':lo,'endNs':hi,'copy_host_duration_ms':(hi-lo)/1e6,'device_stream_observed_union_ms':union(ins)/1e6,'overlapping_task_rows':len(ins)})
 res.append({'call_id':window['call_id'],'interval_ns':[a,b],'interval_ms':(b-a)/1e6,'device':8,'stream':47,'overlapping_task_rows':len(intervals),'device_observed_union_ms':union(intervals)/1e6,'copy_windows':copies})
print(json.dumps({'db_sha256':item['sha256'],'scope':'Recorded TASK intervals on EP8 stream 47, clipped to queue windows and selected long host copy windows. TASK outer/inner duplicates merged by interval union; includes all task types/model IDs.','caveats':['Profiler DB timestamps treated as same normalized time domain; exact host-device flow corroborates selected launches but absolute alignment error not bounded.','SQLite I/O error and profiling overhead; missing task rows mean unknown, not device idle.','Stream 47 is not whole NPU. Device task interval includes possible waits, not compute activity. Overlap does not establish causal independence or speedup.'],'results':res},indent=2))
