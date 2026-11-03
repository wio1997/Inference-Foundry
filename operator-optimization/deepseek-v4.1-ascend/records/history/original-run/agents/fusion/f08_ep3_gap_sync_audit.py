#!/usr/bin/env python3
"""Read existing EP3 profiler only; same-device track and DB stream windows."""
import json, sqlite3, pathlib, ijson
from decimal import Decimal as D
P=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT')
wins={'short_L9': (D('1793241328964353.955'),D('1793241328966065.515')), 'long_L30': (D('1793241329714111.935'),D('1793241329714147.715'))}
results={k:{'trace_track':[],'db_tasks':[]} for k in wins}
with (P/'trace_view.json').open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict) or str(e.get('pid'))!='2179523' or str(e.get('tid'))!='35' or 'ts' not in e: continue
  t=D(str(e['ts'])); end=t+D(str(e.get('dur',0)))
  for k,(a,b) in wins.items():
   if (end>=a and t<=b): results[k]['trace_track'].append({key:e.get(key) for key in ('name','ph','ts','dur','id','args') if key in e})
c=sqlite3.connect((P/'ascend_pytorch_profiler_3.db').as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
for k,(a,b) in wins.items():
 rows=c.execute('''select t.startNs,t.endNs,t.globalPid,t.deviceId,t.streamId,t.taskId,t.globalTaskId,t.taskType,t.connectionId,s.value as name from TASK t left join COMPUTE_TASK_INFO ci on ci.globalTaskId=t.globalTaskId left join STRING_IDS s on s.id=ci.name where t.globalPid=2128 and t.deviceId=3 and t.streamId=35 and t.endNs>=? and t.startNs<=? order by t.startNs''',(int(a*1000),int(b*1000)))
 results[k]['db_tasks']=[dict(r) for r in rows]
print(json.dumps({'windows_us':{k:list(map(str,v)) for k,v in wins.items()},'results':results,'limitations':'Existing trace, fixed device track and DB stream only; time interval is not evidence of idleness, waits, or causal dependency; exporter coverage unknown.'},default=str,indent=2))
