#!/usr/bin/env python3
import sqlite3,pathlib,json
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
out={}
for tag,stream,task in [('short_rotary',35,13225),('short_scatter',35,13226),('long_rotary',35,13467),('long_scatter',35,13468)]:
 taskrows=[dict(x) for x in c.execute('select * from TASK where globalPid=2128 and deviceId=3 and streamId=? and taskId=?',(stream,task))]
 rec={'task':taskrows,'cann_api':[],'pytorch_api':[]}
 for t in taskrows:
  cid=t['connectionId']
  for table in ('CANN_API','PYTORCH_API'):
   rows=c.execute(f'select a.startNs,a.endNs,a.globalTid,a.connectionId,s.value name from {table} a left join STRING_IDS s on a.name=s.id where a.connectionId=?',(cid,))
   rec[table.lower()].extend(dict(x) for x in rows)
 out[tag]=rec
print(json.dumps({'records':out,'limitations':'Exact connectionId matches only; absence of matching PYTORCH_API is not absence of a parent call. No subtraction between clocks is interpreted as delay.'},indent=2))
