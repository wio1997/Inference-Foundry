#!/usr/bin/env python3
import sqlite3,pathlib,json
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA automatic_index=OFF')
out={}
for tag,start,end in [('L9_rotary',1793241327623297540,1793241327623300070),('L9_scatter',1793241327625191190,1793241327625194040),('L30_rotary',1793241328772693180,1793241328772696430),('L30_scatter',1793241328774955570,1793241328774960550)]:
 rows=[dict(r) for r in c.execute('''select a.startNs,a.endNs,a.globalTid,a.type,a.callchainId,s.value name from PYTORCH_API a left join STRING_IDS s on s.id=a.name where a.globalTid=? and cast(a.startNs as integer)<=? and cast(a.endNs as integer)>=?''',(9139690408016,start,end))]
 for r in rows:
  r['duration_ns']=int(r['endNs'])-int(r['startNs']);del r['startNs'];del r['endNs']
 rows.sort(key=lambda r:r['duration_ns'])
 out[tag]={'count':len(rows),'smallest_12':rows[:12]}
print(json.dumps({'results':out,'limits':'same-thread temporal containment; names do not prove parent or correct source binding; no request contents read'},indent=2))
