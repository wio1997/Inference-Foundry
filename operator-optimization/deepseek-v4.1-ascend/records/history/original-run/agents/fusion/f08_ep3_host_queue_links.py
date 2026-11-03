#!/usr/bin/env python3
import sqlite3,pathlib,json
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
out={}
for name,lo,hi in [('short_L9',1793241328964248040,1793241328965935410),('long_L30',1793241329712702210,1793241329713134370)]:
 rows=[dict(r) for r in c.execute('''select a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,a.callchainId,s.value name from PYTORCH_API a join STRING_IDS s on s.id=a.name where cast(a.startNs as integer)<? and cast(a.endNs as integer)>? and (s.value like '%aclnnInplacePartialRotaryMul%' or s.value like '%aclnnScatterNdUpdateSk%') order by cast(a.startNs as integer)''',(hi+50000,lo-50000))]
 for r in rows:
  r['start_offset_ns']=int(r.pop('startNs'))-lo;r['end_offset_ns']=int(r.pop('endNs'))-lo
  cid=r['connectionId']
  r['connection_rows']=[dict(x) for x in c.execute('select * from CONNECTION_IDS where id=? or connectionId=?',(cid,cid))] if cid is not None else []
 out[name]=rows
print(json.dumps({'windows':out,'limits':'±50us host API rows by operator name; window overlap/name do not prove enqueue/dequeue pairing; missing rows not proof of absence outside window'},indent=2))
