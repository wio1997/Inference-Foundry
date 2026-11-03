#!/usr/bin/env python3
import sqlite3,pathlib,json
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA automatic_index=OFF')
out={}
for id in [1008801,1008802,1008812,1008813,1018267,1018268,1018278,1018279]:
 rows=[dict(r) for r in c.execute('select pc.stackDepth,s.value frame from PYTORCH_CALLCHAINS pc left join STRING_IDS s on s.id=pc.stack where pc.id=? order by pc.stackDepth',(id,))]
 out[str(id)]={'total_frames':len(rows),'selected_frames':[r for r in rows if r['frame'] and any(x in r['frame'].lower() for x in ['dsa_v41','dsa_cp','deepseek_v4','torch_npu'])]}
print(json.dumps({'stacks':out,'limit':'callchain corresponds to time-enclosing producer-thread API; no stack attached to target dequeue or CANN API; containment alone is not definitive parent binding'},indent=2))
