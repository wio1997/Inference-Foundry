#!/usr/bin/env python3
import sqlite3,pathlib,json
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
ranges={'short_L9':(1793241328964248040,1793241328965935410),'long_L30':(1793241329712702210,1793241329713134370)}
out={}
for tag,(a,b) in ranges.items():
 rows=c.execute('''select s.value as name,count(*) as n,min(a.startNs) as first_ns,max(a.endNs) as last_ns from CANN_API a left join STRING_IDS s on a.name=s.id where a.globalTid=? and a.startNs>=? and a.endNs<=? group by a.name order by n desc,s.value''',(9139690412736,a,b))
 out[tag]={'gap_ns':b-a,'api_counts':[dict(r) for r in rows]}
print(json.dumps({'result':out,'limits':'CANN_API same globalTid wholly enclosed entries only; nested API calls may overlap, counts cannot be added as time and absent records do not prove idle.'},indent=2))
