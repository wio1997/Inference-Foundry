#!/usr/bin/env python3
"""Bounded lookup for same saved EP3 host interval and exact flow IDs."""
import sqlite3,pathlib,json,time
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;start=time.monotonic();c.set_progress_handler(lambda:int(time.monotonic()-start>25),100000)
tid=9139690408016;lo='1793241329101711120';hi='1793241329159378540'
q='SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,s.value name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTid=? AND p.startNs<=? AND p.endNs>=? ORDER BY p.startNs DESC LIMIT 80'
scopes=[dict(r) for r in c.execute(q,(tid,lo,hi))]
links=[dict(r) for r in c.execute('SELECT * FROM CONNECTION_IDS WHERE id IN (431371,431372,431373,431383,431396,431408)')]
conns=sorted({r['connectionId'] for r in links}); ops=[]
if conns:
 q='SELECT o.*,s.value name FROM COMMUNICATION_OP o LEFT JOIN STRING_IDS s ON o.opName=s.id WHERE o.connectionId IN ('+','.join('?' for _ in conns)+')'
 ops=[dict(r) for r in c.execute(q,conns)]
print(json.dumps({'scope':'single saved EP3 host thread; candidate 431371 and anchor 431408; exact 6 flow IDs','host_enclosing_both':scopes,'links':links,'communication_ops':ops,'limitations':'host nesting and connection mappings do not prove tensor identity, actual call path source file SHA or sampled layer unless decoder scope is independently identified'},indent=2))
