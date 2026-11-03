#!/usr/bin/env python3
"""Aggregate selected existing rank-8 profiler host event names; read-only, no device access."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--db',required=True);args=p.parse_args()
path=Path(args.db).resolve(strict=True);c=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY');c.row_factory=sqlite3.Row
required={'PYTORCH_API','STRING_IDS','TASK','COMPUTE_TASK_INFO'}
tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if not required<=tables:raise SystemExit('required tables missing')
patterns={'indexer_function':'%deepseek_v41/indexer.py%', 'dsa_function':'%attention/dsa_v41.py%', 'torch_copy':'aten::copy_', 'qli':'%lightning%indexer%'}
out={}
for key,pattern in patterns.items():
 rows=[dict(x) for x in c.execute('''SELECT s.value AS name, COUNT(*) AS n, MIN(CAST(p.startNs AS INTEGER)) AS first_ns, MAX(CAST(p.endNs AS INTEGER)) AS last_ns FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name WHERE s.value LIKE ? GROUP BY s.value ORDER BY n DESC LIMIT 101''',(pattern,))]
 if len(rows)>100:raise SystemExit('excess event names, query cannot faithfully report')
 out[key]=rows
ops=[dict(x) for x in c.execute('''SELECT op.value AS operator_type, COUNT(*) AS tasks FROM TASK t JOIN COMPUTE_TASK_INFO i ON i.globalTaskId=t.globalTaskId JOIN STRING_IDS op ON op.id=i.opType WHERE op.value LIKE '%lightning%indexer%' GROUP BY op.value ORDER BY tasks DESC LIMIT 101''')]
if len(ops)>100:raise SystemExit('excess device names')
print(json.dumps({'host_names':out,'all_device_tasks':ops,'limits':['One rank DB; global counts do not link requests, tensor identities or operators.','Graph replay and asynchronous launch require explicit correlations; presence or absence of host labels does not establish execution or absence of source branches.']},indent=2))
c.close()
