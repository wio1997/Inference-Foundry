#!/usr/bin/env python3
"""Bounded read-only name/shape inventory in one previously recorded profiler DB."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--db',required=True);args=p.parse_args()
f=Path(args.db).resolve(strict=True);c=sqlite3.connect(f.as_uri()+'?mode=ro',uri=True)
c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY');c.row_factory=sqlite3.Row
q='''SELECT s.value AS name,op.value AS op_type,sh.value AS input_shape,COUNT(*) AS task_records,COUNT(DISTINCT t.globalTaskId) AS distinct_global_task_ids, MIN(t.startNs) AS first_ns, MAX(t.endNs) AS last_ns
FROM TASK t JOIN COMPUTE_TASK_INFO i ON i.globalTaskId=t.globalTaskId
LEFT JOIN STRING_IDS s ON s.id=i.name LEFT JOIN STRING_IDS op ON op.id=i.opType LEFT JOIN STRING_IDS sh ON sh.id=i.inputShapes
WHERE lower(coalesce(s.value,'')) LIKE ? OR lower(coalesce(op.value,'')) LIKE ?
GROUP BY s.value,op.value,sh.value ORDER BY task_records DESC LIMIT 201'''
h='''SELECT s.value AS name, COUNT(*) AS records FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name WHERE lower(s.value) LIKE ? GROUP BY s.value ORDER BY records DESC LIMIT 201'''
duplicates=c.execute('SELECT COUNT(*) FROM (SELECT globalTaskId FROM COMPUTE_TASK_INFO GROUP BY globalTaskId HAVING COUNT(*) > 1)').fetchone()[0]
task_reuse=c.execute('SELECT COUNT(*) FROM (SELECT globalTaskId FROM TASK GROUP BY globalTaskId HAVING COUNT(*) > 1)').fetchone()[0]
patterns=['%pool_kernel%','%cache_update_kernel%','%compressor%','%engram%','%rmsnorm%']
out={}
for pat in patterns:
 rows=[dict(r) for r in c.execute(q,(pat,pat))];hosts=[dict(r) for r in c.execute(h,(pat,))]
 if len(rows)>200 or len(hosts)>200:raise SystemExit('group limit exceeded')
 out[pat]={'device_task_groups':rows,'host_event_groups':hosts}
print(json.dumps({'rank_db':str(f),'patterns':out,'join_cardinality_audit':{'compute_info_global_id_duplicate_groups':duplicates,'task_global_id_reused_groups':task_reuse,'meaning':'A unique metadata row does not prove event-level shape or replay identity; task_records are DB row counts, not validated unique invocations.'},'limitations':['Aggregation describes this rank DB only; absence of a name does not prove a kernel did not execute.','Task globalTaskId may be reused in replay. Even if joined metadata is unique per ID, its name/shape are not validated per execution; row count is not a validated invocation count.','Graph capture and replay may record host and device counts differently.']},indent=2))
c.close()
