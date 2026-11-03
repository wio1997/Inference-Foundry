#!/usr/bin/env python3
"""Bounded read-only audit for a fixed EP8 DispatchFFNCombine task."""
import argparse, json, sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--db',required=True);a=ap.parse_args()
path=Path(a.db).resolve(strict=True)
c=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
gid=18529; connection=184330; start=1793152958875785664; end=1793152958877820884
tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
def limited(sql,args=(),maximum=100):
 rs=[dict(r) for r in c.execute(sql,args).fetchmany(maximum+1)]
 if len(rs)>maximum:raise RuntimeError('Too many rows; scope not assumed complete')
 return rs
tasks=limited('SELECT * FROM TASK WHERE globalTaskId=? ORDER BY CAST(startNs AS INTEGER)',(gid,))
target=[x for x in tasks if int(x['startNs'])==start and int(x['endNs'])==end and x['connectionId']==connection]
if len(target)!=1:raise RuntimeError('Target TASK row not unique')
pmu=limited('''SELECT p.timestampNs,s.value AS counter_name,p.value FROM TASK_PMU_INFO p
LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTaskId=?
AND s.value IN ('aic_total_cycles','aic_mac_ratio','aic_mte2_ratio')
ORDER BY CAST(p.timestampNs AS INTEGER),s.value''',(gid,))
pmu_times=limited('''SELECT timestampNs,COUNT(*) AS n,COUNT(DISTINCT name) AS names
FROM TASK_PMU_INFO WHERE globalTaskId=? GROUP BY timestampNs ORDER BY CAST(timestampNs AS INTEGER)''',(gid,))
comm={}
for table in ('COMMUNICATION_TASK_INFO','COMMUNICATION_SCHEDULE_TASK_INFO'):
 if table in tables: comm[table+'_globalTaskId_exact']=limited('SELECT * FROM '+table+' WHERE globalTaskId=?',(gid,))
if 'COMMUNICATION_OP' in tables:comm['COMMUNICATION_OP_connectionId_exact']=limited('SELECT * FROM COMMUNICATION_OP WHERE connectionId=?',(connection,))
print(json.dumps({'db':str(path),'target':target[0],'all_TASK_rows_sharing_globalTaskId':tasks,
'pmu_timestamps_and_counts':pmu_times,'selected_counter_rows':pmu,'exact_communication_links':comm,
'limits':['A shared ID is a candidate association until profiler schema defines its uniqueness and replay semantics.',
'Timestamp coincidence is diagnostic only unless clock and collection semantics are established.',
'No exact communication row does not disprove communication implemented inside the fused kernel.',
'Results cover one rank and one globalTaskId only; no NPU work.']},ensure_ascii=False,indent=2,default=str))
c.close()
