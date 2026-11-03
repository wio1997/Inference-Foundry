#!/usr/bin/env python3
"""One frozen rank/task SQLite audit; writes JSON only to stdout, performs no device work."""
import argparse, json, sqlite3
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--db',required=True,help='one authorized profiler sqlite DB path')
args=ap.parse_args()
path=Path(args.db).resolve(strict=True)
c=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
c.row_factory=sqlite3.Row
tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
required={'TASK','COMPUTE_TASK_INFO','STRING_IDS'}
if not required <= tables:raise SystemExit('Missing required tables: '+str(sorted(required-tables)))
target=1793152958875785664
rows=c.execute('''SELECT t.globalTaskId,t.startNs,t.endNs,t.streamId,t.taskId,t.modelId,t.connectionId,
                         s.value AS compute_name,o.value AS op_type
FROM TASK t JOIN COMPUTE_TASK_INFO i ON i.globalTaskId=t.globalTaskId
LEFT JOIN STRING_IDS s ON s.id=i.name LEFT JOIN STRING_IDS o ON o.id=i.opType
WHERE t.streamId=? AND t.taskId=? AND t.modelId=? AND CAST(t.startNs AS INTEGER) BETWEEN ? AND ?''',
 (89,64,49,target-2000,target+2000)).fetchall()
if len(rows)!=1 or not any('DispatchFFNCombine' in str(rows[0][n]) for n in ('compute_name','op_type')):
 raise SystemExit('Exact sample did not match uniquely; stop without reinterpretation')
t=dict(rows[0]); gid=t['globalTaskId']; start=int(t['startNs']);end=int(t['endNs'])
assert end>start
def ident(s): return '"'+s.replace('"','""')+'"'
def cols(table):return [dict(r)['name'] for r in c.execute('PRAGMA table_info('+ident(table)+')')]
selected=sorted(t for t in tables if any(x in t.upper() for x in ('MSTX','TASK','EVENT','PMU','WAIT','COMMUNICATION')))
schemas={k:cols(k) for k in selected}
pmu=[]
if 'TASK_PMU_INFO' in tables and {'globalTaskId','name','value'} <= set(schemas['TASK_PMU_INFO']):
 pmu=[dict(x) for x in c.execute('''SELECT s.value AS counter_name,p.value FROM TASK_PMU_INFO p
 LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTaskId=? LIMIT 101''',(gid,))]
assert len(pmu)<=100, 'PMU list truncated; revise query before reporting'
mstx={}
for table in sorted(x for x in selected if 'MSTX' in x.upper()):
 cs=schemas[table]
 # Host/device clock comparability is unknown. Temporal overlap alone cannot establish causality.
 if {'startNs','endNs'}<=set(cs):
  q='SELECT * FROM '+ident(table)+' WHERE CAST(startNs AS INTEGER)<=? AND CAST(endNs AS INTEGER)>=? LIMIT 21'
  rr=[dict(x) for x in c.execute(q,(end,start))]
  mstx[table]={'overlap_count_capped_at_21':len(rr),'examples':rr[:5],'truncated':len(rr)>5}
 else:mstx[table]={'timestamp_columns_not_found':True}
print(json.dumps({'db':str(path),'target':t,'candidate_table_schemas':schemas,'exact_task_pmu':pmu,
 'mstx_temporal_overlap_only':mstx,'limits':['One fixed rank and invocation only.','MSTX overlap is temporal and does not prove MC2-internal origin, phase boundaries or dependency; host/device clock equivalence is unverified.','Kernel aggregate counters and CANN launch wait are not automatically internal communication or readiness wait.','No absence conclusion across the other 15 ranks or other profiler exports.']},indent=2,ensure_ascii=False,default=str))
c.close()
