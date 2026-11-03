#!/usr/bin/env python3
"""Read one hash-verified EP8 profiler SQLite task. No imports of device libraries."""
import argparse, hashlib, json, sqlite3
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--db', required=True); a=p.parse_args()
db=Path(a.db).resolve(strict=True)
h=hashlib.sha256()
with db.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
expected='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
assert h.hexdigest()==expected, 'DB hash mismatch'
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
c.row_factory=sqlite3.Row
c.execute('PRAGMA query_only=ON')
tasks=[dict(x) for x in c.execute('SELECT * FROM TASK WHERE modelId=? AND streamId=? AND taskId=? AND startNs=? AND endNs=?',(49,89,1275,1793152960208211062,1793152960208709553)).fetchmany(2)]
assert len(tasks)==1, 'task not unique'
t=tasks[0]; gid=t['globalTaskId']
infos=[dict(x) for x in c.execute('SELECT * FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(gid,)).fetchmany(2)]
assert len(infos)==1, 'compute info not unique'
i=infos[0]
def string(id):
    row=c.execute('SELECT value FROM STRING_IDS WHERE id=?',(id,)).fetchone()
    return row[0] if row else None
names={key:string(i.get(key)) for key in ('name','opType','inputFormats','outputFormats')}
metadata=[dict(x) for x in c.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND (lower(name) LIKE '%tiling%' OR lower(name) LIKE '%kernel%' OR lower(name) LIKE '%binary%') ORDER BY name LIMIT 51")]
assert len(metadata)<=50
out={'scope':'one EP8 layer24 run2 QLI historical task; read-only','db_sha256':expected,'task':t,'compute_task_info':i,'decoded_fields':names,'global_id_task_count':c.execute('SELECT COUNT(*) FROM TASK WHERE globalTaskId=?',(gid,)).fetchone()[0], 'potential_schema_tables':metadata,'limits':['Compute name identifies profiler string, not binary SHA, compiled variant, tiling or source provenance.','Schema inventory absence is not absence from other files.','No NPU execution or writes.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
