#!/usr/bin/env python3
"""Inspect two EP8 fixed profiler DB task records; no tensor values, device or PMU."""
import argparse, hashlib, json, re, sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--db',required=True);a=p.parse_args()
db=Path(a.db).resolve(strict=True)
h=hashlib.sha256()
with db.open('rb') as f:
    for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
assert h.hexdigest()=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
targets=[('first',1282,1793152960208808431,1793152960208858990),('second',1283,1793152960208859010,1793152960208873789)]
def string(sid):
    row=c.execute('SELECT value FROM STRING_IDS WHERE id=?',(sid,)).fetchone()
    return row[0] if row else None
def clean_attr(raw):
    if raw is None:return {'present':False}
    raw=str(raw); names=[]; values={}
    allowed={'perm_x1','perm_x2','perm_y','transpose_x1','transpose_x2','tiling_key','workspace_size','input_strides','output_strides','stride_x1','stride_x2','stride_y'}
    for item in raw.split('|'):
        key,sep,value=item.partition(':'); key=key.strip()
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,80}',key):continue
        names.append(key)
        if key in allowed and re.fullmatch(r'[0-9,;()\[\] -]{1,120}',value.strip()):values[key]=value.strip()
    return {'present':True,'sha256':hashlib.sha256(raw.encode()).hexdigest(),'keys':sorted(set(names)),
            'allowlisted_numeric_values':values,'raw_content_not_exported':True}
rows=[]
for label,tid,start,end in targets:
    tasks=[dict(row) for row in c.execute('SELECT * FROM TASK WHERE modelId=? AND streamId=? AND taskId=? AND startNs=? AND endNs=?',(49,89,tid,start,end)).fetchmany(3)]
    entry={'label':label,'target':{'model':49,'stream':89,'task_id':tid,'start_ns':start,'end_ns':end},'exact_task_count_up_to_3':len(tasks),'tasks':[]}
    for t in tasks:
        infos=[dict(row) for row in c.execute('SELECT * FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchmany(3)]
        item={'task':t,'global_id_task_count':c.execute('SELECT COUNT(*) FROM TASK WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()[0],
              'compute_info_count_up_to_3':len(infos),'compute_info':[]}
        for i in infos:
            decoded={k:string(i.get(k)) for k in ('name','opType','inputFormats','outputFormats','inputDataTypes','outputDataTypes','inputShapes','outputShapes')}
            item['compute_info'].append({'numbers':i,'decoded':decoded,'attr':clean_attr(string(i.get('attrInfo')))})
        entry['tasks'].append(item)
    rows.append(entry)
interesting=[dict(row) for row in c.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND (lower(name) LIKE '%tiling%' OR lower(name) LIKE '%kernel%' OR lower(name) LIKE '%stride%' OR lower(name) LIKE '%binary%' OR lower(name) LIKE '%tensor%') ORDER BY name LIMIT 51")]
assert len(interesting)<=50
print(json.dumps({'scope':'two fixed EP8 model49 stream89 layer24 run2 task intervals, DB tables only',
 'db_sha256':h.hexdigest(),'rows':rows,'matching_schema_tables':interesting,
 'limits':['Decoded names are profiler labels and do not certify binary SHA, compiled source, runtime nodes, strides or tiling.',
 'Attribute names and numeric values are not assigned semantics without exporter and ABI contracts; raw attr contents not exported.',
 'Task ID and globalTaskId may be reused across graph replay. Exact timestamp binding is limited to the selected DB/clock.',
 'No payload/weight reading, NPU call, profiling, or raw PMU join.']},ensure_ascii=False,indent=2))
