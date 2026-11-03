#!/usr/bin/env python3
"""Read only batch_split_factor from one hash-pinned EP8 profiler DB task."""
import argparse,hashlib,json,re,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--db',required=True);a=p.parse_args()
db=Path(a.db).resolve(strict=True); h=hashlib.sha256()
with db.open('rb') as f:
    for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
assert h.hexdigest()=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True); c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
rows=c.execute('SELECT globalTaskId FROM TASK WHERE modelId=? AND streamId=? AND taskId=? AND startNs=? AND endNs=?',(49,89,1282,1793152960208808431,1793152960208858990)).fetchmany(3)
out={'scope':'one fixed EP8 layer24 run2 TASK and batch_split_factor only', 'db_sha256':h.hexdigest(), 'task_matches_up_to_3':len(rows),
     'compute_matches_up_to_3':None,'attr_hash_matches_stage236':None,'key_matches':None,'status':None,'value':None}
if len(rows)!=1:
    out['status']='TASK_ABSENT_OR_NONUNIQUE'
else:
    gid=rows[0]['globalTaskId']; assert gid==19471
    infos=c.execute('SELECT attrInfo FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(gid,)).fetchmany(3)
    out['compute_matches_up_to_3']=len(infos)
    if len(infos)!=1:
        out['status']='COMPUTE_ABSENT_OR_NONUNIQUE'
    else:
        decoded=c.execute('SELECT value FROM STRING_IDS WHERE id=?',(infos[0]['attrInfo'],)).fetchone()
        if decoded is None or decoded[0] is None:
            out['status']='ATTRIBUTE_STRING_ABSENT'
        else:
            raw=str(decoded[0]); attr_sha=hashlib.sha256(raw.encode()).hexdigest()
            out['attr_hash_matches_stage236']=(attr_sha=='319e40b4829a0df03cefc29dba88e2ed16da7e4e29c8de6a44416cad05d12eeb')
            if not out['attr_hash_matches_stage236']:
                out['status']='ATTRIBUTE_HASH_MISMATCH'
            else:
                matches=[]
                for segment in raw.split('|'):
                    key,sep,value=segment.partition(':')
                    if sep and key.strip()=='batch_split_factor':matches.append(value.strip())
                out['key_matches']=len(matches)
                if len(matches)!=1:
                    out['status']='KEY_ABSENT_OR_NONUNIQUE'
                elif not re.fullmatch(r'[+-]?[0-9]+',matches[0]):
                    out['status']='VALUE_NOT_PLAIN_INTEGER'
                else:
                    out['value']=int(matches[0]);out['status']='VALUE_AVAILABLE'
out['limits']=['Numeric profiler attribute is not a proof of source instance, tensor identity, internal physical copies, compiler binary, or runtime implementation.',
               'Unknown formatting is not guessed; raw attribute and other values are not exported.',
               'Exact key is specific to the selected TASK; globalTaskId alone is reused.', 'No NPU call or profiler execution.']
print(json.dumps(out,ensure_ascii=False,indent=2))
