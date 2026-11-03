#!/usr/bin/env python3
"""Select only six existing EP8 layer-zero HC pre records; read-only SQLite."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--db',required=True);a=ap.parse_args()
t=Path(a.task_root);wp=t/'captured-layer-attribution/module-call-metrics.json'
w=[x for x in json.loads(wp.read_text()) if '_ep8_' in x['rank'] and x['run_in_model'] in (1,2,3) and x['layer']==0]
assert len(w)==3 and len({x['rank'] for x in w})==1
p=Path(a.db).resolve(strict=True);h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
sha=h.hexdigest();assert sha=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")};assert {'TASK','COMPUTE_TASK_INFO','STRING_IDS'}<=tables
names=('name','opType','inputFormats','inputDataTypes','inputShapes','outputFormats','outputDataTypes','outputShapes','attrInfo')
def textid(i):
 if i is None:return None
 r=c.execute('SELECT value FROM STRING_IDS WHERE id=?',(i,)).fetchone();return r[0] if r else None
rows=[]
for x in sorted(w,key=lambda q:q['run_in_model']):
 for phase,mi in (('attention_pre',0),('ffn_pre',2)):
  ts=list(c.execute('''SELECT * FROM TASK WHERE streamId=? AND modelId=? AND taskId=?
   AND CAST(startNs AS INTEGER)>=? AND CAST(endNs AS INTEGER)<=? ORDER BY CAST(startNs AS INTEGER) LIMIT 3''',
   (int(x['main_stream_id']),int(x['model_id']),int(x['marker_task_ids'][mi]),x['window_start_ns'],x['window_end_ns'])))
  rec={'run':x['run_in_model'],'layer':x['layer'],'phase':phase,'task_matches_within_window':len(ts)}
  if len(ts)==1:
   task=dict(ts[0]);gid=task['globalTaskId'];infos=list(c.execute('SELECT * FROM COMPUTE_TASK_INFO WHERE globalTaskId=? LIMIT 3',(gid,)))
   rec['task']={k:task[k] for k in ('startNs','endNs','streamId','modelId','taskId','globalTaskId','connectionId')}
   rec['compute_info_row_count_capped_at_3']=len(infos)
   rec['task_rows_with_same_gid']=c.execute('SELECT count(*) FROM TASK WHERE globalTaskId=?',(gid,)).fetchone()[0]
   if len(infos)==1:
    info=dict(infos[0]);rec['raw_info']=info;rec['decoded']={k:textid(info.get(k)) for k in names}
  rows.append(rec)
c.close()
print(json.dumps({'scope':'EP8 layer0, three interior target windows, two HC pre markers each',
 'window_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'db_sha256':sha,'samples':rows,
 'limitations':['Window marker role is source-order/invariant inference, not tensor identity.',
 'Repeated globalTaskId across TASK may be consistent with graph replay; reuse alone does not prove replay, unique binary or build.',
 'Decoded profiler attributes may omit tiling internals; absence is not evidence of nonexistence.',
 'No NPU execution or tensor values read.']},ensure_ascii=False,indent=2))
