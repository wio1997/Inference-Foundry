#!/usr/bin/env python3
"""Read five exact pre-existing EP8 tail tasks; CPU only, no service interaction."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--db',required=True);a=ap.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
f=Path(a.task_root)/'agents/fusion/f26-tail-window-evidence.json';assert sha(f)=='2bb2dd756e0c201731b5caacd89a5b0cbb1c25f0ad47abf2222266bea9fe61f0'
x=json.loads(f.read_text());p=Path(a.db);assert sha(p)=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
items=[v for v in x['events'] if v['stream']=='89' and v['task_id'] in ('1994','1995','1996','1997','1998')];assert len(items)==5
out=[]
for e in items:
 s=x['anchor_end_ns']+e['start_rel_ns'];end=x['anchor_end_ns']+e['end_rel_ns']
 rows=c.execute('SELECT * FROM TASK WHERE modelId=? AND streamId=? AND taskId=? AND startNs=? AND endNs=?',(int(e['model']),int(e['stream']),int(e['task_id']),s,end)).fetchall();assert len(rows)==1,(e,len(rows))
 t=dict(rows[0]);gid=t['globalTaskId'];cid=t['connectionId']
 task_count=c.execute('SELECT count(*) FROM TASK WHERE connectionId=?',(cid,)).fetchone()[0]
 conn=c.execute('SELECT id FROM CONNECTION_IDS WHERE connectionId=?',(cid,)).fetchall()
 cann=c.execute('SELECT a.startNs,a.endNs,a.globalTid,s.value FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? LIMIT 51',(cid,)).fetchall()
 torch=[]
 for v in conn[:20]:
  tt=c.execute('SELECT a.startNs,a.endNs,a.globalTid,s.value FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? LIMIT 51',(v[0],)).fetchall();torch.append({'connection':v[0],'matches':[tuple(x) for x in tt[:50]],'truncated':len(tt)>50})
 out.append({'event':e,'task':t,'connection_task_count':task_count,'global_id_task_count':c.execute('SELECT count(*) FROM TASK WHERE globalTaskId=?',(gid,)).fetchone()[0],'cann':[tuple(r) for r in cann[:50]],'cann_truncated':len(cann)>50,'connection_count':len(conn),'pytorch':torch})
c.close()
print(json.dumps({'scope':'five specified EP8 model49 tail device tasks only','input_sha256':sha(f),'db_sha256':sha(p),'rows':out,'limitations':['Exact task identity only; connection ID reuse and graph replay prevent assuming direct producer identity.','Shapes and source/time adjacency do not establish storage identity or causality.','Bounded table queries, not proof no API or stack in other sources.']},ensure_ascii=False))
