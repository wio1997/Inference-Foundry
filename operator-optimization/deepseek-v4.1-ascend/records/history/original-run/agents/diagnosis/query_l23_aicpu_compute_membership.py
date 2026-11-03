#!/usr/bin/env python3
"""Stage only: check selected exact TASK and COMPUTE_TASK_INFO membership."""
import pathlib,sqlite3,json,hashlib
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
out=[]
for sp in SPEC:
 p=base/sp['rank']/'ASCEND_PROFILER_OUTPUT'/('ascend_pytorch_profiler_%d.db'%sp['ep']);wal=pathlib.Path(str(p)+'-wal');assert not wal.exists() and sha(p)==sp['dbsha']
 c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');items=[]
 for x in sp['items']:
  rows=c.execute('SELECT * FROM TASK WHERE startNs=? AND globalTaskId=?',(x['start'],x['globaltask'])).fetchall();assert len(rows)==1 and rows[0]['deviceId']==sp['ep'] and rows[0]['endNs']==x['end']
  cs=c.execute('SELECT globalTaskId FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(x['globaltask'],)).fetchall();items.append({'digest':x['digest'],'task_exact_count':len(rows),'compute_info_matches':len(cs)})
 c.close();assert not wal.exists() and sha(p)==sp['dbsha'];out.append({'ep':sp['ep'],'items':items})
print(json.dumps({'scope':'16 selected AI_CPU TASK identities; same DB COMPUTE_TASK_INFO globalTaskId membership','ranks':out,'limits':['Absence in one metadata table does not mean absent from TASK or hardware execution','Not a clock conversion error bound or proof of exporter version','no device activity']},indent=2))
