#!/usr/bin/env python3
"""Read-only saved EP8 L14/L39 wait: inspect trace and SQLite identity fields."""
import hashlib,json,sqlite3
from pathlib import Path
from decimal import Decimal
import ijson
root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT')
trace=root/'trace_view.json'; db=root/'ascend_pytorch_profiler_8.db'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
tracehash=sha(trace)
if tracehash!='53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23': raise RuntimeError('trace hash mismatch')
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
targets=[dict(x) for x in c.execute('SELECT * FROM TASK WHERE globalPid=737 AND streamId=47 AND taskId IN (44205,45351) ORDER BY taskId')]
if len(targets)!=2:raise ValueError('target tasks not unique')
for t in targets:
 rows=[dict(x) for x in c.execute('SELECT a.startNs,a.endNs,a.type,a.globalTid,a.connectionId,s.value AS api_name FROM CANN_API a LEFT JOIN STRING_IDS s ON a.name=s.id WHERE a.connectionId=?',(t['connectionId'],))]
 t['cann_api']=rows
 if len(rows)!=1 or rows[0]['api_name']!='aclrtStreamWaitEvent':raise ValueError('API not unique or differs')
keys={(t['startNs'],t['streamId'],t['taskId']):t for t in targets};found=[]
with trace.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict) or e.get('ph')!='X' or e.get('name')!='EVENT WAIT':continue
  a=e.get('args') or {};ts=int(Decimal(str(e.get('ts')))*1000);key=(ts,a.get('Physic Stream Id'),a.get('Task Id'))
  if key in keys:
   found.append({'taskId':key[2],'top_keys':sorted(e),'arg_keys':sorted(a),'top_id_present':e.get('id') is not None,'top_id2_present':e.get('id2') is not None,'duration_us':str(e.get('dur'))})
if len(found)!=2 or {x['taskId'] for x in found}!={44205,45351}:raise ValueError('trace event not unique')
tables=[]
for (name,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
 cols=[x[1] for x in c.execute('PRAGMA table_info('+ '"'+name.replace('"','""')+'"'+')')]
 hits=[x for x in cols if any(w in x.lower() for w in ('event','notify','handle'))]
 if hits:tables.append({'table':name,'columns':hits})
print(json.dumps({'trace_sha256':tracehash,'database_sha256':sha(db),'scope':'two EP8 wait tasks preceding selected MC2; same database and trace','tasks':targets,'trace_fields':found,'schema_columns_with_event_notify_handle':tables,'limitations':['Connection and API names associate submission; they do not establish the event-object identity or record/wait pair.','Matching only exact start/stream/task; no inferred dependency from time proximity.','Schema inspection establishes names only, not all values or hidden external data.'],'no_new_device_work':True},indent=2))
