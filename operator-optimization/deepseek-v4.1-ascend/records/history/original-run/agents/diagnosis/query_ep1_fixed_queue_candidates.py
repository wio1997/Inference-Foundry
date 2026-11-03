#!/usr/bin/env python3
"""Three anchored PyTorch rows: same-table candidate queue/correlation records."""
import argparse,hashlib,json,re,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);prior=T/'agents/diagnosis/ep1-pytorch-clock-provenance.json';rank='dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt';db=R/rank/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_1.db'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(prior)!='c9f9ac9173e73dd67cb5a3b6cc8446690d67e1e22bb55f772e4491654454085e' or sha(db)!='1d823cb13f49703ebd0a86ade38030ac8d6ef66b7085b23acf06f19ea42fa38b':raise RuntimeError('input mismatch')
x=json.loads(prior.read_text());c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
def bounded(sql,args,cap=200):
 rows=[dict(z) for z in c.execute(sql,args).fetchmany(cap+1)]
 if len(rows)>cap:raise RuntimeError('query cap exceeded')
 return rows
def sanitize(row):
 name=str(row.pop('name',None));lower=name.lower();row['name_sha256']=hashlib.sha256(name.encode()).hexdigest();row['name_class']=[k for k in ('enqueue','dequeue','queue','launch','matmul') if k in lower];return row
enums=bounded('SELECT * FROM ENUM_API_TYPE LIMIT 51',(),50)
if any(any(isinstance(v,str) and len(v)>100 for v in row.values()) for row in enums):raise RuntimeError('unexpected enum value')
samples=[]
for sample in x['samples']:
 p=sample['pytorch'][0];ca=sample['cann'][0];start=p['startNs'];end=p['endNs']
 if not re.fullmatch(r'[0-9]{1,19}',start) or not re.fullmatch(r'[0-9]{1,19}',end) or int(end)<int(start):raise RuntimeError('invalid anchor time')
 # Time comparisons are limited to PYTORCH_API against another PYTORCH_API row.
 children=bounded('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON a.name=s.id
 WHERE a.globalTid=? AND a.startNs NOT GLOB '*[^0-9]*' AND a.endNs NOT GLOB '*[^0-9]*' AND CAST(a.startNs AS INTEGER)>=? AND CAST(a.endNs AS INTEGER)<=? ORDER BY CAST(a.startNs AS INTEGER) LIMIT 201''',(p['globalTid'],int(start),int(end)))
 links=[]
 for child in children:
  if child['connectionId'] is None:continue
  rawids=bounded('SELECT connectionId FROM CONNECTION_IDS WHERE id=? LIMIT 101',(child['connectionId'],),100)
  for corr in rawids:
   related=bounded('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value AS name FROM CONNECTION_IDS ids JOIN PYTORCH_API a ON a.connectionId=ids.id LEFT JOIN STRING_IDS s ON s.id=a.name WHERE ids.connectionId=? AND a.globalTid=? LIMIT 101''',(corr['connectionId'],ca['globalTid']),100)
   for z in related:links.append({'child_connection_set_id':child['connectionId'],'shared_local_correlation_id':corr['connectionId'],'worker_candidate':sanitize(z)})
 if len(links)>500:raise RuntimeError('link cap')
 samples.append({'opId_local':sample['opId_local'],'pytorch_anchor_connection':p['connectionId'],'anchor_startNs':start,'anchor_endNs':end,'anchor_globalTid':p['globalTid'],'cann_globalTid':ca['globalTid'],'inside_anchor_same_globalTid': [sanitize(z) for z in children], 'same_correlation_other_globalTid_candidates':links})
c.close()
root=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')
fixed=[root/'prof_common_func/_id_manager.py',root/'prof_common_func/_constant.py']
fixed+=list((root/'prof_view/prof_db_parse').glob('*pre*parser.py'))
fixed+=list((root/'prof_bean').glob('*queue*.py'))
pat=re.compile(r'global.?tid|enqueue|dequeue|correlation|connection|timestamp|get.*time|time.?offset|syscnt|class ApiType|TASK_QUEUE',re.I)
sources=[]
if len(fixed)>15:raise RuntimeError('source file bound')
for path in sorted(set(fixed)):
 if not path.exists():sources.append({'path':str(path),'missing':True});continue
 raw=path.read_bytes()
 if len(raw)>1048576:raise RuntimeError('oversized source')
 hits=[{'line':i,'text':s.strip()[:300]} for i,s in enumerate(raw.decode().splitlines(),1) if pat.search(s)]
 sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'hits':hits[:120],'truncated':len(hits)>120})
print(json.dumps({'scope':'Three associated PYTORCH_API anchors: rows nested on same globalTid, candidate shared connection-set IDs on CANN-bound globalTid; fixed source files','inputs':{'prior_sha256':sha(prior),'db_sha256':sha(db)},'api_type_enum':enums,'samples':samples,'source_files':sources,'limits':['Only same-PYTORCH_API time containment used; no host-device or cross-table timestamp arithmetic.','Name class by substring and same connection-set values are candidates; queue correlation contract and provenance must be independently verified.','Queue marker may be absent, omitted, unmatched, or outside searched parent; no match does not prove queue absent.','Thread identity encoding and cross-table time transform are unverified.','No prompt, input shapes, raw arbitrary names or new NPU work.']},ensure_ascii=False,indent=2))
