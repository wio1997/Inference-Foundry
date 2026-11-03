#!/usr/bin/env python3
"""Read EP3/EP8 saved DBs; bounded selected indexer→attention task intervals."""
import pathlib,sqlite3,json,hashlib,collections
R=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
source=R/'agents/diagnosis/f08-attention-intervals.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
assert sha(source)=='355cd1a831403c1129ab76e128c0803da39b05712c3304c5c285aa82e6744e6c'
data=json.loads(source.read_text());results=[]; hashes={}; schemas={}
for ep in (3,8):
 paths=list((R/'diagnostic-20261029/runtime/diagnostic-traces').glob(f'*ep{ep}_rank*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{ep}.db'));assert len(paths)==1
 db=paths[0];hashes[str(ep)]=sha(db); c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.execute('pragma query_only=on');c.row_factory=sqlite3.Row
 for n in ('TASK','COMPUTE_TASK_INFO','CANN_API','PYTORCH_API','CONNECTION_IDS'):
  schemas.setdefault(n,[r[1] for r in c.execute('pragma table_info('+n+')')])
 def strs(v):
  r=c.execute('select value from STRING_IDS where id=?',(v,)).fetchone();return r[0] if r else None
 def task(start):
  rows=c.execute('select * from TASK where startNs=?',(start,)).fetchall(); assert len(rows)==1,(ep,start,len(rows));return dict(rows[0])
 def info(t):
  row=c.execute('select * from COMPUTE_TASK_INFO where globalTaskId=?',(t['globalTaskId'],)).fetchone()
  if row is None:return None
  r=dict(row);return {k:strs(r[k]) for k in ('name','opType','inputShapes','outputShapes','inputDataTypes','outputDataTypes')}
 for p in [z for z in data['indexer_attention_pairs'] if z['ep']==ep]:
  a=task(int(p['indexer_flow'])); b=task(int(p['attention_flow']));assert a['streamId']==b['streamId'] and a['endNs']<=b['startNs']
  mid=[dict(x) for x in c.execute('select * from TASK where streamId=? and startNs>=? and endNs<=? order by startNs,globalTaskId',(a['streamId'],a['endNs'],b['startNs']))]
  # Full exact same connection API set (names only); no address payloads are present in DB columns.
  def apis(t):
   rows=c.execute('select s.value from CANN_API a left join STRING_IDS s on s.id=a.name where a.connectionId=?',(t['connectionId'],)).fetchall();return [x[0] for x in rows]
  events=collections.Counter()
  for r in c.execute('select taskType,count(*) n from TASK where startNs>=? and startNs<? group by taskType',(a['endNs'],b['startNs'])):
   v=strs(r['taskType']);
   if v and ('EVENT' in v.upper() or 'NOTIFY' in v.upper()):events[v]+=r['n']
  results.append({'ep':ep,'layer':p['layer'],'scope_id':p['scope_id'],'indexer_task_id':a['taskId'],'attention_task_id':b['taskId'],'same_stream':a['streamId'],'duration_between_ns':b['startNs']-a['endNs'],'indexer':info(a),'attention':info(b),'indexer_cann_api_names':apis(a),'attention_cann_api_names':apis(b),'middle_tasks':[{'task_id':t['taskId'],'global_task_id':t['globalTaskId'],'task_type':strs(t['taskType']),'duration_ns':t['endNs']-t['startNs'],'info':info(t),'cann_api_names':apis(t)} for t in mid],'all_stream_event_type_counts':dict(events)})
 c.close()
print(json.dumps({'input_sha256':sha(source),'db_sha256':hashes,'tables_columns':schemas,'pairs':results,'limits':['EP3 and EP8 32 selected matched pairs only. Same-stream order is not tensor identity. Exact task connection API is not a tensor-identity relationship.','Intermediate set is fully contained inside target gap; tasks partially overlapping its edges are excluded. All-stream event task count covers startNs within the target gap, not event object matching.','No source-specific binding, no clock-domain subtraction, no NPU work.']},ensure_ascii=False,indent=2))
