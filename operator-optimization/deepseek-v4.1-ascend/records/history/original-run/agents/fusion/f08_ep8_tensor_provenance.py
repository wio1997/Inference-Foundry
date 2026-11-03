#!/usr/bin/env python3
"""Read-only fixed EP8 F08 task provenance. Emits only operator metadata, no tensors."""
import json, sqlite3, pathlib, hashlib, collections
from decimal import Decimal
import ijson
root=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT')
db=(root/'ascend_pytorch_profiler_8.db').resolve(strict=True); trace=(root/'trace_view.json').resolve(strict=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
con.execute('pragma query_only=ON');con.execute('pragma temp_store=MEMORY')
ids=['1793241329134062817','1793241329134098777','1793241329134135656']; tasks=[13274,13275,13276]
starts=['1793241329134062.817','1793241329134098.777','1793241329134135.656']
flows=collections.defaultdict(list)
with trace.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict):continue
  if str(e.get('id')) in ids and e.get('ph') in ('s','f'):
   flows[str(e['id'])].append({k:str(e[k]) for k in ('ph','pid','tid','ts','name','cat') if k in e})

rows=[]
for task,start,id in zip(tasks,starts,ids):
 ns=int(Decimal(start)*1000)
 endpoints=[x for x in flows[id] if x.get('ph')=='f']
 if len(endpoints)!=1 or Decimal(endpoints[0]['ts'])!=Decimal(start) or endpoints[0].get('tid')!='35':raise RuntimeError('flow endpoint not unique/exact')
 q='''select t.*,i.name as kernel_name,i.opType,i.inputFormats,i.inputDataTypes,i.inputShapes,i.outputFormats,i.outputDataTypes,i.outputShapes,i.attrInfo,i.opState from TASK t join COMPUTE_TASK_INFO i on t.globalTaskId=i.globalTaskId where t.streamId=? and t.taskId=? and abs(t.startNs-?)<10'''
 ts=[dict(x) for x in con.execute(q,(35,task,ns))]
 if len(ts)!=1:raise RuntimeError('expected unique task '+str((task,ns,len(ts))))
 t=ts[0]
 def resolve(v):
  r=con.execute('select value from STRING_IDS where id=?',(v,)).fetchone();return r[0] if r else None
 decoded={k:resolve(t[k]) for k in ('kernel_name','opType','inputFormats','inputDataTypes','inputShapes','outputFormats','outputDataTypes','outputShapes','attrInfo','opState')}
 cann=[dict(x) for x in con.execute('select startNs,endNs,globalTid,connectionId,name from CANN_API where connectionId=? limit 101',(t['connectionId'],))]
 if len(cann)>100:raise RuntimeError('CANN connection >100')
 for x in cann:x['name']=resolve(x['name'])
 # Exact connection correlation only; connection can be reused and does not prove argument identity.
 apis=[dict(x) for x in con.execute('select startNs,endNs,globalTid,connectionId,name,inputDtypes,inputShapes,callchainId,type from PYTORCH_API where connectionId=? limit 101',(t['connectionId'],))]
 if len(apis)>100:raise RuntimeError('PYTORCH connection >100')
 for x in apis:
  for k in ('name','inputDtypes','inputShapes'):x[k]=resolve(x[k])
 flowstarts=[x for x in flows[id] if x.get('ph')=='s' and x.get('pid')=='737' and x.get('tid')=='737']
 if len(flowstarts)!=1:raise RuntimeError('flow start not unique')
 hostns=int(Decimal(flowstarts[0]['ts'])*1000);tid=(737<<32)|737
 enclosing=[dict(x) for x in con.execute('select startNs,endNs,globalTid,connectionId,name,inputDtypes,inputShapes,callchainId,type from PYTORCH_API where globalTid=? and startNs<=? and endNs>=? order by startNs desc,endNs asc limit 101',(tid,hostns,hostns))]
 if len(enclosing)>100:raise RuntimeError('host nesting >100')
 for x in enclosing:
  for k in ('name','inputDtypes','inputShapes'):x[k]=resolve(x[k])
  chain=con.execute('select stack,stackDepth from PYTORCH_CALLCHAINS where id=?',(x['callchainId'],)).fetchone() if x['callchainId'] is not None else None
  if chain:x['callchain']={'stack':resolve(chain['stack']),'stackDepth':chain['stackDepth']}
 rows.append({'task':t,'decoded':decoded,'cann_same_connection':cann,'pytorch_same_connection':apis,'flow':flows[id],'pytorch_enclosing_flow_start':enclosing})
print(json.dumps({'scope':'EP8 L14 scope 737-20 selected stream35 task13274-13276 only','sha256':{'db':sha(db),'trace':sha(trace)},'task_rows':rows,'schema':{k:[x[1] for x in con.execute('pragma table_info('+k+')')] for k in ('TASK','COMPUTE_TASK_INFO','CANN_API','PYTORCH_API','PYTORCH_CALLCHAINS','CONNECTION_IDS')},'limitations':['Fields present/absent pertain to selected exported rows only.','IDs and shape do not prove storage identity; API same connection can be reused.','No NPU execution.']},ensure_ascii=False,indent=2))
