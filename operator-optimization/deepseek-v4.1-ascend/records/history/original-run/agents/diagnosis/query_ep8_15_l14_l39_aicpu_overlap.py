#!/usr/bin/env python3
"""Only inside owned stage: per-rank frozen MC2 anchors, other-stream AI CPU records."""
import pathlib, sqlite3, hashlib, json, sys, csv, collections
from decimal import Decimal
spec=json.loads(SPEC_JSON)
root=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def union(rows):
 out=0; prev=None
 for a,b in sorted(rows):
  if prev is None:prev=(a,b)
  elif a<=prev[1]:prev=(prev[0],max(b,prev[1]))
  else:out+=prev[1]-prev[0];prev=(a,b)
 return out+(prev[1]-prev[0] if prev else 0)
out=[]
for rank in spec:
 ep=rank['ep']; d=root/rank['rank']/'ASCEND_PROFILER_OUTPUT'; db=d/('ascend_pytorch_profiler_%s.db'%ep); csvp=d/'kernel_details.csv'
 if pathlib.Path(str(db)+'-wal').exists():raise ValueError('WAL exists')
 pre=sha(db); cp=sha(csvp)
 if cp!=rank['csv_sha']:raise ValueError('csv hash mismatch ep %s'%ep)
 c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('pragma query_only=ON')
 samples=[]
 for target in rank['targets']:
  start=int(Decimal(target['start_us'])*1000);stream=int(target['stream']);tid=int(target['task'])
  found=[dict(x) for x in c.execute('SELECT t.*,s.value AS typeName FROM TASK t LEFT JOIN STRING_IDS s ON t.taskType=s.id WHERE t.deviceId=? AND t.globalPid=? AND t.streamId=? AND t.taskId=? AND t.startNs=?',(ep,rank['pid'],stream,tid,start))]
  if len(found)!=1:raise ValueError('anchor not unique ep/layer %s/%s found %s'%(ep,target['layer'],len(found)))
  a=found[0]; end=a['endNs']; delta=Decimal(end-start)-Decimal(target['duration_us'])*1000
  if abs(delta)>1 or a['typeName']!='KERNEL_MIX_AIC':raise ValueError('anchor differs ep/layer %s/%s'%(ep,target['layer']))
  rows=[dict(x) for x in c.execute('SELECT t.*,s.value AS typeName FROM TASK t LEFT JOIN STRING_IDS s ON t.taskType=s.id WHERE t.deviceId=? AND (t.streamId IS NULL OR t.streamId<>?) AND t.startNs<? AND t.endNs>? AND s.value=? ORDER BY t.streamId,t.startNs,t.endNs,t.taskId',(ep,stream,end,start,'KERNEL_AICPU'))]
  keys=('deviceId','globalPid','streamId','taskId','startNs','endNs','contextId','modelId','connectionId','taskType','globalTaskId'); unique={tuple(x[k] for k in keys):x for x in rows}; valid=[x for x in unique.values() if x['endNs']>x['startNs']]
  for x in valid:
   x['relative_start_ns']=max(start,x['startNs'])-start;x['relative_end_ns']=min(end,x['endNs'])-start
  samples.append({'layer':target['layer'],'anchor':a,'duration_difference_ns':str(delta),'raw_rows':len(rows),'duplicate_rows':len(rows)-len(unique),'nonpositive_rows':len(unique)-len(valid),'records':valid,'union_ns':union([(x['relative_start_ns'],x['relative_end_ns']) for x in valid])})
 c.close()
 desired={ (x['deviceId'],x['streamId'],x['taskId'],x['modelId'],x['startNs']) for s in samples for x in s['records'] }
 matched=collections.defaultdict(list)
 with csvp.open(newline='') as f:
  for row in csv.DictReader(f):
   try:key=(int(row['Device_id']),int(row['Stream ID']),int(row['Task ID']),int(row['Model ID']),int(Decimal(row['Start Time(us)'].strip())*1000))
   except (ValueError,ArithmeticError):continue
   if key in desired:matched[key].append({k:row[k].strip() for k in ('Name','Type','Accelerator Core','Duration(us)')})
 for s in samples:
  for x in s['records']:
   key=(x['deviceId'],x['streamId'],x['taskId'],x['modelId'],x['startNs']);m=matched[key]
   if len(m)!=1 or abs(Decimal(m[0]['Duration(us)'])*1000-(x['endNs']-x['startNs']))>1 or m[0]['Accelerator Core']!='AI_CPU':raise ValueError('CSV match failed ep/layer %s/%s'%(ep,s['layer']))
   x['csv']=m[0]
 post=sha(db)
 if pre!=post or pathlib.Path(str(db)+'-wal').exists():raise ValueError('db changed')
 out.append({'ep':ep,'db_sha256_first_observed':pre,'csv_sha256_verified':cp,'samples':samples})
print(json.dumps({'scope':'each DB own clock; exact prior trace/CSV anchor; other stream KERNEL_AICPU; no cross-rank time arithmetic, dependency or causal inference','ranks':out},indent=2))
