#!/usr/bin/env python3
"""Inside owned stage only: enumerate recorded TASK overlap in two frozen MC2 windows."""
import sqlite3,pathlib,json,hashlib,collections,csv
from decimal import Decimal
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
assert h.hexdigest()=='e07ee2428be27c6129afd6dd0be8b700830048f30f13411b701e97c5d88e2486'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
def union(it):
 xs=sorted(it); s=e=None; total=0
 for a,b in xs:
  if s is None:s,e=a,b
  elif a<=e:e=max(e,b)
  else:total+=e-s;s,e=a,b
 return total+(e-s if s is not None else 0)
targets=[(14,44217,1793241329140513188,'160365.95575'),(39,45363,1793241329850316287,'8290.41425')]
csvp=p.parent/'kernel_details.csv'
ch=hashlib.sha256()
with csvp.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):ch.update(b)
assert ch.hexdigest()=='20cfb97abe03befda3fbf2c1194e48531aac44ce17d4f5fc130ea8ffb5261730'
with csvp.open(newline='') as f: csv_rows=list(csv.DictReader(f))
results=[]
for layer,tid,start,tracedur in targets:
 target=[dict(x) for x in c.execute('SELECT t.*,s.value AS taskTypeName FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType WHERE t.deviceId=8 AND t.globalPid=737 AND t.streamId=47 AND t.taskId=? AND t.startNs=?',(tid,start))]
 assert len(target)==1
 t=target[0];end=t['endNs']; assert end>start
 rows=[dict(x) for x in c.execute('SELECT t.*,s.value AS taskTypeName FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType WHERE t.deviceId=8 AND (t.streamId IS NULL OR t.streamId<>47) AND t.startNs<? AND t.endNs>? ORDER BY t.streamId,t.startNs,t.endNs,t.taskId',(end,start))]
 identity=[tuple(r[k] for k in ('deviceId','globalPid','streamId','taskId','startNs','endNs','contextId','modelId','connectionId','taskType','globalTaskId')) for r in rows]
 count=collections.Counter(identity);duplicates=sum(v-1 for v in count.values()); unique=list({k:r for k,r in zip(identity,rows)}.values())
 groups=collections.defaultdict(list)
 nonpositive=[r for r in unique if r['endNs']<=r['startNs']]
 unique=[r for r in unique if r['endNs']>r['startNs']]
 for r in unique:
  a=max(start,r['startNs']);b=min(end,r['endNs']);assert b>a
  r['clipped_start_ns']=a-start;r['clipped_end_ns']=b-start;r['clipped_duration_ns']=b-a
  if r['taskTypeName']=='KERNEL_AICPU':
   matches=[]
   for cr in csv_rows:
    try:
     if (int(cr['Device_id'])==r['deviceId'] and int(cr['Stream ID'])==r['streamId'] and int(cr['Task ID'])==r['taskId'] and int(cr['Model ID'])==r['modelId'] and Decimal(cr['Start Time(us)'].strip())*1000==r['startNs']):
      matches.append({k:cr[k].strip() for k in ['Name','Type','Accelerator Core','Start Time(us)','Duration(us)']})
    except (ValueError,ArithmeticError): pass
   assert len(matches)==1
   assert abs(Decimal(matches[0]['Duration(us)'])*1000-(r['endNs']-r['startNs']))<=Decimal(1)
   r['exact_csv_matches']=matches
   r['same_connection_TASK']=[dict(x) for x in c.execute('SELECT t.startNs,t.endNs,t.streamId,t.taskId,t.globalTaskId,s.value AS taskTypeName FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType WHERE t.deviceId=? AND t.globalPid=? AND t.connectionId=? ORDER BY t.startNs',(r['deviceId'],r['globalPid'],r['connectionId']))]
   r['same_connection_CANN_API']=[dict(x) for x in c.execute('SELECT a.startNs,a.endNs,a.globalTid,s.value AS apiName FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=?',(r['connectionId'],))]
  groups[str(r['taskTypeName'])].append(r)
 g=[]
 for name,rr in sorted(groups.items()):
  ordered=sorted(rr,key=lambda r:r['clipped_duration_ns'],reverse=True)
  streams=collections.Counter(str(r['streamId']) for r in rr)
  g.append(dict(task_type=name,rows=len(rr),union_ns=union((r['clipped_start_ns'],r['clipped_end_ns']) for r in rr),streams=dict(streams),top_10=ordered[:10]))
 results.append(dict(layer=layer,anchor=t,trace_duration_ns=str(Decimal(tracedur)*1000),db_minus_trace_duration_ns=str(Decimal(end-start)-Decimal(tracedur)*1000),other_stream_rows_raw=len(rows),full_key_duplicate_rows=duplicates,nonpositive_duration_rows=len(nonpositive),nonpositive_types=dict(collections.Counter(str(r['taskTypeName']) for r in nonpositive)),other_stream_rows_distinct=len(unique),other_stream_union_ns=union((r['clipped_start_ns'],r['clipped_end_ns']) for r in unique),groups=g))
print(json.dumps(dict(db_sha256=h.hexdigest(),csv_sha256=ch.hexdigest(),clock_scope='same DB device 8 TASK coordinates; anchor start is exact prior trace match, task durations differ by 0.25ns',limits='No inference of device idleness, dependencies, causes, or complete capture. Type groups may overlap and include parent/child task records; unions use timestamps not duration sums.',samples=results),indent=2))
