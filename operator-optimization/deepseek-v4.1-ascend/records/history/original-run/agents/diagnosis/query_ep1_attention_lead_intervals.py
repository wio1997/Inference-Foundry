#!/usr/bin/env python3
"""Three frozen EP1 intervals from HcPre to alltoall, local device clock only."""
import argparse,csv,hashlib,json,sqlite3
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);args=ap.parse_args()
T=Path(args.task_root);R=Path(args.trace_root)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
prior=T/'agents/diagnosis/16rank-draft-alltoall-distribution.json'
if sha(prior)!='633011028e6d5953f03f1493c197085bdd1a810c095cc646e4f5b0e115f61ec6':raise RuntimeError('prior mismatch')
d=json.loads(prior.read_text());rank='dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt';base=R/rank/'ASCEND_PROFILER_OUTPUT';cp=base/'kernel_details.csv';db=base/'ascend_pytorch_profiler_1.db'
for p,k in [(cp,'csv'),(db,'db')]:
 if sha(p)!=d['file_sha256_by_rank'][rank][k]:raise RuntimeError('input mismatch '+p.name)
brackets=[]
for b in d['brackets']:
 if b['rank']!=rank or b['gap_ordinal']!=1:continue
 for h in b['alltoall']:
  if len(h['exact_communication_op_matches'])==1 and h['exact_communication_op_matches'][0]['opId'] in (402,405,408):
   brackets.append((b,h))
if sorted(h['exact_communication_op_matches'][0]['opId'] for b,h in brackets)!=[402,405,408]:raise RuntimeError('target selection changed')
def merged(segments,start,end):
 out=[]
 for a,b in sorted((max(start,a),min(end,b)) for a,b in segments if a<end and b>start and a<b):
  if not out or a>out[-1][1]:out.append([a,b])
  else:out[-1][1]=max(out[-1][1],b)
 return out
def length(segments):return sum(b-a for a,b in segments)
def gaps(ints,start,end):
 out=[];pos=start
 for a,b in ints:
  if a>pos:out.append((pos,a))
  pos=max(pos,b)
 if pos<end:out.append((pos,end))
 return out
csvrows=[]
with cp.open(newline='') as f:
 for row in csv.DictReader(f):
  a=int(Decimal(row['Start Time(us)'].strip())*1000);b=a+int(Decimal(row['Duration(us)'].strip())*1000)
  if any(a<h['alltoall_start_ns'] and b>z['bracket_start_ns'] for z,h in brackets):
   csvrows.append({'start':a,'end':b,'stream':row['Stream ID'],'model':row['Model ID'],'type':row['Type'],'input_shapes':row['Input Shapes']})
if len(csvrows)>5000:raise RuntimeError('csv bound exceeded')
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
dv=c.execute('SELECT DISTINCT deviceId FROM RANK_DEVICE_MAP').fetchall()
if len(dv)!=1 or dv[0][0]!=1:raise RuntimeError('unexpected rank/device')
out=[]
for bracket,h in sorted(brackets,key=lambda v:v[1]['exact_communication_op_matches'][0]['opId']):
 start=bracket['bracket_start_ns'];end=h['alltoall_start_ns'];main=int(bracket['main_stream']);opid=h['exact_communication_op_matches'][0]['opId']
 tasks=[dict(x) for x in c.execute('''SELECT t.startNs,t.endNs,t.streamId,t.modelId,t.taskId,t.globalTaskId,t.connectionId,s.value AS task_type_name
 FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType WHERE t.deviceId=? AND ((t.startNs<? AND t.endNs>?) OR (t.startNs>=? AND t.startNs<? AND t.endNs=t.startNs)) ORDER BY t.startNs,t.streamId LIMIT 5001''',(1,end,start,start,end))]
 if len(tasks)>5000:raise RuntimeError('task bound exceeded')
 cr=[x for x in csvrows if x['start']<end and x['end']>start]
 tm=[x for x in tasks if x['streamId']==main];to=[x for x in tasks if x['streamId']!=main]
 numeric=[x for x in cr if x['stream'].isdigit()];nonnum=[x for x in cr if not x['stream'].isdigit()]
 cm=[x for x in numeric if int(x['stream'])==main];co=[x for x in numeric if int(x['stream'])!=main]
 mt=merged([(x['startNs'],x['endNs']) for x in tm],start,end);ot=merged([(x['startNs'],x['endNs']) for x in to],start,end)
 mc=merged([(x['start'],x['end']) for x in cm],start,end);oc=merged([(x['start'],x['end']) for x in co],start,end);nc=merged([(x['start'],x['end']) for x in nonnum],start,end)
 allgaps=sorted(gaps(mt,start,end),key=lambda x:x[1]-x[0],reverse=True)
 def series(rows,sk,ek,kind):
  vals=[]
  for x in rows:
   z={'start_relative_us':(x[sk]-start)/1000,'duration_us':(x[ek]-x[sk])/1000}
   if kind=='task':z.update({k:x[k] for k in ['streamId','modelId','taskId','globalTaskId','connectionId','task_type_name']})
   else:z.update({k:x[k] for k in ['stream','model','type','input_shapes']})
   vals.append(z)
  return vals
 gapout=[]
 for ga,gb in allgaps[:12]:
  gapout.append({'start_relative_us':(ga-start)/1000,'duration_us':(gb-ga)/1000,'other_task_union_us':length(merged(ot,ga,gb))/1000,'other_numeric_csv_union_us':length(merged(oc,ga,gb))/1000,'nonnumeric_csv_union_us':length(merged(nc,ga,gb))/1000})
 stream_summary=[]
 for stream in sorted({x['streamId'] for x in tasks}):
  z=[x for x in tasks if x['streamId']==stream];stream_summary.append({'stream':stream,'count':len(z),'task_type_counts':dict(Counter(str(x['task_type_name']) for x in z)),'union_us':length(merged([(x['startNs'],x['endNs']) for x in z],start,end))/1000})
 out.append({'opId_local':opid,'duration_us':(end-start)/1000,'main_stream':main,'task_union_main_us':length(mt)/1000,'task_union_other_us':length(ot)/1000,'csv_union_main_us':length(mc)/1000,'csv_union_other_numeric_us':length(oc)/1000,'csv_union_nonnumeric_us':length(nc)/1000,'main_task_rows':series(tm,'startNs','endNs','task'),'other_task_rows':series(to,'startNs','endNs','task'),'csv_rows':series(cr,'start','end','csv'),'task_stream_summary':stream_summary,'largest_main_task_gaps':gapout,'main_gap_count':len(allgaps)})
c.close()
print(json.dumps({'scope':'EP1 only, 3 fixed [HcPre start, alltoall start) windows; all recorded TASK types, including zero-duration events; local device clock','inputs':{'prior_sha256':sha(prior),'csv_sha256':sha(cp),'db_sha256':sha(db)},'samples':out,'limits':['Intervals overlap only; no cross-rank clock comparisons or causal assertions.', 'Main gaps mean no exported positive-duration TASK on that stream, not device idle or absent CPU launches.', 'Other streams may belong to other requests/models, and TASK and CSV rows are not independent workloads.', 'N/A CSV rows may duplicate numeric rows; unions and coverage are not additive.', 'Zero-duration events are included in rows but do not contribute to interval union.', 'Source/profile capture may be incomplete.']},ensure_ascii=False,indent=2))
