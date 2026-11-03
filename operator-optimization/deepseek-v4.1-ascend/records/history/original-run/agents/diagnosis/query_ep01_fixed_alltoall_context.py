#!/usr/bin/env python3
"""Six fixed local all-to-all candidates: matrix and local task context only."""
import argparse,csv,hashlib,json,sqlite3
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
files=[('agents/diagnosis/16rank-draft-alltoall-distribution.json','633011028e6d5953f03f1493c197085bdd1a810c095cc646e4f5b0e115f61ec6'),('agents/diagnosis/24-draft-alltoall-identity.json','bc572249e448c5470d3a9c59a08fa6f872c7e9674b78b37e33e926bad1ddf9ef'),('evidence/profiler-export-inventory.json','ff0f4c8a7e46a48ca53f376dc749406fdced163de9cdca000c7681ff41b592f7')]
for path,expected in files:
 if sha(T/path)!=expected:raise RuntimeError('input identity mismatch '+path)
dist=json.loads((T/files[0][0]).read_text());ids=json.loads((T/files[1][0]).read_text());inv=json.loads((T/files[2][0]).read_text())
targets=[x for x in ids['samples'] if '_ep0_' in x['rank'] or '_ep1_' in x['rank']]
if len(targets)!=6 or sorted(x['opId_local'] for x in targets)!=[402,402,405,405,408,408]:raise RuntimeError('unexpected target set')
sizes={r['rank']:next((f['bytes'] for f in r['files'] if f['path']=='ASCEND_PROFILER_OUTPUT/communication_matrix.json'),None) for r in inv['exports']}
result=[];matrix_files=[]
for rank in sorted({x['rank'] for x in targets}):
 sample=[x for x in targets if x['rank']==rank];base=R/rank/'ASCEND_PROFILER_OUTPUT';cp=base/'kernel_details.csv';dbs=list(base.glob('ascend_pytorch_profiler_*.db'))
 if len(dbs)!=1:raise RuntimeError('database cardinality')
 db=dbs[0];expected=dist['file_sha256_by_rank'][rank]
 if sha(cp)!=expected['csv'] or sha(db)!=expected['db']:raise RuntimeError('frozen CSV/DB mismatch')
 matrix=base/'communication_matrix.json';before=matrix.stat()
 if before.st_size!=sizes[rank]:raise RuntimeError('matrix size mismatch')
 raw=matrix.read_bytes();doc=json.loads(raw);after=matrix.stat()
 if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('matrix changed')
 matrix_files.append({'rank':rank,'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'identity_limit':'previous inventory only froze path and size'})
 matrix_matches={x['op_name_sha256']:[] for x in sample}
 def inspect(v,depth=0):
  if depth>10:return
  if isinstance(v,dict):
   for key,value in v.items():
    if not isinstance(key,str):continue
    for h in matrix_matches:
     if digest(key)==h or digest(key.partition('@')[0])==h:
      # Values are restricted to anonymous numeric metrics, never arbitrary strings.
      metrics=[]
      def gather(n,level=0):
       if level>5:return
       if isinstance(n,dict):
        for k,w in n.items():
         if isinstance(k,str) and type(w) in (int,float) and len(k)<80:metrics.append({'field':k,'value':w})
         elif isinstance(w,(dict,list)):gather(w,level+1)
       elif isinstance(n,list):
        for w in n[:100]:gather(w,level+1)
      gather(value);matrix_matches[h].append({'has_at_suffix':'@' in key,'node_type':type(value).__name__,'numeric_metrics':metrics[:100],'numeric_metrics_truncated':len(metrics)>100})
    if isinstance(value,(dict,list)):inspect(value,depth+1)
  elif isinstance(v,list):
   for child in v[:1000]:inspect(child,depth+1)
 inspect(doc)
 contexts=[]
 for x in sample:
  matches=[(b,h) for b in dist['brackets'] if b['rank']==rank for h in b['alltoall'] if len(h['exact_task_matches'])==1 and h['exact_task_matches'][0]['globalTaskId']==x['selected_task_global_id']]
  if len(matches)!=1:raise RuntimeError('selected task not unique')
  b,h=matches[0]; contexts.append((x,b,h))
 rows={x['opId_local']:[] for x,_,_ in contexts}
 with cp.open(newline='') as f:
  for row in csv.DictReader(f):
   if row['Model ID']!='4294967295':continue
   start=int(Decimal(row['Start Time(us)'].strip())*1000);end=start+int(Decimal(row['Duration(us)'].strip())*1000)
   for x,b,h in contexts:
    # Main stream: whole HcPre-to-HcPost attention bracket. Communication stream: +/-100us around selected task.
    main=row['Stream ID']==b['main_stream'] and start>=b['bracket_start_ns'] and end<=b['attention_end_ns']
    comm=row['Stream ID']==h['stream'] and start>=h['alltoall_start_ns']-100000 and end<=h['alltoall_end_ns']+100000
    if main or comm:
     rows[x['opId_local']].append({'stream_role':'main' if main else 'comm','type':row['Type'],'start_relative_us':(start-h['alltoall_start_ns'])/1000,'duration_us':(end-start)/1000,'input_shapes':row.get('Input Shapes'),'input_dtypes':row.get('Input Data Types'),'output_shapes':row.get('Output Shapes')})
 con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row;con.execute('PRAGMA query_only=ON');device=int(con.execute('SELECT DISTINCT deviceId FROM RANK_DEVICE_MAP').fetchone()[0])
 for x,b,h in contexts:
  waits=[dict(q) for q in con.execute('''SELECT t.startNs,t.endNs,t.streamId FROM TASK t JOIN STRING_IDS s ON s.id=t.taskType WHERE t.deviceId=? AND t.modelId=? AND t.streamId=? AND s.value='EVENT WAIT' AND t.startNs<? AND t.endNs>? ORDER BY t.startNs LIMIT 101''',(device,4294967295,int(b['main_stream']),h['alltoall_end_ns']+100000,h['alltoall_start_ns']-100000))]
  if len(waits)>100:raise RuntimeError('wait count bound exceeded')
  waits=[{'start_relative_us':(w['startNs']-h['alltoall_start_ns'])/1000,'duration_us':(w['endNs']-w['startNs'])/1000} for w in waits]
  selected=sorted(rows[x['opId_local']],key=lambda z:z['start_relative_us']);mm=matrix_matches[x['op_name_sha256']]
  result.append({'rank':rank,'opId_local':x['opId_local'],'op_name_sha256':x['op_name_sha256'],'comm_duration_us':x['duration_ns']/1000,'hc_shape':b['hc_input_shape'],'mc2_shape':b['mc2_input_shape'],'attention_rows_in_bracket':b['attention_rows_in_bracket'],'main_waits':waits,'local_kernel_rows':selected[:100],'local_kernel_rows_truncated':len(selected)>100,'matrix_matches':mm[:20],'matrix_matches_truncated':len(mm)>20})
 con.close()
print(json.dumps({'scope':'EP0 and EP1, three frozen op names each; local rank time only','files':matrix_files,'samples':result,'limits':['Matrix only size previously frozen; recorded SHA now.','Matrix nesting depth/arrays bounded; match by hashed key or prefix is a candidate, not same-call proof.','CSV only numeric main/comm streams, exported model ID and exact windows; graph-internal or N/A activity may be absent.','No cross-rank timestamp comparisons, request/collective identity or causality inferred.']},ensure_ascii=False,indent=2))
