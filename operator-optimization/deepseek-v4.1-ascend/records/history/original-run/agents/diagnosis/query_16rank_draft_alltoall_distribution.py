#!/usr/bin/env python3
"""Frozen draft-candidate brackets: numeric-stream alltoall + same-rank waits."""
import argparse,bisect,csv,hashlib,json,sqlite3
from collections import defaultdict,Counter
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root)
mp=T/'captured-layer-attribution/module-call-metrics.json';cm=T/'evidence/TRACE-EXPORT-SHA256.json';dm=T/'evidence/used-raw-source-sha256.json';audit=T/'eager-draft-mainstream-attribution-v2/summary.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for p,h in ((mp,'d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a'),(cm,'45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'),(dm,'cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a'),(audit,'1125b875dceb727ee5bf21c1816cbed20fcc7ce91e3b2fc30a0054a18e36e43b')):
 if sha(p)!=h:raise RuntimeError('input hash mismatch: '+p.name)
mapping=json.loads(mp.read_text());csv_manifest={r['rank']:r['kernel_csv_sha256'] for r in json.loads(cm.read_text())['exports']}
db_manifest={str(r['relative_path']):r['sha256'] for r in json.loads(dm.read_text())['records'] if '/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_' in r['relative_path'] and r['relative_path'].endswith('.db')}
audits={r['rank']:r for r in json.loads(audit.read_text())['rank_audit']};out=[];issues=[];filehash={}
def union(intervals):
 total=0;last=None
 for a,b in sorted(intervals):
  if last is None or a>last:total+=b-a;last=b
  elif b>last:total+=b-last;last=b
 return total
for rank in sorted(audits):
 root=R/rank/'ASCEND_PROFILER_OUTPUT';cp=root/'kernel_details.csv';dbs=list(root.glob('ascend_pytorch_profiler_*.db'))
 if len(dbs)!=1:raise RuntimeError('expected one database for '+rank)
 db=dbs[0];relative=str(db.relative_to(R));ch=sha(cp);dh=sha(db)
 if ch!=csv_manifest.get(rank) or dh!=db_manifest.get(relative):raise RuntimeError('frozen hash mismatch: '+rank)
 filehash[rank]={'csv':ch,'db':dh};rows=[];markers=defaultdict(list);streams=defaultdict(list)
 with cp.open(newline='') as f:
  for row in csv.DictReader(f):
   row['_start']=int(Decimal(row['Start Time(us)'].strip())*1000);row['_end']=row['_start']+int(Decimal(row['Duration(us)'].strip())*1000);rows.append(row)
   streams[(row['Model ID'],row['Stream ID'])].append(row)
   if row['Type'] in ('HcPre','HcPost'):markers[(row['Model ID'],row['Stream ID'])].append(row)
 for xs in markers.values():xs.sort(key=lambda x:x['_start'])
 mods=[m for m in mapping if m['rank']==rank and m['run_in_model'] in (1,2,3)]
 if len(mods)!=120:raise RuntimeError('wrong layer window count: '+rank)
 gaps=[]
 for i in (1,2):
  left=[x for x in mods if x['run_in_model']==i and x['layer']==39];right=[x for x in mods if x['run_in_model']==i+1 and x['layer']==0]
  if len(left)!=1 or len(right)!=1:raise RuntimeError('bad target anchors')
  gaps.append((left[0]['window_end_ns'],right[0]['window_start_ns']))
 con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row;con.execute('PRAGMA query_only=ON');con.execute('PRAGMA temp_store=MEMORY')
 device_rows=con.execute('SELECT DISTINCT deviceId FROM RANK_DEVICE_MAP').fetchall()
 if len(device_rows)!=1:raise RuntimeError('rank/device mapping not unique: '+rank)
 device=int(device_rows[0]['deviceId'])
 counts=Counter()
 for row in rows:
  if row['Model ID']!='4294967295' or row['Type']!='DispatchFFNCombine':continue
  ss=row['Input Shapes'].strip('"').split(';')
  try:valid=int(ss[1].split(',')[0])==8 and int(ss[3].split(',')[-1])==3
  except (ValueError,IndexError):valid=False
  containing=[i for i,(a,b) in enumerate(gaps,1) if a<=row['_start'] and row['_end']<=b]
  if not valid or not containing:continue
  counts['candidates']+=1;key=(row['Model ID'],row['Stream ID']);ms=markers[key];i=bisect.bisect_right([x['_start'] for x in ms],row['_start'])
  if i<3 or i>=len(ms):issues.append({'rank':rank,'reason':'insufficient markers'});continue
  x,y,z,v=ms[i-3:i+1]
  if [r['Type'] for r in (x,y,z,v)]!=['HcPre','HcPost','HcPre','HcPost'] or z['_end']>row['_start'] or row['_end']>v['_start']:
   issues.append({'rank':rank,'reason':'marker boundary'});continue
  same=[r for r in streams[key] if x['_start']<=r['_start'] and r['_end']<=v['_end']]
  if sum(r['Type']=='DispatchFFNCombine' for r in same)!=1:issues.append({'rank':rank,'reason':'mc2 not unique'});continue
  area=[r for r in rows if r['Model ID']==row['Model ID'] and x['_start']<=r['_start'] and r['_end']<=y['_end']]
  natt=sum(r['Type']=='SparseFlashMla' for r in area)
  comm=[r for r in area if r['Type']=='hcom_alltoall_' and r['Stream ID'].isdigit()]
  if len(comm)>20:raise RuntimeError('too many alltoall rows')
  detail=[]
  for h in comm:
   matches=[dict(r) for r in con.execute('''SELECT globalTaskId,connectionId,startNs,endNs,taskId,streamId,modelId FROM TASK
    WHERE deviceId=? AND modelId=? AND streamId=? AND taskId=? AND startNs=? AND endNs=? LIMIT 3''',
    (device,int(h['Model ID']),int(h['Stream ID']),int(h['Task ID']),h['_start'],h['_end']))] if h['Task ID'].isdigit() else []
   cop=[]
   if len(matches)==1:
    cop=[dict(r) for r in con.execute('''SELECT opId,startNs,endNs,count,dataType,rankSize,connectionId FROM COMMUNICATION_OP
      WHERE deviceId=? AND connectionId=? AND startNs=? AND endNs=? LIMIT 3''',(device,matches[0]['connectionId'],h['_start'],h['_end']))]
   waits=[dict(r) for r in con.execute('''SELECT t.startNs,t.endNs,t.streamId,t.taskId,t.globalTaskId,t.connectionId FROM TASK t
    JOIN STRING_IDS s ON s.id=t.taskType WHERE t.deviceId=? AND t.modelId=? AND t.streamId=? AND s.value='EVENT WAIT'
    AND t.startNs<? AND t.endNs>? ORDER BY t.startNs LIMIT 101''',
    (device,int(row['Model ID']),int(row['Stream ID']),h['_end'],h['_start']))]
   if len(waits)>100:raise RuntimeError('too many wait rows')
   clipped=[(max(r['startNs'],h['_start'],x['_start']),min(r['endNs'],h['_end'],y['_end'])) for r in waits]
   clipped=[(a,b) for a,b in clipped if a<b]
   detail.append({'alltoall_start_ns':h['_start'],'alltoall_end_ns':h['_end'],'duration_ns':h['_end']-h['_start'],
    'stream':h['Stream ID'],'task':h['Task ID'],'exact_task_matches':matches,'exact_communication_op_matches':cop,
    'main_event_waits_overlapping':waits,'wait_overlap_union_within_communication_and_bracket_ns':union(clipped)})
  out.append({'rank':rank,'gap_ordinal':containing[0],'bracket_start_ns':x['_start'],'attention_end_ns':y['_end'],
   'main_stream':row['Stream ID'],'hc_input_shape':x['Input Shapes'],'mc2_input_shape':row['Input Shapes'],
   'attention_rows_in_bracket':natt,'status':'one_exported_smla' if natt==1 else ('no_exported_smla' if natt==0 else 'multiple_exported_smla'),
   'alltoall_numeric_count':len(detail),'alltoall':detail})
 con.close()
 if counts['candidates']!=audits[rank]['draft_mc2_examined']:issues.append({'rank':rank,'reason':'candidate count mismatch','got':counts['candidates'],'expected':audits[rank]['draft_mc2_examined']})
print(json.dumps({'scope':'Frozen 16 rank, two inter-target windows each; all candidate four-HC-marker draft brackets',
 'input_sha256':{'mapping':sha(mp),'csv_manifest':sha(cm),'db_manifest':sha(dm),'prior_audit':sha(audit)},
 'file_sha256_by_rank':filehash,'issues':issues,'brackets':out,
 'limits':['Candidate brackets are source/shape based and do not prove request, stage, block ordinal, Q length or identical work.',
 'Only numeric stream alltoall CSV rows fully within attention bracket; N/A rows excluded to avoid double counting, not proof of no other communications.',
 'Each rank uses its own device timestamp; never compare absolute times across ranks.',
 'Main EVENT WAIT overlap is temporal only, not evidence of event-object pair or communication causality.',
 'Communication task duration includes implementation, scheduling and waiting; not pure network time.',
 'Same shapes and count are not proof of same peer topology or request workload; unassigned outputs stay unresolved.']},ensure_ascii=False,indent=2))
