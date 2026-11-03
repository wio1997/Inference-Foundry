#!/usr/bin/env python3
"""Inspect nine EP8 Q/KV windows in saved CSV, DB and trace. No target imports/NPU."""
import argparse,bisect,csv,hashlib,json,re,sqlite3
from decimal import Decimal,InvalidOperation
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
t=Path(a.task_root);rank='dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt';d=Path(a.trace_root)/rank/'ASCEND_PROFILER_OUTPUT'
wp=t/'captured-layer-attribution/module-call-metrics.json';mp=t/'evidence/TRACE-EXPORT-SHA256.json';csvpath=d/'kernel_details.csv';db=d/'ascend_pytorch_profiler_8.db';trace=d/'trace_view.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
if sha(db)!='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58':raise RuntimeError('DB differs')
tracehash=sha(trace)
if tracehash!='3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99':raise RuntimeError('trace differs')
manifest=json.loads(mp.read_text());expected={x['rank']:x['kernel_csv_sha256'] for x in manifest['exports']};csvhash=sha(csvpath)
if csvhash!=expected[rank]:raise RuntimeError('CSV differs')
windows=[x for x in json.loads(wp.read_text()) if x['rank']==rank and x['run_in_model'] in (1,2,3) and x['layer'] in (0,2,24)]
assert len(windows)==9
want={'QuantBatchMatmulV3','InplacePartialRotaryMul','ScatterNdUpdateSk','SparseFlashMla'}
events={(w['run_in_model'],w['layer']):[] for w in windows}
with csvpath.open(newline='') as f:
 for z in csv.DictReader(f):
  if z['Type'] not in want:continue
  start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
  for w in windows:
   if w['window_start_ns']<=start and end<=w['window_end_ns']:
    events[(w['run_in_model'],w['layer'])].append({'type':z['Type'],'stream':z['Stream ID'],'model':z['Model ID'],
      'start_ns':start,'end_ns':end,'input_shapes':z['Input Shapes']})
con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
con.execute('PRAGMA query_only=ON');con.execute('PRAGMA temp_store=MEMORY')
selected=[];anchors=[]
for w in windows:
 ev=events[(w['run_in_model'],w['layer'])];m=str(w['main_stream_id']);model=str(w['model_id']);same=lambda e:e['model']==model
 q=[e for e in ev if same(e) and e['stream']==m and e['type']=='QuantBatchMatmulV3' and e['input_shapes'].strip('"').startswith('3,1280;1024,80,16,32;')]
 a=[e for e in ev if same(e) and e['stream']==m and e['type']=='SparseFlashMla']
 if len(q)!=1 or len(a)!=1:raise RuntimeError('ambiguous Q/attention candidates')
 r=[e for e in ev if same(e) and e['stream']==m and e['type']=='InplacePartialRotaryMul' and e['input_shapes'].strip('"').startswith('3,1,64,512;') and e['end_ns']<=a[0]['start_ns']]
 s=[e for e in ev if same(e) and e['stream']!=m and e['type']=='ScatterNdUpdateSk' and e['input_shapes'].strip('"').endswith(';24,2;24,512')]
 if len(r)!=1 or len(s)!=1:raise RuntimeError('ambiguous rope/scatter candidates')
 start=min(q[0]['start_ns'],s[0]['start_ns']);end=r[0]['end_ns']
 rows=[dict(v) for v in con.execute('''SELECT t.startNs,t.endNs,t.deviceId,t.streamId,t.modelId,t.taskId,t.globalTaskId,t.connectionId,t.taskType,st.value AS task_type_name
 FROM TASK t LEFT JOIN STRING_IDS st ON st.id=t.taskType
 WHERE t.deviceId=8 AND t.modelId=? AND t.streamId IN (?,?) AND t.startNs<? AND t.endNs>=?
 AND (st.value LIKE '%EVENT%' OR st.value LIKE '%NOTIFY%' OR st.value LIKE '%WAIT%' OR st.value LIKE '%SYNC%')
 ORDER BY t.startNs,t.streamId LIMIT 101''',(int(model),int(m),int(s[0]['stream']),end,start))]
 if len(rows)>100:raise RuntimeError('event rows exceeded bound')
 runlayer={'run':w['run_in_model'],'layer':w['layer'],'candidate_qb':q[0],'candidate_qrope':r[0],'candidate_scatter':s[0],'event_tasks':rows};selected.append(runlayer)
 for e in rows:anchors.append({'run':w['run_in_model'],'layer':w['layer'],**e})
con.close()
anchors.sort(key=lambda e:e['startNs']);starts=[e['startNs'] for e in anchors];matches=[[] for e in anchors];totals=[0]*len(anchors)
valid_key=re.compile(r'(?i)^(event|stream|task|model|flow|device|connection|correlation|notify|global|graph|trace|id|pid|tid|.*_id|.* id)[a-z0-9_ ]{0,32}$')
valid_id=re.compile(r'^(?:[0-9]{1,30}|0x[0-9a-fA-F]{1,32})$')
def clean_id(v):
 if isinstance(v,bool):return None
 if isinstance(v,int):return v
 if isinstance(v,str) and valid_id.fullmatch(v):return v
 if isinstance(v,dict):return {k:clean_id(q) for k,q in v.items() if k in ('local','global') and clean_id(q) is not None}
 return None
decoder=json.JSONDecoder(parse_float=Decimal);chunk=1048576;max_record=4194304;before=trace.stat();count=0
with trace.open('r',encoding='utf-8') as f:
 buf=f.read(chunk);pos=0;started=False;eof=False
 while True:
  while pos<len(buf) and (buf[pos].isspace() or buf[pos]==','):pos+=1
  if not started:
   if pos>=len(buf) or buf[pos]!='[':raise RuntimeError('unexpected trace format')
   pos+=1;started=True;continue
  if pos>=len(buf):
   if eof:raise RuntimeError('unexpected EOF')
   buf=f.read(chunk);pos=0;eof=not buf;continue
  if buf[pos]==']':break
  try:obj,end=decoder.raw_decode(buf,pos)
  except json.JSONDecodeError:
   rem=buf[pos:];more=f.read(chunk)
   if not more or len(rem)+len(more)>max_record:raise RuntimeError('truncated trace or record exceeds cap')
   buf=rem+more;pos=0;continue
  pos=end;count+=1
  if not isinstance(obj,dict) or obj.get('ph')!='X' or obj.get('name') not in ('EVENT WAIT','EVENT RECORD'):continue
  try:ns=int(Decimal(str(obj.get('ts')))*1000)
  except (InvalidOperation,ValueError,TypeError):continue
  args=obj.get('args') if isinstance(obj.get('args'),dict) else {}
  for idx in range(bisect.bisect_left(starts,ns-2000),bisect.bisect_right(starts,ns+2000)):
   an=anchors[idx]
   if obj.get('name')!=an['task_type_name'] or obj.get('tid')!=an['streamId']:continue
   if args.get('Task Id')!=an['taskId'] or args.get('Model Id')!=an['modelId']:continue
   totals[idx]+=1
   if len(matches[idx])<10:
    vals={k:clean_id(v) for k,v in args.items() if isinstance(k,str) and valid_key.fullmatch(k) and clean_id(v) is not None}
    matches[idx].append({'name':obj.get('name'),'timestamp_ns':ns,'stream':obj.get('tid'),
      'id':clean_id(obj.get('id')),'id2':clean_id(obj.get('id2')),'numeric_identifier_args':vals})
after=trace.stat()
if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise RuntimeError('trace changed')
print(json.dumps({'scope':'EP8 three runs x layers 0,2,24; only candidate Qb→Q-rope window event rows and matching exported trace args',
 'window_sha256':sha(wp),'db_sha256':sha(db),'csv_sha256':csvhash,'trace_sha256':tracehash,'items_scanned':count,
 'windows':selected,'event_trace_matches':[{'anchor':v,'matched_trace_events':matches[i],'match_count':totals[i],
 'truncated':totals[i]>10} for i,v in enumerate(anchors)],
 'limitations':['Shape and timing do not establish tensor identity or event causal edge.',
 'Task/event labels are profiler exports; no event-object IDs means record-to-wait cannot be paired by proximity or connection alone.',
 'Trace matching allows up to 2us and requires same name/stream/model/taskId; candidate only, cannot prove identity.',
 'No matching identifiers or fields does not establish absence in runtime, other traces or code.',
 'No NPU operations, tensor values or requests were read.']},ensure_ascii=False,indent=2))
