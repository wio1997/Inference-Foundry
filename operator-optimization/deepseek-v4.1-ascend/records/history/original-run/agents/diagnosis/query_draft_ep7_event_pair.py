#!/usr/bin/env python3
"""EP7 six preselected tasks: bounded DB links and trace event-identity fields."""
import argparse,hashlib,json,re,sqlite3
from decimal import Decimal,InvalidOperation
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--db',required=True);ap.add_argument('--trace',required=True);a=ap.parse_args()
T=Path(a.task_root);db=Path(a.db).resolve(strict=True);trace=Path(a.trace).resolve(strict=True);prior=T/'agents/diagnosis/draft-ep7-long-interval.json'
rank='dp0_pp0_tp7_dcp0_ep7_rank7_4869_20261028020238581_ascend_pt'
if (trace.name,trace.parent.name,trace.parent.parent.name)!=('trace_view.json','ASCEND_PROFILER_OUTPUT',rank) or trace.stat().st_size!=163920234:raise RuntimeError('trace identity mismatch')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(db)!='8b8e72a61e2521f0d91050bc53ab852ddf26dd4e40e2d5ec5d3973652fd807c6':raise RuntimeError('DB hash mismatch')
if sha(prior)!='575eaa12caf97e612615e8af27069b54f4ab761d1c205a87d104aad656e8fedb':raise RuntimeError('prior hash mismatch')
trace_hash=sha(trace);before=trace.stat()
prev=json.loads(prior.read_text());selected=[r for r in prev['device_task_rows_all_streams'] if r['globalTaskId'] in (91804,91805,4301,91806,91807,91808)]
if len(selected)!=6:raise RuntimeError('selected task count mismatch')
targets={(x['startNs'],x['streamId'],x['taskId']):x for x in selected};starts=sorted({x['startNs'] for x in selected})
def cleanid(x):
 if type(x) is int or (isinstance(x,str) and re.fullmatch(r'(?:[0-9]{1,32}|0x[0-9a-fA-F]{1,32})',x)):
  return hashlib.sha256(str(x).encode()).hexdigest()
 return None
def safeapi(x):
 name=x.pop('name',None);lower=str(name).lower()
 x['name_class']=[s for s in ('event','record','wait','notify','alltoall') if s in lower]
 x['name_sha256']=hashlib.sha256(str(name).encode()).hexdigest()
 return x
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
def lim(sql,args=(),cap=30):
 r=[dict(x) for x in c.execute(sql,args).fetchmany(cap+1)]
 return {'rows':r[:cap],'truncated':len(r)>cap,'limit':cap}
links=[]
for x in selected:
 conn=x['connectionId'];gid=x['globalTaskId']
 exact=lim('SELECT * FROM TASK WHERE startNs=? AND endNs=? AND streamId=? AND taskId=? AND modelId=?', (x['startNs'],x['endNs'],x['streamId'],x['taskId'],x['modelId']),3)
 cann=lim('SELECT a.connectionId,a.startNs,a.endNs,s.value AS name FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=?',(conn,))
 cann['rows']=[safeapi(r) for r in cann['rows']]
 sets=lim('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId=?',(conn,))
 pyt=[]
 for g in sets['rows']:
  a=lim('SELECT a.connectionId,a.sequenceNumber,a.callchainId,s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=?',(g['id'],))
  a['rows']=[safeapi(r) for r in a['rows']];pyt.append({'connection_set_id':g['id'],'matches':a})
 comm=lim('SELECT * FROM COMMUNICATION_TASK_INFO WHERE globalTaskId=?',(gid,))
 comm_op=lim('SELECT * FROM COMMUNICATION_OP WHERE connectionId=?',(conn,))
 links.append({'target':x,'exact_task':exact,'cann_same_connection':cann,'connection_sets':sets,'pytorch_via_sets':pyt,'communication_task_gid_exact':comm,'communication_op_connection_exact':comm_op})
c.close()
key_re=re.compile(r'(?i)^[a-z0-9_ ]{1,70}$');argtag=re.compile(r'(?i)(event|notify|handle|address|stream|task|model|connection|batch|subtask|(^|_)id($|_))')
trace_rows=[];flow_rows=[];flow_count=0;decoder=json.JSONDecoder(parse_float=Decimal);count=0;chunk=1048576;max_record=4194304
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
   if not more or len(rem)+len(more)>max_record:raise RuntimeError('trace item too large or truncated')
   buf=rem+more;pos=0;continue
  pos=end;count+=1
  if not isinstance(obj,dict) or obj.get('ph') not in ('X','s','t','f') or 'ts' not in obj:continue
  try:ns=int(Decimal(str(obj['ts']))*1000)
  except (InvalidOperation,TypeError,ValueError):continue
  if obj.get('ph')=='X':
   args=obj.get('args',{});args=args if isinstance(args,dict) else {}
   taskid=args.get('Task Id');streamid=args.get('Physic Stream Id');key=(ns,streamid,taskid)
   if key not in targets:continue
   rawname=obj.get('name');type_name=targets[key]['task_type_name']
   if type_name.startswith('EVENT') and rawname!=type_name:continue
   fields=[]
   for k,v in args.items():
    if isinstance(k,str) and key_re.fullmatch(k) and argtag.search(k):fields.append({'key':k,'value_id_sha256':cleanid(v),'value_type':type(v).__name__})
   trace_rows.append({'target_gid':targets[key]['globalTaskId'],'trace_name_is_expected_event':rawname==type_name,
    'exact_start_stream_task_match':True,'fields':fields,'top_level_id_sha256':cleanid(obj.get('id'))})
  else:
   if not any(abs(ns-t)<=1000 for t in starts):continue
   flow_count+=1
   if len(flow_rows)<100:
    cat=obj.get('cat');flow_rows.append({'ts_ns':ns,'tid':obj.get('tid') if type(obj.get('tid')) is int else None,
     'phase':obj['ph'],'category':'HostToDevice' if cat=='HostToDevice' else 'OTHER_OR_EMPTY','flow_id_sha256':cleanid(obj.get('id'))})
after=trace.stat()
if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise RuntimeError('trace changed during scan')
print(json.dumps({'scope':'EP7 six exact stage131 task rows and trace events, nearby flow +/-1us',
 'db_sha256':sha(db),'prior_sha256':sha(prior),'trace_sha256':trace_hash,'trace_identity':'path and frozen inventory size checked; hash first recorded here',
 'trace_items_scanned':count,'links':links,'matching_trace_events':trace_rows,'nearby_flow_rows':flow_rows,
 'nearby_flow_count':flow_count,'nearby_flow_truncated':flow_count>100,
 'limits':['Trace outputs only selected identity-related arg keys and hashes of numeric IDs; other raw args not disclosed or evaluated.',
 'Connection IDs may be reused; exact DB or temporal matches do not establish event-object pair, data dependency, request or causality.',
 'No event-object identifier found under selected keys does not prove no identifier elsewhere in the trace or runtime.',
 'COMMUNICATION_OP association by exact connection is only a candidate; task duration is not necessarily network transfer time.',
 'Only one EP7 window, no host-device subtraction, no NPU activity.']},ensure_ascii=False,indent=2))
