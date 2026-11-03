#!/usr/bin/env python3
"""Three fixed largest EP1 positive-task gaps: boundary task/API/flow evidence."""
import argparse,hashlib,json,sqlite3
from decimal import Decimal,InvalidOperation
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);prior=T/'agents/diagnosis/ep1-attention-lead-intervals.json';dist=T/'agents/diagnosis/16rank-draft-alltoall-distribution.json';manifest=T/'evidence/profiler-export-inventory.json'
rank='dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt';base=R/rank/'ASCEND_PROFILER_OUTPUT';db=base/'ascend_pytorch_profiler_1.db';trace=base/'trace_view.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1048576),b''):h.update(z)
 return h.hexdigest()
for p,expected in [(prior,'3f98642af4e6dc3ad5c8582d30de5e6600685bb766ec1bfe1958110a0dd06326'),(dist,'633011028e6d5953f03f1493c197085bdd1a810c095cc646e4f5b0e115f61ec6'),(manifest,'ff0f4c8a7e46a48ca53f376dc749406fdced163de9cdca000c7681ff41b592f7'),(db,'1d823cb13f49703ebd0a86ade38030ac8d6ef66b7085b23acf06f19ea42fa38b')]:
 if sha(p)!=expected:raise RuntimeError('input mismatch '+p.name)
records=json.loads(manifest.read_text())['exports'];tr=[f for z in records if z['rank']==rank for f in z['files'] if f['path']=='ASCEND_PROFILER_OUTPUT/trace_view.json']
if len(tr)!=1 or trace.stat().st_size!=tr[0]['bytes']:raise RuntimeError('trace path/size mismatch')
trace_sha=sha(trace)
source=json.loads(prior.read_text());selected=[]
for s in source['samples']:
 gap=s['largest_main_task_gaps'][0];lo=gap['start_relative_us'];hi=lo+gap['duration_us'];rows=s['main_task_rows']
 before=[r for r in rows if r['duration_us']>0 and abs(r['start_relative_us']+r['duration_us']-lo)<0.001]
 after=[r for r in rows if r['duration_us']>0 and abs(r['start_relative_us']-hi)<0.001]
 interior=[r for r in rows if r['duration_us']==0 and lo<=r['start_relative_us']<hi]
 if len(before)!=1 or len(after)!=1:raise RuntimeError('gap boundary not unique: '+str(s['opId_local']))
 selected.extend([{'op':s['opId_local'],'role':role,**r} for role,group in [('before',before),('after',after),('zero_event_inside',interior)] for r in group])
if len(selected)>30:raise RuntimeError('target bound')
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
def safe(r):
 d=dict(r);name=str(d.pop('name',None));d['name_sha256']=hashlib.sha256(name.encode()).hexdigest();d['name_class']=[k for k in ('event','record','wait','notify','launch','kernel','quant','matmul') if k in name.lower()];return d
links=[];flowkeys={}
for x in selected:
 gid=x['globalTaskId'];conn=x['connectionId'];exact=[dict(q) for q in c.execute('SELECT startNs,endNs,streamId,taskId,modelId,globalTaskId,connectionId FROM TASK WHERE globalTaskId=? LIMIT 3',(gid,))]
 if len(exact)!=1:raise RuntimeError('global id not unique '+str(gid))
 api=[safe(q) for q in c.execute('SELECT a.connectionId,a.startNs,a.endNs,s.value AS name FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? LIMIT 31',(conn,))]
 sets=[dict(q) for q in c.execute('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId=? LIMIT 31',(conn,))]
 if len(api)>30 or len(sets)>30:raise RuntimeError('API relation bound')
 pyt=[]
 for z in sets:
  arr=[safe(q) for q in c.execute('SELECT a.connectionId,a.sequenceNumber,a.callchainId,s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? LIMIT 31',(z['id'],))]
  if len(arr)>30:raise RuntimeError('PyTorch relation bound')
  pyt.append({'connection_set_id':z['id'],'rows':arr})
 task=exact[0];links.append({'opId_local':x['op'],'role':x['role'],'task':task,'task_type_name':x['task_type_name'],'cann_same_connection':api,'connection_sets':sets,'pytorch_via_sets':pyt})
 for key,role in [(str(conn<<32),'connection_shift_32'),(str(task['startNs']),'device_start_ns')]:flowkeys.setdefault(key,[]).append({'opId_local':x['op'],'task_gid':gid,'role':role})
c.close()
decoder=json.JSONDecoder(parse_float=Decimal);flow={k:[] for k in flowkeys};counts={k:0 for k in flowkeys};event_matches=[];n=0;before=trace.stat()
lookup={(x['task']['startNs'],x['task']['streamId'],x['task']['taskId']):x for x in links}
with trace.open(encoding='utf8') as f:
 buf=f.read(1048576);pos=0;started=False;eof=False
 while True:
  while pos<len(buf) and (buf[pos].isspace() or buf[pos]==','):pos+=1
  if not started:
   if pos>=len(buf) or buf[pos]!='[':raise RuntimeError('unexpected trace format')
   pos+=1;started=True;continue
  if pos>=len(buf):
   if eof:raise RuntimeError('truncated trace')
   buf=f.read(1048576);pos=0;eof=not buf;continue
  if buf[pos]==']':break
  try:obj,end=decoder.raw_decode(buf,pos)
  except json.JSONDecodeError:
   rem=buf[pos:];more=f.read(1048576)
   if not more or len(rem)+len(more)>4194304:raise RuntimeError('truncated or oversized trace item')
   buf=rem+more;pos=0;continue
  pos=end;n+=1
  if not isinstance(obj,dict):continue
  ph=obj.get('ph')
  if ph not in ('s','t','f','X'):continue
  try:ts=int(Decimal(str(obj['ts']))*1000)
  except (KeyError,InvalidOperation,TypeError,ValueError):continue
  if ph=='X':
   args=obj.get('args',{});args=args if isinstance(args,dict) else {};key=(ts,args.get('Physic Stream Id'),args.get('Task Id'))
   if key in lookup:event_matches.append({'task_gid':lookup[key]['task']['globalTaskId'],'exact_start_stream_task':True,'event_name_sha256':hashlib.sha256(str(obj.get('name')).encode()).hexdigest()})
  else:
   ident=obj.get('id')
   if isinstance(ident,bool) or not isinstance(ident,(str,int)) or str(ident) not in flow:continue
   key=str(ident);counts[key]+=1
   if len(flow[key])<30:
    cat=obj.get('cat');flow[key].append({'phase':ph,'timestamp_us':str(obj['ts']),'pid':obj.get('pid') if type(obj.get('pid')) is int else None,'tid':obj.get('tid') if type(obj.get('tid')) is int else None,'category':'HostToDevice' if cat=='HostToDevice' else 'OTHER_OR_EMPTY'})
after=trace.stat()
if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('trace changed')
print(json.dumps({'scope':'EP1 3 maximal positive-duration main TASK gaps: boundary tasks and zero events inside; fixed DB connections and candidate trace flow identifiers','input_sha256':{'prior':sha(prior),'db':sha(db),'trace':trace_sha},'trace_identity_limit':'path and size previously inventoried; SHA first recorded now','targets':links,'trace_events_exact':event_matches,'flow_candidates':[{'candidate_id_sha256':hashlib.sha256(k.encode()).hexdigest(),'derived_from':v,'count':counts[k],'rows':flow[k],'truncated':counts[k]>30} for k,v in flowkeys.items()],'trace_items_scanned':n,'limits':['Main gaps can contain zero-duration events; no assumption of device idle.','Same connection may be reused, and flow ID candidate connection<<32 is not a verified universal contract.','Exact start/stream/task trace key identifies candidates; no event identity or dependency proof.','Trace timestamps parsed as decimal microseconds; DB timestamps in ns. No host/device clock equivalence assumed, nor cross-domain subtraction.','PyTorch API mapping follows connection set records only; no match is not proof of no host launch.','Flow id2 or alternate encodings omitted; repeated IDs and capped results cannot establish unique endpoint association.','No new NPU activity.']},ensure_ascii=False,indent=2))
