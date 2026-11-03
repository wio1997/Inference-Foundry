#!/usr/bin/env python3
"""Stream the existing EP8 trace for bounded, redacted target event/flow metadata."""
import argparse, bisect, hashlib, json, re
from decimal import Decimal, InvalidOperation
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace',required=True);a=ap.parse_args()
T=Path(a.task_root);p=Path(a.trace).resolve(strict=True)
expected_rank='dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt'
if p.name!='trace_view.json' or p.parent.name!='ASCEND_PROFILER_OUTPUT' or p.parent.parent.name!=expected_rank:
 raise RuntimeError('Unexpected rank or trace path')
before=p.stat()
if before.st_size!=150646621:raise RuntimeError('Unexpected trace size; identity not established')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
trace_hash=sha(p)
exact=T/'agents/diagnosis/engram-producer-events-exact.json'
matmul=T/'agents/diagnosis/engram-matmul-task-links.json'
if sha(exact)!='ea31f7d421d10109a6cab2a14d13c084dda169761b4085177a658741821964cb':raise RuntimeError('Unexpected stage120 results')
if sha(matmul)!='3e6196f3005366117421c78a359dc036e9d79f7299f2b72d239c2401202e76d1':raise RuntimeError('Unexpected stage114 results')
e=json.loads(exact.read_text());mm=json.loads(matmul.read_text());targets=[]
for win in e['producer_windows']:
 if win['exact_named_producers']['truncated']:raise RuntimeError('Producer rows truncated')
 for row in win['exact_named_producers']['rows']:
  targets.append({'run':win['run'],'kind':row['compute_name'],'start_ns':int(row['startNs']),
   'end_ns':int(row['endNs']),'stream':row['streamId'],'global_task_id':row['globalTaskId']})
for gap in e['gap_windows']:
 if gap['event_task_rows']['truncated']:raise RuntimeError('Gap rows truncated')
 for row in gap['event_task_rows']['rows']:
  if row['task_type_name']=='EVENT WAIT' and row['streamId'] in (87,88) and row['modelId']==49:
   targets.append({'run':gap['run'],'after_layer':gap['after_layer'],'kind':'EVENT WAIT',
    'start_ns':int(row['startNs']),'end_ns':int(row['endNs']),'stream':row['streamId'],'global_task_id':row['globalTaskId']})
for s in mm['samples']:
 t=s['target'];targets.append({'run':t['run'],'after_layer':t['after_layer'],'kind':'MatMulV2',
  'start_ns':int(t['start_ns']),'end_ns':int(t['end_ns']),'stream':t['stream'], 'global_task_id':s['exact_task']['globalTaskId']})
if len(targets)!=28:raise RuntimeError('Unexpected target count: '+str(len(targets)))
targets.sort(key=lambda x:x['start_ns']); starts=[t['start_ns'] for t in targets]
matches=[[] for _ in targets]; totals=[0]*len(targets)
valid_key=re.compile(r'(?i)^(event|stream|task|model|flow|device|connection|correlation|notify|global|graph|trace|id|pid|tid|.*_id|.* id)[a-z0-9_ ]{0,32}$')
valid_id=re.compile(r'^(?:[0-9]{1,30}|0x[0-9a-fA-F]{1,32})$')
def identifier(value):
 if isinstance(value,bool):return None
 if isinstance(value,int):return value
 if isinstance(value,str) and valid_id.fullmatch(value):return value
 if isinstance(value,dict):return {k:identifier(v) for k,v in value.items() if k in ('local','global') and identifier(v) is not None}
 return None
def clean_event(obj):
 name=obj.get('name');raw_name=name if isinstance(name,str) else ''
 allowed_names={'EVENT WAIT','EVENT RECORD','MatMulV2','_engram_req_index_kernel','_write_hash_cache_kernel','_hash_ids_kernel','_engram_host_uva_gather_dequant_kernel'}
 safe_name=raw_name if raw_name in allowed_names else None
 args=obj.get('args',{});keys=[];values={}
 if isinstance(args,dict):
  for k,v in args.items():
   if not isinstance(k,str) or not valid_key.fullmatch(k):continue
   keys.append(k);n=identifier(v)
   if n is not None:values[k]=n
 return {'name':safe_name,'other_name_sha256':None if safe_name else hashlib.sha256(raw_name.encode()).hexdigest(),
  'phase':obj.get('ph') if obj.get('ph') in ('X','s','t','f','b','e','i','I') else None,
  'timestamp_us':str(obj.get('ts')),'duration_us':str(obj.get('dur')) if obj.get('dur') is not None else None,
  'pid':identifier(obj.get('pid')),'tid':identifier(obj.get('tid')), 'id':identifier(obj.get('id')),
  'id2':identifier(obj.get('id2')),'relevant_arg_keys':sorted(keys)[:40], 'numeric_identifier_args':values}

# Only the 28 already selected start times are candidates. A +-2 us timestamp
# association is deliberately not treated as a task or clock identity proof.
decoder=json.JSONDecoder(parse_float=Decimal);chunk=1024*1024;max_record=4*1024*1024;events=0
with p.open('r',encoding='utf-8') as f:
 buf=f.read(chunk);pos=0;started=False;eof=False
 while True:
  while pos<len(buf) and (buf[pos].isspace() or buf[pos]==','):pos+=1
  if not started:
   if pos>=len(buf) or buf[pos]!='[':raise RuntimeError('Unexpected trace format')
   pos+=1;started=True;continue
  if pos>=len(buf):
   if eof:raise RuntimeError('Unexpected EOF')
   buf=f.read(chunk);pos=0;eof=not buf;continue
  if buf[pos]==']':break
  try:obj,end=decoder.raw_decode(buf,pos)
  except json.JSONDecodeError:
   rem=buf[pos:];more=f.read(chunk)
   if not more or len(rem)+len(more)>max_record:raise RuntimeError('Truncated or oversized trace item')
   buf=rem+more;pos=0;continue
  pos=end;events+=1
  if not isinstance(obj,dict) or 'ts' not in obj:continue
  try:ns=int(Decimal(str(obj['ts']))*1000)
  except (InvalidOperation,ValueError,TypeError):continue
  lo=bisect.bisect_left(starts,ns-2000);hi=bisect.bisect_right(starts,ns+2000)
  for idx in range(lo,hi):
   totals[idx]+=1
   if len(matches[idx])<30:matches[idx].append(clean_event(obj))
after=p.stat()
if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise RuntimeError('Trace changed during read')
rows=[{'target':t,'nearby_trace_events':matches[i],'total_nearby':totals[i], 'truncated':totals[i]>30} for i,t in enumerate(targets)]
print(json.dumps({'scope':'Selected EP8 tasks and nearby trace events, +/- 2 us', 'trace_size':before.st_size,
 'trace_sha256':trace_hash,'identity_limit':'Path and expected inventory size checked; no previously frozen EP8 trace hash is available, so hash is recorded but not independently verified.',
 'exact_sha256':sha(exact),'matmul_sha256':sha(matmul),'items_scanned':events,'targets':rows,
 'limitations':['No prompt, request text, tensor contents or arbitrary argument values are emitted.',
 'Timestamp proximity is an unverified candidate association, even for identical names.',
 'Flow ID presence does not prove the target event endpoint unless exporter semantics and both endpoints are established.',
 'No matches can mean clock conversion, trace formatting, filtering, sampling or capture gaps; it does not prove no runtime event.',
 'Per-target 30-item output cap flagged; whole trace streamed with a 4 MiB individual record cap.']},ensure_ascii=False,indent=2))
