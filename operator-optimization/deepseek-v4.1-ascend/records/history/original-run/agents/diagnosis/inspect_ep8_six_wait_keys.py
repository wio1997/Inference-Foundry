#!/usr/bin/env python3
"""Read one preselected trace; output only structural keys and event-id presence."""
import argparse, hashlib, json, re
from decimal import Decimal, InvalidOperation
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--task-root',required=True); ap.add_argument('--trace',required=True); a=ap.parse_args()
p=Path(a.trace).resolve(strict=True); root=Path(a.task_root).resolve(strict=True)
rank='dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt'
if (p.name,p.parent.name,p.parent.parent.name)!=('trace_view.json','ASCEND_PROFILER_OUTPUT',rank): raise RuntimeError('trace path mismatch')
before=p.stat()
if before.st_size!=150646621: raise RuntimeError('trace size mismatch')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
trace_hash=sha(p)
if trace_hash!='3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99': raise RuntimeError('trace differs from stage121')
selected=root/'agents/diagnosis/ep8-mc2-selected-events.json'
if sha(selected)!='ff97d5716e2bd1564b82303590c33b38226c1cfd72b8d5b06d5580d27165ca01': raise RuntimeError('selection differs from stage217')
x=json.loads(selected.read_text()); targets={}
for sample in x['samples']:
 for row in sample['selected_events']:
  key=(int(row['start_ns']),int(row['stream']),int(row['task_id']))
  if key in targets:raise RuntimeError('duplicate target')
  targets[key]={'run':sample['run'],'stream':row['stream'],'task_id':row['task_id']}
if len(targets)!=6:raise RuntimeError('expected six targets')
allowed_key=re.compile(r'^[A-Za-z0-9_ .:/-]{1,96}$')
def safe_key(k):
 if isinstance(k,str) and allowed_key.fullmatch(k):return k
 return 'sha256:'+hashlib.sha256(str(k).encode()).hexdigest()
def descriptor(v):
 return {'type':type(v).__name__,'non_null':v is not None}
def summarize(obj):
 args=obj.get('args'); args=args if isinstance(args,dict) else {}
 # Never output arbitrary trace values. Candidate names are leads, not established handles.
 candidates={safe_key(k):descriptor(v) for k,v in args.items() if isinstance(k,str) and re.search(r'event|notify|signal|handle|semaphore|fence|flow|correlation',k,re.I)}
 return {'top_level_keys':sorted(safe_key(k) for k in obj), 'arg_keys':sorted(safe_key(k) for k in args),
         'candidate_key_descriptors':candidates,'id':descriptor(obj.get('id')),'id2':descriptor(obj.get('id2'))}
found={key:[] for key in targets}; decoder=json.JSONDecoder(parse_float=Decimal);chunk=1024*1024;cap=4*1024*1024;count=0
with p.open('r',encoding='utf-8') as f:
 buf=f.read(chunk);pos=0;started=False;eof=False
 while True:
  while pos<len(buf) and (buf[pos].isspace() or buf[pos]==','):pos+=1
  if not started:
   if pos>=len(buf) or buf[pos]!='[':raise RuntimeError('trace format')
   pos+=1;started=True;continue
  if pos>=len(buf):
   if eof:raise RuntimeError('unexpected EOF')
   buf=f.read(chunk);pos=0;eof=not buf;continue
  if buf[pos]==']':break
  try:obj,end=decoder.raw_decode(buf,pos)
  except json.JSONDecodeError:
   rem=buf[pos:];more=f.read(chunk)
   if not more or len(rem)+len(more)>cap:raise RuntimeError('truncated or oversized record')
   buf=rem+more;pos=0;continue
  pos=end;count+=1
  if not isinstance(obj,dict) or obj.get('name')!='EVENT WAIT' or obj.get('ph')!='X':continue
  args=obj.get('args');
  if not isinstance(args,dict) or args.get('Model Id')!=49:continue
  try:
   n=Decimal(str(obj.get('ts')))*1000
   if n!=int(n):continue
   key=(int(n),obj.get('tid'),args.get('Task Id'))
  except (InvalidOperation,TypeError,ValueError):continue
  if key in found:found[key].append(summarize(obj))
after=p.stat()
if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise RuntimeError('trace changed during read')
if any(len(v)!=1 for v in found.values()):raise RuntimeError('six unique matches required: '+str([len(v) for v in found.values()]))
print(json.dumps({'trace_sha256':trace_hash,'trace_identity':'same hash as stage121; not independently frozen','selected_sha256':sha(selected),'items_scanned':count,'rows':[dict(targets[k],**found[k][0]) for k in targets], 'limits':['Keys and candidate presence only; no arbitrary values.','Key names alone do not prove runtime event identity or producer/consumer pairing.','Only six chosen waits; no general absence conclusion.']},ensure_ascii=False,indent=2))
