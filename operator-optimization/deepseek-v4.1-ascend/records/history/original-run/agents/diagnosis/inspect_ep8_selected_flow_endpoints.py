#!/usr/bin/env python3
"""Read only preselected flow IDs in the three existing EP8 time windows."""
import argparse, hashlib, json
from decimal import Decimal, InvalidOperation
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace',required=True);a=ap.parse_args()
T=Path(a.task_root);p=Path(a.trace).resolve(strict=True)
rank='dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt'
if (p.name,p.parent.name,p.parent.parent.name)!=('trace_view.json','ASCEND_PROFILER_OUTPUT',rank):raise RuntimeError('Wrong trace path')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
if sha(p)!='3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99':raise RuntimeError('Trace differs from stage121')
data=T/'agents/diagnosis/engram-producer-events-exact.json'
if sha(data)!='ea31f7d421d10109a6cab2a14d13c084dda169761b4085177a658741821964cb':raise RuntimeError('Unexpected producer input')
source=json.loads(data.read_text()); selected={}
for w in source['producer_windows']:
 if w['exact_named_producers']['truncated']:raise RuntimeError('Incomplete producer rows')
 run=w['run'];selected[run]={'start':w['lookback_start_ns'],'end':w['end_ns'],
    'ids':{'791691321671680':'shared_graph_connection'}}
 for r in w['exact_named_producers']['rows']:
  selected[run]['ids'][str(int(r['connectionId'])<<32)]='producer_connection'
  selected[run]['ids'][str(int(r['startNs']))]='producer_timestamp'
out={run:{k:{'role':v,'items':[],'total':0} for k,v in s['ids'].items()} for run,s in selected.items()}
decoder=json.JSONDecoder(parse_float=Decimal);events=0;chunk=1048576;max_record=4194304;before=p.stat()
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
  if not isinstance(obj,dict) or obj.get('ph') not in ('s','t','f'):continue
  ident=obj.get('id')
  if isinstance(ident,bool) or not isinstance(ident,(str,int)):continue
  key=str(ident)
  try:ts=int(Decimal(str(obj.get('ts')))*1000)
  except (InvalidOperation,ValueError,TypeError):continue
  for run,s in selected.items():
   if key not in s['ids'] or not(s['start']<=ts<s['end']):continue
   bucket=out[run][key];bucket['total']+=1
   if len(bucket['items'])<30:
    cat=obj.get('cat') if isinstance(obj.get('cat'),str) else ''
    bucket['items'].append({'phase':obj['ph'],'ts_us':str(obj['ts']),
     'pid':obj.get('pid') if isinstance(obj.get('pid'),int) else None,
     'tid':obj.get('tid') if isinstance(obj.get('tid'),int) else None,
     'category':'HostToDevice' if cat=='HostToDevice' else 'OTHER_OR_EMPTY',
     'other_category_sha256':None if cat=='HostToDevice' else hashlib.sha256(cat.encode()).hexdigest()})
after=p.stat()
if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise RuntimeError('Trace changed during read')
print(json.dumps({'scope':'Only preselected IDs and 3 per-run EP8 device candidate windows','trace_sha256':sha(p),'items_scanned':events,
 'rows':[{'run':run,'window_start_ns':selected[run]['start'],'window_end_ns':selected[run]['end'],
          'flow_id':ident,**val,'truncated':val['total']>30} for run,ids in out.items() for ident,val in ids.items()],
 'limitations':['Only top-level exact flow IDs selected; id2 and differently encoded identifiers are not searched.',
 'Only per-run 10ms lookback and graph interval searched; a flow endpoint outside it is not reported.',
 'Names and args omitted; no prompt or request content emitted.',
 'Flow/category identity and timestamps alone do not prove event-object, buffer, request or graph replay identity.',
 'Repeated flow IDs and output truncation prevent simplistic endpoint pairing; host and device timestamp differences are not calculated.']},ensure_ascii=False,indent=2))
