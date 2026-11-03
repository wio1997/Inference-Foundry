#!/usr/bin/env python3
"""Only eight frozen trace event args; no addresses or user values returned."""
import argparse,hashlib,hmac,json,re,secrets
from decimal import Decimal,InvalidOperation
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--trace',required=True);a=p.parse_args()
path=Path(a.trace).resolve(strict=True)
if path.name!='trace_view.json' or path.parent.name!='ASCEND_PROFILER_OUTPUT' or path.parent.parent.name!='dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt':raise RuntimeError('unexpected trace path')
expected='3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99';before=path.stat()
if before.st_size!=150646621:raise RuntimeError('unexpected trace size')
def sha():
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
if sha()!=expected:raise RuntimeError('trace hash mismatch')
# Source: stage121 exact device X events. start_ns, pid, tid, model, task, name hash.
G='aabefad5b9fffb556ccfb9c4bc60dfae3f09481af6f2a7e763d46d564cf1b67a';M='f9206cf5c15bb889ff4ca02e54034369ec963388ca74f7a8ab2bb6e86e3ece57'
targets=[(2,'gather',1793152960185069105,744904,47,4294967295,53682,G),(2,'gather',1793152960185562115,744904,47,4294967295,53689,G),(2,'matmul_after0',1793152960187571215,744904,89,49,72,M),(2,'matmul_after13',1793152960198990467,744904,89,49,730,M),(3,'gather',1793152960260742113,744904,47,4294967295,55039,G),(3,'gather',1793152960261206424,744904,47,4294967295,55046,G),(3,'matmul_after0',1793152960263910410,744904,89,49,72,M),(3,'matmul_after13',1793152960275488198,744904,89,49,730,M)]
lookup={(t[2],t[3],t[4],t[5],t[6],t[7]):i for i,t in enumerate(targets)}
if len(lookup)!=8:raise RuntimeError('nonunique anchors')
secret=secrets.token_bytes(32);safe={'Model Id','Physic Stream Id','Task Id','Task Type','Batch Id','Subtask Id','connection_id','Input Address','Output Address','Input Addr','Output Addr','Input Pointer','Output Pointer','Tensor Id','Tensor Address','Storage Id','Buffer Id','Input Shapes','Output Shapes','Input Data Types','Output Data Types'}
def digest(prefix,s):return hmac.new(secret,(prefix+str(s)).encode(),hashlib.sha256).hexdigest()
def num(v):
 if isinstance(v,bool):return None
 if isinstance(v,int):return v if 0<=v<2**128 else None
 if isinstance(v,str) and re.fullmatch(r'(?:0[xX][0-9a-fA-F]{1,32}|[0-9]{1,38})',v):
  n=int(v,16) if v.lower().startswith('0x') else int(v,10)
  return n if n<2**128 else None
 return None
def atoms(v,depth=0):
 if depth>2:return [],True
 n=num(v)
 if n is not None:return [n],False
 if isinstance(v,(list,tuple)):
  out=[];cut=len(v)>32
  for z in v[:32]:
   part,flag=atoms(z,depth+1);out+=part;cut|=flag
  return out,cut
 return [],False
rows=[[] for _ in targets];decoder=json.JSONDecoder(parse_float=Decimal);chunk=1048576
with path.open('r',encoding='utf-8') as f:
 buf=f.read(chunk);pos=0;started=False;eof=False
 while True:
  while pos<len(buf) and (buf[pos].isspace() or buf[pos]==','):pos+=1
  if not started:
   if pos>=len(buf) or buf[pos]!='[':raise RuntimeError('invalid trace format')
   pos+=1;started=True;continue
  if pos>=len(buf):
   if eof:raise RuntimeError('unexpected EOF')
   buf=f.read(chunk);pos=0;eof=not buf;continue
  if buf[pos]==']':break
  try:obj,end=decoder.raw_decode(buf,pos)
  except json.JSONDecodeError:
   rem=buf[pos:];more=f.read(chunk)
   if not more or len(rem)+len(more)>4194304:raise RuntimeError('invalid trace or record too large')
   buf=rem+more;pos=0;continue
  pos=end
  if not isinstance(obj,dict) or obj.get('ph')!='X' or not isinstance(obj.get('args'),dict):continue
  args=obj['args']
  try:
   ts=Decimal(str(obj['ts']))*1000
   if ts!=int(ts):continue
   key=(int(ts),obj.get('pid'),obj.get('tid'),args.get('Model Id'),args.get('Task Id'),hashlib.sha256(str(obj.get('name')).encode()).hexdigest())
  except (InvalidOperation,TypeError,ValueError,OverflowError):continue
  i=lookup.get(key)
  if i is None:continue
  info={'key_count':len(args),'key_list_truncated':len(args)>80,'keys':[],'candidate_id_fields':[]}
  for name,value in list(args.items())[:80]:
   if not isinstance(name,str):info['keys'].append({'name':None,'name_hmac':digest('key:',repr(name))});continue
   info['keys'].append({'name':name if name in safe else None,'name_hmac':None if name in safe else digest('key:',name)})
   norm=re.sub(r'[^a-z0-9]','',name.lower())
   if any(x in norm for x in ('addr','pointer','ptr','tensor','storage','buffer')):
    vals,cut=atoms(value)
    cut |= len(vals)>64
    vals=vals[:64]
    info['candidate_id_fields'].append({'key_name':name if name in safe else None,'key_hmac':None if name in safe else digest('key:',name),'numeric_count':len(vals),'numeric_hmacs':[digest('numeric:',v) for v in vals],'truncated':cut})
  rows[i].append(info)
  if len(rows[i])>1:raise RuntimeError('nonunique exact target')
after=path.stat()
if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or sha()!=expected:raise RuntimeError('trace changed during read')
result=[{'run':t[0],'kind':t[1],'start_ns':t[2],'pid':t[3],'tid':t[4],'model':t[5],'task':t[6],'match_count':len(rows[i]),'args':rows[i][0] if rows[i] else None} for i,t in enumerate(targets)]
print(json.dumps({'scope':'exact 4 gather + 4 candidate MatMul X events; no neighbors','trace_sha256':expected,'samples':result,'limitations':['The trace SHA was first observed in stage121, not independently frozen at capture.','Salt is freshly generated and discarded; HMAC equality is meaningful only within this output, not reproducible across runs.','Identity fields are heuristic key categories; equal identifiers do not prove tensor identity, version, or producer-consumer dependency.','Unknown key names are HMAC only, raw values are never returned. Missing fields and absence in one trace do not prove no runtime relationship.','No device calls, prompt, token, tensor payload, or raw addresses.']},ensure_ascii=False,indent=2))
