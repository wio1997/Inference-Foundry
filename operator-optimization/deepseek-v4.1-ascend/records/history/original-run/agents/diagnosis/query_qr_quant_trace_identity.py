#!/usr/bin/env python3
"""Read-only fixed EP8 trace sample; inspect exact device events and flow identity metadata."""
import sys,json,hashlib,collections,decimal
import ijson
from pathlib import Path
p=Path(sys.argv[1]); hasher=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):hasher.update(b)
assert hasher.hexdigest() == '3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99', 'trace hash mismatch'
lo=decimal.Decimal(1793152960208017846)/1000
hi=decimal.Decimal(1793152960209362219)/1000
expected={'1261':lo+decimal.Decimal(83199)/1000,'1264':lo+decimal.Decimal(92559)/1000,'1267':lo+decimal.Decimal(149557)/1000,'1268':lo+decimal.Decimal(153577)/1000}
events={}
with p.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict) or e.get('ph')!='X':continue
  a=e.get('args') or {}; task=str(a.get('Task Id'))
  if task not in expected or str(a.get('Model Id'))!='49' or str(a.get('Physic Stream Id'))!='89':continue
  try: ts=decimal.Decimal(str(e.get('ts')))
  except decimal.InvalidOperation:continue
  if ts!=expected[task]:continue
  events.setdefault(task,[]).append({'pid':str(e.get('pid')),'tid':str(e.get('tid')),'ts':str(ts),'name':e.get('name'),'arg_keys':sorted(a),'args_with_identity_key':[k for k in sorted(a) if any(s in k.lower() for s in ('addr','tensor','storage','input','output'))]})
assert all(len(events.get(k,[]))==1 for k in expected),events
keys={(v[0]['pid'],v[0]['tid'],v[0]['ts']):k for k,v in events.items()}; flows=collections.defaultdict(list)
with p.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict) or e.get('ph') not in ('s','f'):continue
  key=(str(e.get('pid')),str(e.get('tid')),str(e.get('ts')))
  if key in keys: flows[keys[key]].append({'phase':e.get('ph'),'name':e.get('name'),'cat':e.get('cat'),'id':str(e.get('id'))})
ids={e['id'] for rows in flows.values() for e in rows}; associated=collections.defaultdict(collections.Counter)
with p.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict) or e.get('ph') not in ('s','f') or str(e.get('id')) not in ids:continue
  associated[str(e.get('id'))][(str(e.get('ph')),str(e.get('name')),str(e.get('cat')),str(e.get('pid')),str(e.get('tid')))]+=1
print(json.dumps({'scope':'EP8 layer24 run2 exact model49 stream89 task1261/1264/1267/1268 only; flow matches use pid/tid/decimal-string timestamp equality','trace_sha256':hasher.hexdigest(),'events':events,'flow_hits':flows,'flow_events_by_id':{k:[{'phase':a,'name':b,'cat':c,'pid':d,'tid':e,'count':n} for (a,b,c,d,e),n in sorted(v.items())] for k,v in associated.items()},'limits':['Flow IDs are not tensor or storage identities.','Only matched event fields and exact endpoints inspected; does not prove absence of metadata elsewhere.','No host-device duration comparisons or NPU work.']},indent=2))
