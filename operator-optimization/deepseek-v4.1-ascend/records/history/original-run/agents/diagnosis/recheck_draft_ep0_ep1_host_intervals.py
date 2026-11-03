#!/usr/bin/env python3
"""Compare same-table candidate endpoints; never compare clocks across ranks."""
import hashlib, json
from pathlib import Path
base=Path(__file__).resolve().parent
files={"ep0":("draft-ep0-mc2-host-links.json","6660ac83362cb6b8dc80b7cc845e3aeac21eb93a60c4f360590faedbf2b02ad2"),"ep1":("draft-mc2-gap-host-links.json","2b07456e853931f2764b494ccc5e315ad6ed9af5745cfa87f0b2da19940b157e")}
output={}
for rank,(name,expected) in files.items():
 b=(base/name).read_bytes()
 if hashlib.sha256(b).hexdigest()!=expected:raise RuntimeError('input SHA mismatch')
 d=json.loads(b);a={(x['opId_local'],x['role']):x for x in d['targets']}
 if set(a)!={(i,role) for i in (402,405,408) for role in ('gating','mc2')}:raise RuntimeError('unexpected anchors')
 out=[]
 for i in (402,405,408):
  g,m=a[i,'gating'],a[i,'mc2']; rec={'local_op_id':i}
  for table in ('cann_same_connection','pytorch_via_sets'):
   def rows(x):
    if table=='cann_same_connection':return x[table]
    return [y for z in x[table] for y in z['rows']]
   left,right=rows(g),rows(m)
   if len(left)!=1 or len(right)!=1 or left[0]['globalTid']!=right[0]['globalTid']:raise RuntimeError('candidate not unique same table/thread')
   delta=int(right[0]['startNs'])-int(left[0]['endNs'])
   if delta<0:raise RuntimeError('negative interval')
   rec['cann_us' if table=='cann_same_connection' else 'pytorch_us']=delta/1000
  out.append(rec)
 output[rank]=out
print(json.dumps({'input_sha256':{rank:sha for rank,(_,sha) in files.items()},'same_table_local_intervals':output,'scope':'Each rank and table independently; candidate interval is not CPU utilization, device interval or cause.'},ensure_ascii=False,indent=2))
