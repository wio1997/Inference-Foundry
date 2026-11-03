#!/usr/bin/env python3
"""Check exact four saved anchors and closest same-thread select/contiguous metadata."""
import hashlib,json,pathlib
from collections import Counter
p=pathlib.Path(__file__).resolve().parent
prov=p/'copy-provenance-audit.json';raw=p/'engram-copy-producers.json'
a=json.loads(prov.read_text());b=json.loads(raw.read_text())
copy={}
for s in a['samples']:
 for corr in s['correlations']:
  for scope in corr['scopes']:
   for row in scope['scopes']:
    stack=row.get('callchain',[])
    if row['name']=='aten::copy_' and any('engram/embedding.py(342)' in (x.get('value') or '') for x in stack):
     key=int(row['startNs'])
     if key in copy:raise ValueError('duplicate copy anchor')
     copy[key]=row
if len(copy)!=4 or len(b['anchors'])!=4:raise ValueError('expected 4 anchors')
res=[]
for q in b['anchors']:
 t=q['start_ns'];rows=q['rows'];c=copy[t]
 current=[r for r in rows if r['name']=='aten::contiguous' and int(r['startNs'])<=t and int(r['endNs'])>=int(c['endNs'])]
 if len(current)!=1:raise ValueError('contiguous scope missing or ambiguous')
 cont=current[0]; sel=[r for r in rows if r['name']=='aten::select' and int(r['endNs'])<=int(cont['startNs'])]
 if not sel:raise ValueError('no preceding select')
 sel=max(sel,key=lambda r:int(r['endNs']))
 tdim=int(c['inputShapes'].split(',')[0]);want=f'{tdim},2,24'
 if sel['inputShapes'].split(';')[0].strip()!=want:raise ValueError('shape mismatch')
 if not any('model.py(1217): prepare_engram' in (v['value'] or '') for v in sel['callchain']):raise ValueError('select stack mismatch')
 res.append({'anchor':q['anchor'],'copy_start_ns':t,'copy_shape':c['inputShapes'].split(';')[0].strip(),'select_start_ns':int(sel['startNs']),'select_end_ns':int(sel['endNs']),'select_input_shape':want,'select_callchain_id':sel['callchainId'],'contiguous_start_ns':int(cont['startNs']),'contiguous_end_ns':int(cont['endNs']),'contiguous_callchain_id':cont['callchainId']})
if Counter(r['select_input_shape'] for r in res)!=Counter({'8184,2,24':2,'4888,2,24':2}):raise ValueError('unexpected scope')
out={'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (prov,raw)},'rows':res,'limits':['Temporal and same-stack links are not tensor identity; strides, offset, selected scalar, branch, stream and lifetime unknown.','Bounded profiler export may have omitted events; timings are host scopes and profiling-perturbed.']}
(p/'engram-copy-producers-recheck.json').write_text(json.dumps(out,indent=2)+'\n')
print('4 exact copy anchors matched; 4 select shapes agree; tensor identity and strides remain unknown')
