#!/usr/bin/env python3
"""Resolve saved API name hashes only against already-exported task names."""
from collections import Counter
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
api=root/'agents/diagnosis/draft-argmax-api-links-ep8.json'
shape=root/'agents/diagnosis/draft-argmax-intervals-shapes-ep8.json'
def load(p,expected):
 b=p.read_bytes()
 if hashlib.sha256(b).hexdigest()!=expected: raise RuntimeError('source SHA mismatch: '+p.name)
 return json.loads(b)
x=load(api,'8eab586ca22e79ddce8b88a5b304ba54d0f18a4aca840f0979e584b9e6e01007')
y=load(shape,'59930fe7433cb31708f906713b8f24685dd307cb43ad1ab7a0b2d35310766dfe')
names={row['match']['fields']['Name'].split('_',1)[0] for row in y['rows'] if row['matches']==1 and row['match']['fields']['Name'].startswith('aclnn')}
d={hashlib.sha256(n.encode()).hexdigest():n for n in names}; counts=Counter(); event_classes=Counter()
for r in x['rows']:
 for link in r['links']:
  for group in link['pytorch_via_sets']:
   for c in group['candidates']:
    h=c['other_name_sha256']; name=d.get(h)
    counts[name if name else 'unmatched']+=1;event_classes[(r['target']['type'],name if name else 'unmatched')]+=1
print(json.dumps({'candidate_prefixes':sorted(names),'counts':dict(counts),'task_types':[{'type':a,'name':b,'count':n} for (a,b),n in sorted(event_classes.items())],'limitations':['Exact hash equality to an already exported name is a name match, not a Python callstack, request, tensor, or causal link.','Unmatched is not a negative proof of a specific source path.']},ensure_ascii=False,indent=2))
