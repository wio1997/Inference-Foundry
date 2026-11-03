#!/usr/bin/env python3
"""Read existing per-layer profiler aggregate; compare exported physical dimensions, not tensors."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
OUT = ROOT / 'agents/fusion/f22-recapture-physical-row-domains.json'
hash_src = sha256()
with SRC.open('rb') as f:
    for chunk in iter(lambda: f.read(1024*1024), b''):
        hash_src.update(chunk)
counts=Counter(); groups=defaultdict(Counter); examples={}; eps=set(); footer=None
with SRC.open() as f:
    for line in f:
        row=json.loads(line)
        if row.get('kind')=='footer': footer=row; continue
        if row.get('kind')!='rank': continue
        rank=row['data']; eps.add(rank['ep'])
        for layer in rank['layers']:
            counts['scopes']+=1
            def select(name): return [x for x in layer['operator_groups'] if x['operator_name']==name]
            m=select('DispatchFFNCombine'); d=select('DequantSwigluQuant')
            if len(m)!=1 or len(d)!=1 or m[0]['task_count']!=1 or d[0]['task_count']!=1:
                counts['excluded_nonunique']+=1; continue
            def first_shape(g):
                return g['physical_shape_dtype_format']['Input Shapes'].strip('"').split(';')[0]
            ms=first_shape(m[0]); ds=first_shape(d[0]); mr=int(ms.split(',')[0]); dr=int(ds.split(',')[0])
            typ='equal' if mr==dr else ('mc2_larger' if mr>dr else 'mc2_smaller')
            counts[typ]+=1
            key=(layer['scheduler_digest'],ms,ds)
            groups[key][rank['ep']]+=1
            examples.setdefault(typ,{'ep':rank['ep'],'layer':layer['layer'],'scheduler_digest':layer['scheduler_digest'],
              'mc2_input_shape':ms,'activation_input_shape':ds,'mc2_flow_id':m[0]['tasks'][0]['flow_id'],
              'activation_flow_id':d[0]['tasks'][0]['flow_id']})
result={'source':str(SRC.relative_to(ROOT)),'source_sha256':hash_src.hexdigest(),
        'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'definition':'one uniquely matched task per name inside the same target layer CPU submission scope; physical first dimensions only; not a tensor identity or producer-consumer proof',
        'counts':dict(counts),'rank_count':len(eps),'examples':examples,
        'groups':[{'scheduler_digest':k[0],'mc2_input_shape':k[1], 'activation_input_shape':k[2],
                   'per_ep_scope_count':dict(sorted(c.items()))} for k,c in sorted(groups.items())],
        'footer':footer}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts':dict(counts),'rank_count':len(eps),'output':str(OUT.relative_to(ROOT))},ensure_ascii=False))
