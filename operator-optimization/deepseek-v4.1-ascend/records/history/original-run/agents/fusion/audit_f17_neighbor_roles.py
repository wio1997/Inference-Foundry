#!/usr/bin/env python3
"""Check same-track nearest exported tasks in selected F17 scopes, CPU only."""
import collections, hashlib, json
from decimal import Decimal as D
from pathlib import Path

root=Path(__file__).resolve().parents[2]
source=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
signatures=collections.defaultdict(collections.Counter); counts=collections.Counter(); seen=set()
for line in source.open():
    item=json.loads(line)
    if item.get('kind')!='rank': continue
    for scope in item['data']['layers']:
        if scope['layer'] not in (2,8,14): continue
        key=(scope['ep'],scope['layer'],scope['scheduler_digest']);assert key not in seen;seen.add(key)
        tasks=[t for g in scope['operator_groups'] for t in g['tasks']]; choices=collections.defaultdict(list)
        for task in tasks:
            c=task['csv']; inp=c['Input Shapes'].strip('"[]');m=inp.split(',')[0]
            if c['Type']=='MatMulV2' and m in ('4760','5144','8180') and inp==f'{m},5120;512,5120' and c['Output Shapes'].strip('"[]')==f'{m},512':
                choices[(int(m),c['Input Data Types'],c['Output Data Types'])].append(task)
        matches=[]
        for m in (4760,5144,8180):
            bf=choices.get((m,'DT_BF16;DT_BF16','DT_BF16'),[]);fp=choices.get((m,'FLOAT;FLOAT','FLOAT'),[])
            if bf or fp:
                assert len(bf)==1 and len(fp)==2
                matches.append((m,bf[0],sorted(fp,key=lambda t:D(t['device_start_us']))))
        assert len(matches)==1
        m,bf,fp=matches[0]
        def neighbors(task,before):
            start=D(task['device_start_us']);end=start+D(task['trace_duration_us'])
            selected=[u for u in tasks if (u['device_track_pid'],u['device_track_tid'])==(task['device_track_pid'],task['device_track_tid']) and (D(u['device_start_us'])+D(u['trace_duration_us'])<=start if before else D(u['device_start_us'])>=end)]
            return sorted(selected,key=lambda x:D(x['device_start_us']),reverse=before)
        pre=neighbors(fp[0],True);after=neighbors(fp[1],False);afterbf=neighbors(bf,False)
        assert pre and len(after)>=3 and len(afterbf)>=3
        selected=[('before_float',pre[0]),*[(f'after_float_{i}',x) for i,x in enumerate(after[:3],1)],*[(f'after_bf16_{i}',x) for i,x in enumerate(afterbf[:3],1)]]
        expected=['Cast','ZerosLike','_pool_kernel','_cache_update_kernel','RmsNorm','InplacePartialRotaryMul','ScatterNdUpdateSk']
        assert [x['csv']['Type'] for _,x in selected]==expected,(key,m,[x['csv']['Type'] for _,x in selected])
        for role,t in selected:
            c=t['csv'];signature=tuple(c[k] for k in ('Type','Input Shapes','Input Data Types','Output Shapes','Output Data Types'))
            signatures[role][signature]+=1
        counts[m]+=1
assert len(seen)==96 and dict(counts)=={4760:24,5144:24,8180:48}
for role, sig in signatures.items():
    assert sum(sig.values())==96
print(json.dumps({'scope_count':len(seen),'row_groups':counts,'neighbor_type_count':{k:sum(v.values()) for k,v in signatures.items()},'signatures':{k:[{'fields':x,'count':n} for x,n in v.items()] for k,v in signatures.items()},'limitations':'Nearest among selected same-track tasks only; no full-stream completeness, storage identity, causal edge or historical source binding.'},indent=2))
