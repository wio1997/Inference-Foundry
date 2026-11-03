#!/usr/bin/env python3
"""Aggregate existing layer-scoped unique device matches; never executes device work."""
import collections, hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
source = root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
out = root / 'agents/fusion/f22-all-rank-device-chain.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
counts=collections.Counter(); patterns=collections.Counter(); ranks={}; examples={}; footer=None
first=lambda s:s.strip('"').split(';')[0]
for line in source.open():
    row=json.loads(line)
    if row['kind']=='footer': footer=row
    if row['kind']!='rank': continue
    rank=row['data']; ranks[str(rank['ep'])]={'status':rank['status'],'layer_scopes':len(rank['layers']),'selected_start_events':rank['counts']['target_selected_start_events'],'csv_missing':rank['counts']['csv_missing']}
    for layer in rank['layers']:
        counts['scopes']+=1; streams=collections.defaultdict(list)
        for group in layer['operator_groups']:
            for task in group['tasks']:
                c=task['csv']; streams[c['Stream ID']].append((float(task['device_start_us']),group['operator_name'],c,task['flow_id']))
        targets=[]
        for stream, events in streams.items():
            events.sort(key=lambda v:(v[0],v[3]))
            for i,e in enumerate(events):
                if e[1]!='DequantSwigluQuant': continue
                targets.append((stream,events,i,e))
        counts['activation_records']+=len(targets)
        if len(targets)!=1: counts['scopes_not_one_activation']+=1; continue
        stream,ev,i,e=targets[0]
        if i<2 or i+2>=len(ev): counts['insufficient_matched_neighbors']+=1; continue
        seq=ev[i-2:i+3]; names=tuple(x[1] for x in seq);patterns[' -> '.join(names)]+=1
        pre,act,post=ev[i-1][2],e[2],ev[i+1][2]
        checks=(first(pre['Output Shapes'])==first(act['Input Shapes']),first(act['Output Shapes'])==first(post['Input Shapes']),first(pre['Output Data Types'])==first(act['Input Data Types']),first(act['Output Data Types'])==first(post['Input Data Types']))
        if all(checks): counts['triple_shape_dtype_compatible']+=1
        else: counts['triple_shape_dtype_incompatible']+=1
        if 'example' not in examples:
            examples['example']={'ep':layer['ep'],'layer':layer['layer'],'scope':layer['layer_scope_id'],'scheduler_digest':layer['scheduler_digest'],'stream':stream,'events':[{'name':x[1],'flow_id':x[3],'input_shapes':x[2]['Input Shapes'],'output_shapes':x[2]['Output Shapes'],'input_types':x[2]['Input Data Types'],'output_types':x[2]['Output Data Types']} for x in seq]}
assert footer is not None
result={'source':str(source.relative_to(root)),'source_sha256':sha(source),'script_sha256':sha(Path(__file__)),'scope':'unique matched device tasks in each previously identified same-thread target layer scope, sorted by exported device start time within one stream; neighboring matched records do not exclude unrecorded events, prove a tensor edge, or identify shared experts','counts':dict(counts),'five_name_patterns':dict(patterns),'rank_coverage':ranks,'footer':footer,'examples':examples,'limitations':['Heavy-profiler prefill sample only, not complete request or online throughput','All task records are unique matched exports; selected start events can lack CSV or device matches','Same stream and equal physical shape and dtype do not establish tensor identity, producer-consumer dependency, or existing compiler implementation','Kernel name and boundaries do not prove numerical semantics or hardware instruction fusion','Cannot extrapolate duration, critical path, or benefit']}
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts':result['counts'],'patterns':result['five_name_patterns'],'footer':footer,'output_sha256':sha(out)},ensure_ascii=False,indent=2))
