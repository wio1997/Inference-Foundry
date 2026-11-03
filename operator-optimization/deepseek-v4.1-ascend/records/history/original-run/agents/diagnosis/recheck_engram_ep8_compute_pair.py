#!/usr/bin/env python3
"""Compare already-exported EP8 matched compute fields without retrieving data."""
import collections,hashlib,json
from pathlib import Path
T=Path(__file__).resolve().parents[2]
def read(name,expected):
    data=(T/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected;return json.loads(data)
x=read('agents/diagnosis/engram-matmul-task-links.json','3e6196f3005366117421c78a359dc036e9d79f7299f2b72d239c2401202e76d1')
y=read('agents/diagnosis/16rank-engram-matmul-metrics.json','35ff1b007a471903324d96de5eeb67b80a13251a4c4dadd8619eead77c23db1d')
assert x['db_sha256']=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
ranks={r['target']['rank'] for r in y['samples'] if '_ep8_' in r['target']['rank']};assert len(ranks)==1
rank=next(iter(ranks)); pairs=collections.defaultdict(dict)
for sample in x['samples']:
    t=sample['target']; key=(t['run'],t['after_layer']);assert key[1] in (0,13) and key[1] not in pairs[key[0]]
    raw=sample['exact_task'];found=[v for v in y['samples'] if (v['target']['rank'],v['target']['run'],v['target']['after_layer'])==(rank,*key)]
    assert len(found)==1; anchor=found[0]['target'];assert (raw['startNs'],raw['endNs'],raw['modelId'],raw['streamId'])==(anchor['start_ns'],anchor['end_ns'],int(anchor['model']),int(anchor['stream']))
    pairs[key[0]][key[1]]=sample
assert len(pairs)==3 and all(set(v)=={0,13} for v in pairs.values())
results=[]
for run,pair in sorted(pairs.items()):
    a,b=pair[0],pair[13]; first=a['compute_fields_raw'];second=b['compute_fields_raw']
    assert first['globalTaskId']==18536 and second['globalTaskId']==19042
    raw_keys=set(first)|set(second);diff=[key for key in sorted(raw_keys) if key!='globalTaskId' and first.get(key)!=second.get(key)]
    decoded_keys=set(a['compute_decoded_fields'])|set(b['compute_decoded_fields']);dd=[k for k in sorted(decoded_keys) if a['compute_decoded_fields'].get(k)!=b['compute_decoded_fields'].get(k)]
    results.append({'run':run,'task_ids':[a['exact_task']['taskId'],b['exact_task']['taskId']],
                    'global_task_ids':[18536,19042], 'raw_keys_compared_except_globalTaskId':len(raw_keys)-1,
                    'different_raw_keys':diff,'decoded_keys_compared':len(decoded_keys),'different_decoded_keys':dd,
                    'full_attr_equal':a['compute_decoded_fields']['attrInfo']==b['compute_decoded_fields']['attrInfo']})
print(json.dumps({'scope':'3 fixed EP8 rank/run paired events, profiler metadata only', 'results':results,
 'limitations':['Raw/decoded values are profiler metadata. Equal selected fields do not prove same tensor, values, weights, layout stride, binary, tiling, workload, request, or performance cause.',
 'Compute records are associated with globalTaskId reused in TASK, not unique replay metadata; this check is limited to one rank.',
 'No NPU or remote operations; no raw attribute string is output.']},ensure_ascii=False,indent=2))
