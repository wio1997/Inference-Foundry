"""只用标准库复算已发布的时间戳与计数，不执行模型或NPU任务。"""
from pathlib import Path
import collections
import json
import math
import statistics

B=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (B/'request-timings-138-140.jsonl').read_text().splitlines()]
groups=collections.defaultdict(list)
for row in rows:
    groups[row['generation'],row['concurrency']].append(row)
assert len(rows)==192 and len(groups)==6
metrics={}
for (g,c),items in sorted(groups.items()):
    assert len(items)==32 and len({x['data_id'] for x in items})==32
    assert all(x['success'] and x['input_tokens']==1024 and x['output_tokens']==256 for x in items)
    ts=[x['time_points'] for x in items]
    assert all(len(t)>=3 and all(map(math.isfinite,t)) and all(a<=b for a,b in zip(t,t[1:])) for t in ts)
    duration=max(t[-1] for t in ts)-min(t[0] for t in ts)
    metrics[f'{g}/C{c}']={'requests':32,'output_tokens':8192,'benchmark_seconds':duration,
        'output_token_throughput':8192/duration,
        'ttft_median_ms':statistics.median((t[1]-t[0])*1000 for t in ts),
        'tpot_median_ms':statistics.median((t[-1]-t[1])*1000/255 for t in ts)}
expected=json.loads((B/'measured-results-138-140.json').read_text())
for key,value in metrics.items():
    for name in ('benchmark_seconds','output_token_throughput','ttft_median_ms','tpot_median_ms'):
        assert abs(value[name]-expected[key][name])<1e-8,(key,name,value[name],expected[key][name])
inputs=json.loads((B/'counter-inputs-138-140.json').read_text())
counter_groups=collections.defaultdict(list)
for row in inputs['ranks']:
    delta={k:row['closed_counts'].get(k,0)-row['startup_counts'].get(k,0)
           for k in set(row['closed_counts'])|set(row['startup_counts'])}
    assert all(type(v) is int and v>=0 for v in delta.values())
    counter_groups[row['generation']].append(delta)
for g,rank_counts in counter_groups.items():
    assert len(rank_counts)==8 and all(q==rank_counts[0] for q in rank_counts)
print(json.dumps({'status':'published_timestamp_and_counter_evidence_recomputed',
    'metrics':metrics,'rank_counts_after_startup':{str(g):q[0] for g,q in counter_groups.items()},
    'scope':'single batches;whole-run counters include engineering/warmups;not device kernel counts or causal timing',
    'new_npu_work':False},ensure_ascii=False,indent=2))
