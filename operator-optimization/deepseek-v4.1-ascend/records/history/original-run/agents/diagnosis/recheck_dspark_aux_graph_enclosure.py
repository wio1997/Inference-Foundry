#!/usr/bin/env python3
"""Recheck selected historical ReduceMean containment using local snapshots only."""
import collections, hashlib, json
from pathlib import Path
T = Path(__file__).resolve().parents[2]
paths = {
    'windows': ('captured-layer-attribution/module-call-metrics.json', 'd98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a'),
    'calls': ('model-execute-wait-chain-v2/calls.json', 'ecb1ba603e52cf3cf6722f4b8cb0c1aad3b295c596f4f60c7fc047ecff6ef925'),
    'samples': ('agents/diagnosis/dspark-aux-boundaries.json', '13b5584a6b2e93ddf7a7b19da37364b1212e54a875c7a1720ceb80831af0655e'),
}
inputs = {}
for key, (relative, expected) in paths.items():
    data = (T / relative).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, relative
    inputs[key] = json.loads(data)
windows = {}
for w in inputs['windows']:
    key = (w['rank'], w['run_in_model'], w['layer'])
    assert key not in windows
    windows[key] = w
calls_by_rank = collections.defaultdict(list)
for c in inputs['calls']:
    calls_by_rank[c['rank']].append(c)
counts = collections.Counter()
results = []
for sample in inputs['samples']['samples']:
    if sample['after_layer'] not in (36, 37, 38):
        continue
    key = (sample['rank'], sample['run'], sample['after_layer'])
    w = windows[key]
    assert str(w['model_id']) == '49' and str(w['main_stream_id']) == sample['main_stream']
    start = w['window_end_ns']
    matching = [e for e in sample['events'] if e['type'] == 'ReduceMean' and e['model'] == '49' and e['stream'] == sample['main_stream']]
    assert len(matching) == 1
    event = matching[0]
    assert 0 <= event['event_start_rel_ns'] < event['event_end_rel_ns'] <= sample['gap_ns']
    lo, hi = start + event['event_start_rel_ns'], start + event['event_end_rel_ns']
    parents = []
    for call in calls_by_rank[sample['rank']]:
        waits = [task for task in call['tasks'] if task[5] == 'MODEL_WAIT_COMPLETE' and task[2] == 49 and task[0] <= lo and hi <= task[1]]
        if waits:
            assert len(waits) == 1
            parents.append((call, waits[0]))
    assert len(parents) == 1
    parent, wait = parents[0]
    assert parent['model_id'] == 49 and parent['one_execute_one_wait'] and len(parent['cann_execute_api']) == 1
    counts[(sample['run'], parent['ordinal'])] += 1
    results.append({'rank': sample['rank'], 'run': sample['run'], 'after_layer': sample['after_layer'],
                    'ordinal': parent['ordinal'], 'connection_id': parent['connection_id'],
                    'task_start_ns': lo, 'task_end_ns': hi, 'parent_start_ns': wait[0], 'parent_end_ns': wait[1]})
assert len(results) == 144 and len({r['rank'] for r in results}) == 16
assert counts == {(1, 1): 48, (2, 6): 48, (3, 7): 48}
print(json.dumps({'input_sha256': {k: v[1] for k, v in paths.items()}, 'rows': results,
                  'counts_by_run_ordinal': [{'run': r, 'ordinal': o, 'count': n} for (r, o), n in sorted(counts.items())],
                  'limitations': ['Time containment within a same-rank model wait interval is graph-level association, not node provenance, tensor identity, request identity or a pure wait interval.']}, ensure_ascii=False, indent=2))
