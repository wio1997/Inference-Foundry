#!/usr/bin/env python3
"""Relate two already-exported BF16 tasks to fixed same-rank target brackets.

CPU only; inputs are frozen local aggregates, no profiler or device access.
"""
import collections
import hashlib
import json
import statistics
from pathlib import Path

T = Path(__file__).resolve().parents[2]
W = T / 'captured-layer-attribution/module-call-metrics.json'
M = T / 'agents/diagnosis/16rank-engram-matmul-metrics.json'
expected_w = 'd98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a'
expected_m = '35ff1b007a471903324d96de5eeb67b80a13251a4c4dadd8619eead77c23db1d'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(W) == expected_w and sha(M) == expected_m
windows = collections.defaultdict(list)
for x in json.loads(W.read_text()):
    if x['run_in_model'] in (1, 2, 3):
        windows[(x['rank'], x['run_in_model'])].append(x)
tasks = collections.defaultdict(list)
for x in json.loads(M.read_text())['samples']:
    a = x['target']; tasks[(a['rank'], a['run'])].append(a)
assert set(windows) == set(tasks) and len(windows) == 48
rows = []
for key, group in sorted(windows.items()):
    group.sort(key=lambda x: x['layer'])
    jobs = sorted(tasks[key], key=lambda x: x['start_ns'])
    assert [x['layer'] for x in group] == list(range(40))
    assert [x['after_layer'] for x in jobs] == [0, 13]
    assert len({str(x['main_stream_id']) for x in group}) == 1
    assert len({str(x['model_id']) for x in group}) == 1
    assert all(a['window_end_ns'] <= b['window_start_ns'] for a,b in zip(group,group[1:]))
    assert jobs[0]['end_ns'] <= jobs[1]['start_ns']
    for x in jobs:
        assert str(x['model']) == str(group[0]['model_id'])
        assert str(x['stream']) == str(group[0]['main_stream_id'])
        assert group[x['after_layer']]['window_end_ns'] <= x['start_ns'] < x['end_ns'] <= group[x['after_layer']+1]['window_start_ns']
        assert not any(max(x['start_ns'],g['window_start_ns']) < min(x['end_ns'],g['window_end_ns']) for g in group)
    span = group[-1]['window_end_ns'] - group[0]['window_start_ns']
    matmul = sum(x['end_ns']-x['start_ns'] for x in jobs)
    rows.append({'rank':key[0], 'run':key[1], 'target_span_ns':span,
                 'two_matmul_duration_ns':matmul, 'fraction':matmul/span})
summary = {}
for k in ('target_span_ns', 'two_matmul_duration_ns', 'fraction'):
    vals=[x[k] for x in rows]
    summary[k]={'min':min(vals), 'median':statistics.median(vals), 'max':max(vals)}
out = {'input_sha256': {str(W.relative_to(T)):sha(W),str(M.relative_to(T)):sha(M)},
       'scope':'16 rank x 3 fixed target model-49 samples; same rank/run/stream',
       'checks':'40 ordered non-overlapping layer brackets, two matching matmul tasks in 0->1 and 13->14 gaps; neither intersects any layer bracket',
       'limitations':['brackets inferred from source sequence, not official full modules', 'not a full request or end-to-end time', 'other streams may overlap; no critical-path or actual savings claim', 'shape-compatible Engram candidate only; tensor/weight identity not established'],
       'summary':summary,'samples':rows}
print(json.dumps(out,ensure_ascii=False,indent=2))
