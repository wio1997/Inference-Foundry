#!/usr/bin/env python3
"""Recheck frozen layer0 elapsed intervals only; never accesses devices."""
import contextlib
import csv
import hashlib
import io
import json
import runpy
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / '40layer-mc2-dp-comparison.json'
SOURCE = HERE / 'recheck_40layer_mc2_dp.py'
EXPECTED = {
    REFERENCE: 'cf0962177d27f703143704a44e9030835aa1d4c86a19c35bcf12236c14fe9f3b',
    SOURCE: 'f46c8ca1413c01d4ec7340b5dea04f45ab77a72cf8c4995249b53a01fdc619b7',
}
for path, digest in EXPECTED.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
reference = json.loads(REFERENCE.read_text())
with contextlib.redirect_stdout(io.StringIO()):
    data = runpy.run_path(str(SOURCE))
for key, digest in reference['inputs_sha256'].items():
    assert data['sha'](data['paths'][key]) == digest, key
calls = {(c['rank'], c['run_in_model']): c for c in data['calls'] if c['layer'] == 0}
assert len(calls) == 48 and len(data['timeline']) == 48
samples = []
seen = set()
for row in data['timeline']:
    rank, run = row['rank'], int(row['captured_run_in_model'])
    assert (rank, run) not in seen
    seen.add((rank, run))
    call = calls[(rank, run)]
    start, end = call['window_start_ns'], call['window_end_ns']
    mc2_start, mc2_end = int(row['mc2_start_ns']), int(row['mc2_end_ns'])
    task = data['lookup'][(rank, int(row['target_forward_ordinal']) * 40)]
    assert task['source_layer'] == '0'
    assert int(task['start_ns']) == mc2_start and int(task['end_ns']) == mc2_end
    assert int(row['layer_start_ns']) == start and call['window_ns'] == end - start
    assert start <= mc2_start <= mc2_end <= end
    assert row['dp'] == rank.split('_')[0]
    fields = {'before_ns': mc2_start-start, 'mc2_ns': mc2_end-mc2_start,
              'after_ns': end-mc2_end, 'window_ns': end-start}
    assert fields['before_ns']+fields['mc2_ns']+fields['after_ns'] == fields['window_ns']
    samples.append({'rank': rank, 'dp': row['dp'], 'run': run, **fields})
assert seen == set(calls)
summary=[]
for run in (1, 2, 3):
    groups={}
    for dp in ('dp0', 'dp1'):
        rows=[s for s in samples if s['run']==run and s['dp']==dp]
        assert len(rows)==8
        stats={}
        for k in ('before_ns','mc2_ns','after_ns','window_ns'):
            vals=[r[k] for r in rows]
            stats[k]={'min_ns': min(vals), 'median_ns': statistics.median(vals), 'max_ns': max(vals)}
        groups[dp]={'count':len(rows),'stats':stats}
    summary.append({'run':run,'groups':groups})
print(json.dumps({'inputs_sha256': {p.name:digest for p,digest in EXPECTED.items()},
 'scope': '48 source-attributed layer0 windows; separate per-rank elapsed intervals in integer nanoseconds',
 'samples':samples,'summary':summary,
 'limitations':['same ordinal does not establish same request, collective, workload or cross-device synchronization',
 'pre/after intervals are not necessarily busy, idle, communication or causal critical path',
 'segment medians cannot be added to infer whole-window median',
 'samples are not independent experiments; no cause, removable time or general performance bound inferred'],
 'no_device_work':True},ensure_ascii=False,indent=2))
