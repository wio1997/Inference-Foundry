#!/usr/bin/env python3
"""Read saved CSV only; compare disjoint target windows per rank/replay."""
import argparse, bisect, csv, hashlib, json, statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--task-root', required=True)
ap.add_argument('--trace-root', required=True)
a = ap.parse_args()
wp = Path(a.task_root) / 'captured-layer-attribution/module-call-metrics.json'
windows = json.loads(wp.read_text())
by = defaultdict(list)
for w in windows:
    if w['run_in_model'] in (1, 2, 3):
        by[(w['rank'], str(w['model_id']), str(w['main_stream_id']))].append(w)
for arr in by.values():
    arr.sort(key=lambda w: w['window_start_ns'])
    assert all(x['window_end_ns'] <= y['window_start_ns'] for x, y in zip(arr, arr[1:]))
assert sum(map(len, by.values())) == 1920 and len({k[0] for k in by}) == 16
units = defaultdict(lambda: {'windows': [], 'events': []})
for key, arr in by.items():
    for w in arr:
        units[(key[0], w['run_in_model'])]['windows'].append(w)
for u in units.values():
    assert len(u['windows']) == 40 and {w['layer'] for w in u['windows']} == set(range(40))
hashes = {}
types = {'VllmAscendQuantLightningIndexerV2', 'DispatchFFNCombine'}
for rp in sorted(Path(a.trace_root).glob('*_ascend_pt')):
    if rp.name not in {k[0] for k in by}: continue
    p = rp / 'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
    hashes[rp.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open() as f:
        for z in csv.DictReader(f):
            if z['Type'] not in types: continue
            key = (rp.name, z['Model ID'], z['Stream ID'])
            arr = by.get(key, [])
            if not arr: continue
            start = int(Decimal(z['Start Time(us)'].strip()) * 1000)
            duration = int(Decimal(z['Duration(us)']) * 1000)
            end = start + duration
            i = bisect.bisect_right([w['window_start_ns'] for w in arr], start) - 1
            if i < 0 or not (arr[i]['window_start_ns'] <= start and end <= arr[i]['window_end_ns']): continue
            w = arr[i]
            units[(rp.name, w['run_in_model'])]['events'].append({'type': z['Type'], 'layer': w['layer'], 'start_ns': start, 'end_ns': end, 'duration_ns': duration})
assert len(hashes) == 16 and len(units) == 48
rows = []
for (rank, run), u in sorted(units.items()):
    events = sorted(u['events'], key=lambda e: e['start_ns'])
    assert all(x['end_ns'] <= y['start_ns'] for x, y in zip(events, events[1:]))
    ind = [e for e in events if e['type'] == 'VllmAscendQuantLightningIndexerV2']
    mc2 = [e for e in events if e['type'] == 'DispatchFFNCombine']
    assert len(ind) == 8 and {e['layer'] for e in ind} == {2, 8, 14, 20, 24, 28, 32, 36}
    assert len(mc2) == 40 and {e['layer'] for e in mc2} == set(range(40))
    den = sum(w['window_ns'] for w in u['windows'])
    ins = sum(e['duration_ns'] for e in ind)
    hot = sum(e['duration_ns'] for e in ind if e['layer'] in (24, 28, 32, 36))
    ms = sum(e['duration_ns'] for e in mc2)
    rows.append({'rank': rank, 'run': run, 'window_us': den / 1000, 'indexer_us': ins / 1000,
                 'indexer_hot_four_us': hot / 1000, 'mc2_us': ms / 1000,
                 'indexer_fraction': ins / den, 'indexer_hot_four_fraction': hot / den, 'mc2_fraction': ms / den})
summary = []
for run in (1, 2, 3):
    group = [x for x in rows if x['run'] == run]
    assert len(group) == 16
    o = {'run': run}
    for field in ('window_us', 'indexer_us', 'indexer_hot_four_us', 'mc2_us', 'indexer_fraction', 'indexer_hot_four_fraction', 'mc2_fraction'):
        vals = [x[field] for x in group]
        o[field] = {'min': min(vals), 'median': statistics.median(vals), 'max': max(vals)}
    summary.append(o)
out = {'scope': '16 ranks x 3 already source-attributed target interior samples; 40 disjoint per-rank main-stream layer windows',
       'window_sha256': hashlib.sha256(wp.read_bytes()).hexdigest(), 'kernel_sha256_by_rank': hashes,
       'event_types': sorted(types), 'rank_run_rows': rows, 'run_summary': summary,
       'limitations': ['Each ratio is calculated per rank and sample before aggregate; it is not a proportion of end-to-end request time.',
          'The two types are main-stream device tasks; they may overlap with tasks on other streams, ranks, or host.',
          'Counts are trace occurrence counts, not request frequency; profiler overhead and dynamic lengths are unknown.',
          'Task durations or proportions do not identify causes, critical-path exposure, or possible acceleration.']}
print(json.dumps(out, ensure_ascii=False, indent=2))
