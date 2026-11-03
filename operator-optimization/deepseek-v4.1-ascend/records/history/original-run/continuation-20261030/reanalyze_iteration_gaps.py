#!/usr/bin/env python3
"""Offline union/overlap accounting between repeated main-model segments.

BOUNDARIES is supplied by the audited local graph-signature analysis runner.
"""
import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official-20261030T084726Z-cb252d1d/analysis/operator-export/copies')


def ns(x):
    v = Decimal(str(x).strip())*1000
    assert v.is_finite()
    return int(v)


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return sum(b-a for a, b in merged)


def clip(s, e, a, b):
    return (max(s, a), min(e, b)) if s < b and e > a else None


output = []
for spec in BOUNDARIES:
    folder = ROOT/spec['source_dir']/'ASCEND_PROFILER_OUTPUT'
    path = folder/'kernel_details.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['kernel_sha256']
    rr = []
    with path.open(newline='') as f:
        for x in csv.DictReader(f):
            s, d = ns(x['Start Time(us)']), ns(x['Duration(us)'])
            if d > 0:
                rr.append((s, s+d, x['Type'], x['Accelerator Core'], x['Model ID']))
    gaps = []
    for i, (left, right) in enumerate(zip(spec['executions'], spec['executions'][1:])):
        a, b = left['end_epoch_ns'], right['start_epoch_ns']
        if b <= a:
            continue
        types = collections.defaultdict(list)
        cores = collections.defaultdict(list)
        model = collections.defaultdict(list)
        for s, e, t, c, m in rr:
            iv = clip(s, e, a, b)
            if iv:
                types[t].append(iv)
                cores[c].append(iv)
                model[m].append(iv)
        gaps.append({'ordinal': i, 'start_epoch_ns': a, 'end_epoch_ns': b,
                     'span_ns': b-a,
                     'all_record_union_ns': union([iv for vv in types.values() for iv in vv]),
                     'type_union_ns': {t: union(vv) for t, vv in types.items()},
                     'type_intersecting_record_counts': {t: len(vv) for t, vv in types.items()},
                     'core_union_ns': {c: union(vv) for c, vv in cores.items()},
                     'model_union_ns': {m: union(vv) for m, vv in model.items()}})
    record = {'rank': spec['rank'], 'window': spec['window'], 'source_dir': spec['source_dir'],
              'kernel_sha256': spec['kernel_sha256'], 'gaps': gaps}
    if spec['rank'] in (0, 8):
        trace = folder/'trace_view.json'
        data = json.loads(trace.read_text())
        events = data['traceEvents'] if isinstance(data, dict) else data
        metadata = [x for x in events if x.get('ph') == 'M' and x.get('name') in ('process_name', 'thread_name')]
        cat_counts = collections.Counter(str(x.get('cat', '')) for x in events if x.get('ph') == 'X')
        cpu_gaps = []
        for gap in gaps:
            a, b = gap['start_epoch_ns'], gap['end_epoch_ns']
            bykey = collections.defaultdict(list)
            bycat = collections.defaultdict(list)
            for ev in events:
                if ev.get('ph') != 'X' or 'ts' not in ev or 'dur' not in ev:
                    continue
                s, d = ns(ev['ts']), ns(ev['dur'])
                iv = clip(s, s+d, a, b)
                if iv:
                    cat = str(ev.get('cat', ''))
                    bykey[(cat, str(ev.get('name', '')), str(ev.get('pid')), str(ev.get('tid')))].append(iv)
                    bycat[cat].append(iv)
            top = sorted(({'category': k[0], 'name': k[1], 'pid': k[2], 'tid': k[3],
                           'union_ns': union(vv), 'intersecting_record_count': len(vv)} for k, vv in bykey.items()),
                         key=lambda x: -x['union_ns'])[:45]
            cpu_gaps.append({'ordinal': gap['ordinal'], 'category_union_ns': {k: union(vv) for k, vv in bycat.items()},
                             'top_name_tracks_by_union': top})
        record['trace'] = {'sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
                           'complete_event_category_counts': dict(cat_counts),
                           'process_thread_metadata': metadata,
                           'gaps': cpu_gaps}
    output.append(record)
print(json.dumps({'source_count': len(output), 'scope': 'window 2 inter-main-segment exported records',
                  'limits': ['Intervals are clipped to gaps before union; hierarchical tracks/types can overlap and must not be added.',
                             'Host trace API scopes are not CPU active time; gap coverage does not establish critical dependencies.',
                             'Non-model sentinel records can contain draft, communication, update, and unrelated overhead.',
                             'This analysis does not assume uncovered time is device idle or removable.'], 'rows': output}, indent=2))
