#!/usr/bin/env python3
"""Pair saved same-rank/layer windows; never opens device or raw profile."""
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'indexer-layer-window-occupancy.json'
OUTPUT = HERE / 'indexer-layer-paired-deltas.json'
raw = SOURCE.read_bytes()
data = json.loads(raw)
lookup = {}
for row in data['rows']:
    key = (row['rank'], row['layer'], row['run'])
    if key in lookup:
        raise ValueError(f'duplicate {key}')
    lookup[key] = row
ranks = sorted({k[0] for k in lookup})
layers = sorted({k[1] for k in lookup})
assert len(ranks) == 16 and layers == [2, 8, 14, 20, 24, 28, 32, 36]
assert len(lookup) == 16 * 8 * 3
records = []
summaries = []
for layer in layers:
    for to_run in (2, 3):
        group = []
        for rank in ranks:
            a, b = lookup[(rank, layer, 1)], lookup[(rank, layer, to_run)]
            for r in (a, b):
                assert r['window_start_ns'] <= r['indexer']['start_ns'] <= r['indexer']['end_ns'] <= r['attention']['start_ns'] <= r['attention']['end_ns'] <= r['window_end_ns']
            delta = lambda k: round(b[k] - a[k], 6)
            span, indexer, attention = (delta(k) for k in ('window_duration_us', 'indexer_duration_us', 'attention_duration_us'))
            residual = round(span - indexer - attention, 6)
            rec = {'rank': rank, 'layer': layer, 'from_run': 1, 'to_run': to_run,
                   'window_delta_us': span, 'indexer_delta_us': indexer,
                   'attention_delta_us': attention, 'residual_delta_us': residual}
            group.append(rec)
            records.append(rec)
        summaries.append({'layer': layer, 'from_run': 1, 'to_run': to_run,
            'count': len(group),
            'medians_us': {k: statistics.median(x[k] for x in group) for k in ('window_delta_us', 'indexer_delta_us', 'attention_delta_us', 'residual_delta_us')},
            'positive_counts': {k: sum(x[k] > 0 for x in group) for k in ('window_delta_us', 'indexer_delta_us', 'attention_delta_us', 'residual_delta_us')},
            'negative_residual_count': sum(x['residual_delta_us'] < 0 for x in group)})
out = {'source': SOURCE.name, 'source_sha256': hashlib.sha256(raw).hexdigest(),
       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'scope': '16 ranks, 8 layers, 3 ordinal samples; same rank/layer; runs are not matched workloads',
       'method': 'Per pair residual = layer-window delta - indexer task duration delta - attention task duration delta. Median is computed after pairing. Task spans non-overlap within source windows.',
       'limitations': ['residual is not a single operator or causal attribution', 'no end-to-end effect or achievable gain', 'no causal claim across unmatched input/cache/request', 'other streams and unobserved tasks may overlap', 'window boundaries and source attribution inherit input limitations'],
       'records': records, 'summaries': summaries, 'no_device_work': True}
OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'output': str(OUTPUT), 'source_sha256': out['source_sha256'], 'summaries': summaries}, ensure_ascii=False))
