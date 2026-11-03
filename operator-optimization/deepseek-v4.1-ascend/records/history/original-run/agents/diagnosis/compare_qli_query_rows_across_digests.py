#!/usr/bin/env python3
"""Read-only comparison of selected QLI trace durations and physical shapes."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[2]
path = root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
prefixes, layers = ('7ddf7ae9', 'db852bff'), (24, 28, 32, 36)
records, digests = {}, {prefix: set() for prefix in prefixes}
with path.open() as source:
    for line in source:
        entry = json.loads(line)
        if entry.get('kind') != 'rank':
            continue
        for scope in entry['data']['layers']:
            prefix = scope['scheduler_digest'][:8]
            if prefix not in prefixes or scope['ep'] not in range(7) or scope['layer'] not in layers:
                continue
            key = (prefix, scope['ep'], scope['layer'])
            assert key not in records
            digests[prefix].add(scope['scheduler_digest'])
            tasks = [t for g in scope['operator_groups'] if g['operator_name'] == 'VllmAscendQuantLightningIndexerV2' for t in g['tasks']]
            assert len(tasks) == 1
            records[key] = tasks[0]
assert len(records) == 56 and all(len(x) == 1 for x in digests.values())
ratios = []
for ep in range(7):
    for layer in layers:
        a, b = (records[prefix, ep, layer] for prefix in prefixes)
        ca, cb = a['csv'], b['csv']
        sa, sb = (c['Input Shapes'].strip('"').split(';') for c in (ca, cb))
        assert sa[0] == '595,32,128' and sb[0] == '1023,32,128'
        assert sa[1] == sb[1] == '17315,128,1,128'
        assert sa[4] == sb[4] == '17315,128,1'
        assert (sa[5], sa[8], sa[10]) == ('5', '4', '4,272')
        assert (sb[5], sb[8], sb[10]) == ('4', '3', '3,272')
        assert sa[-1] == '595,1,2048' and sb[-1] == '1023,1,2048'
        assert ca['Output Shapes'] == '"595,1,512;;"' and cb['Output Shapes'] == '"1023,1,512;;"'
        for field in ('Name', 'Type', 'Input Data Types', 'Output Data Types', 'Input Formats', 'Output Formats', 'Accelerator Core', 'Block Num', 'Mix Block Num'):
            assert ca[field] == cb[field]
        assert ca['Input Data Types'].split(';')[1] == 'INT8'
        assert ca['Input Data Types'].split(';')[4] == 'FLOAT16'
        t0, t1 = Decimal(a['trace_duration_us']), Decimal(b['trace_duration_us'])
        assert t0 > 0 and t1 > 0
        ratios.append(t1 / t0)
print('pairs', len(ratios), 'min/median/max', min(ratios), statistics.median(ratios), max(ratios))
print('ratio_above_1', sum(x > 1 for x in ratios), 'below_row_ratio', sum(x < Decimal(1023)/595 for x in ratios), 'above_row_ratio', sum(x > Decimal(1023)/595 for x in ratios))
