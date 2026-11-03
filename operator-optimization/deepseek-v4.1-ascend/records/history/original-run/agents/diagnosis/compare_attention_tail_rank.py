#!/usr/bin/env python3
"""Read only: compare selected 1019-row SparseFlashMla records with peers."""
import hashlib
import json
import re
import statistics
from decimal import Decimal
from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
fields = ('Name', 'Type', 'Input Shapes', 'Output Shapes', 'Input Data Types',
          'Output Data Types', 'Input Formats', 'Output Formats',
          'Accelerator Core', 'Block Num', 'Mix Block Num')
groups = {'db852bff': 0, '1b826883': 8}
records = {}
digests = {key: set() for key in groups}
with path.open() as source:
    for line in source:
        entry = json.loads(line)
        if entry.get('kind') != 'rank':
            continue
        for scope in entry['data']['layers']:
            prefix = scope['scheduler_digest'][:8]
            if prefix not in groups or scope['ep'] not in range(groups[prefix], groups[prefix] + 8):
                continue
            digests[prefix].add(scope['scheduler_digest'])
            key = (prefix, scope['ep'], scope['layer'])
            assert key not in records
            tasks = [task for group in scope['operator_groups']
                     if group['operator_name'] == 'SparseFlashMla' for task in group['tasks']]
            assert len(tasks) == 1
            records[key] = tasks[0]
assert all(len(values) == 1 for values in digests.values())
assert len(records) == 640
for prefix, base in groups.items():
    ratios = []
    for layer in range(40):
        tasks = [records[prefix, base + ep, layer] for ep in range(8)]
        signatures = []
        for ep, task in enumerate(tasks):
            csv = task['csv']
            rows = '1019' if ep == 7 else '1023'
            assert csv['Input Shapes'].startswith('"' + rows + ',64,512;')
            assert csv['Output Shapes'].startswith('"' + rows + ',64,512;')
            def normalized(k):
                value = csv[k]
                if k in ('Input Shapes', 'Output Shapes'):
                    return re.sub(r'(?<!\d)1019(?!\d)', '1023', value)
                return value
            signatures.append(tuple(normalized(k) for k in fields))
        assert len(set(signatures)) == 1
        duration = [Decimal(task['trace_duration_us']) for task in tasks]
        assert all(v > 0 for v in duration)
        ratios.append(duration[7] / statistics.median(duration[:7]))
    print(prefix, 'n', len(ratios), 'min/median/max', min(ratios),
          statistics.median(ratios), max(ratios),
          'within post-hoc [0.9,1.1]', sum(Decimal('0.9') <= v <= Decimal('1.1') for v in ratios))
