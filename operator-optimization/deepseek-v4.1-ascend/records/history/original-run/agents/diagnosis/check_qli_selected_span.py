#!/usr/bin/env python3
"""Summarize three selected same-track device intervals within each scope."""
from decimal import Decimal as D
from hashlib import sha256
from pathlib import Path
from statistics import median
import json
import sys

root = Path(__file__).resolve().parents[2]
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
selected = Path(sys.argv[2]) if len(sys.argv) > 2 else root / 'agents/diagnosis/qli-target-scope-distribution.json'
assert sha256(source.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert sha256(selected.read_bytes()).hexdigest() == '61aa739e43197d4c0725d81aea30624728de0df9c7b2ffea772d5de53b0ca804'
data = json.loads(selected.read_bytes())
want = {}
for row in data['records']:
    if not row['all_signatures_equal_ref']:
        continue
    key = (row['ep'], row['layer'], row['scope_id'])
    assert key not in want
    want[key] = row
assert len(want) == 28
types = ('VllmAscendQuantLightningIndexerV2', '_prepare_indexer_indices_kernel', 'SparseFlashMla')
result = []
with source.open() as f:
    for line in f:
        record = json.loads(line)
        if record.get('kind') != 'rank':
            continue
        for scope in record['data']['layers']:
            key = (scope['ep'], scope['layer'], scope['layer_scope_id'])
            if key not in want:
                continue
            assert scope['ep'] == record['data']['ep'] and scope['scheduler_digest'][:8] == 'db852bff'
            chosen = want.pop(key)
            found = {}
            for g in scope['operator_groups']:
                for task in g['tasks']:
                    for typ in types:
                        if task['flow_id'] == chosen['flows'][typ]:
                            assert typ not in found and task['csv']['Type'] == typ
                            assert task['csv']['Task ID'] == chosen['task_ids'][typ]
                            found[typ] = task
            assert set(found) == set(types)
            x, y, z = (found[t] for t in types)
            ident = lambda t: (t['device_track_pid'], t['device_track_tid'], t['csv']['Stream ID'])
            assert ident(x) == ident(y) == ident(z)
            starts = [D(t['device_start_us']) for t in (x, y, z)]
            duration = [D(t['trace_duration_us']) for t in (x, y, z)]
            assert min(duration) > 0
            assert starts[0] + duration[0] <= starts[1]
            assert starts[1] + duration[1] <= starts[2]
            span = starts[2] + duration[2] - starts[0]
            used = sum(duration)
            result.append({'ep': scope['ep'], 'layer': scope['layer'], 'scope': scope['layer_scope_id'],
                           'span': span, 'selected_fraction': used / span, 'qli_fraction': duration[0] / span,
                           'durations': duration})
assert not want and len(result) == 28
def stats(field):
    values = [r[field] for r in result]
    return [str(min(values)), str(median(values)), str(max(values))]
example = [r for r in result if (r['ep'], r['layer']) == (3, 36)]
assert len(example) == 1
print(json.dumps({
    'scope_count': len(result), 'span_us_min_median_max': stats('span'),
    'selected_duration_fraction_min_median_max': stats('selected_fraction'),
    'qli_fraction_min_median_max': stats('qli_fraction'),
    'ep3_l36': {k: ([str(x) for x in v] if k == 'durations' else str(v) if isinstance(v, D) else v)
                for k, v in example[0].items()},
    'limits': 'Same-track selected trace intervals only; no proof of data dependency, complete activity, '
              'hardware utilization, request critical path or online performance.',
}, indent=2))
