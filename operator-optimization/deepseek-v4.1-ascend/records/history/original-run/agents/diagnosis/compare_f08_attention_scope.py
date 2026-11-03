#!/usr/bin/env python3
"""Compare unique task durations inside the same recorded scope, not wall time."""
import collections
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parents[2]
src = root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
triples = root / 'agents/fusion/f08-actual-triples.json'
expected = {
    src: '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4',
    triples: 'e6c18bd23809d47652cc58faf8d5ac81f7879be0e483fc72d70264cd3a3940d5',
}
for path, digest in expected.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
items = json.loads(triples.read_text())['records']
index = {(r['ep'], r['scope_id']): r for r in items}
assert len(index) == len(items) == 1280
rows = collections.defaultdict(list)
scatter_matches = 0
seen_scopes = set()
match_counts = collections.defaultdict(collections.Counter)
for line in src.open():
    data = json.loads(line)
    if data.get('kind') != 'rank':
        continue
    for scope in data['data']['layers']:
        r = index.get((scope['ep'], scope['layer_scope_id']))
        if r is None:
            continue
        key = (scope['ep'], scope['layer_scope_id'])
        assert key not in seen_scopes
        seen_scopes.add(key)
        tasks = [t for g in scope['operator_groups'] for t in g['tasks']]
        scatters = [t for t in tasks if t['flow_id'] == r['flow_ids'][2]]
        assert len(scatters) == 1
        scatter = scatters[0]
        assert scatter['csv']['Type'] == 'ScatterNdUpdateSk'
        scatter_matches += 1
        for name in ('SparseFlashMla', 'VllmAscendQuantLightningIndexerV2'):
            matches = [t for t in tasks if t['csv']['Type'] == name]
            match_counts[name][len(matches)] += 1
            if len(matches) != 1:
                continue
            task = matches[0]
            rows[name].append({
                'ep': r['ep'], 'layer': r['layer'], 'scope_id': r['scope_id'],
                'rows': r['rows'], 'scatter_flow': scatter['flow_id'],
                'other_flow': task['flow_id'], 'scatter_stream': scatter['csv']['Stream ID'],
                'other_stream': task['csv']['Stream ID'],
                'scatter_us': scatter['trace_duration_us'],
                'other_us': task['trace_duration_us'],
            })
assert scatter_matches == 1280
assert seen_scopes == set(index)
result = {'input_sha256': {str(p.relative_to(root)): s for p, s in expected.items()},
          'scope_count': scatter_matches, 'groups': {},
          'matching_count_distribution': {k: dict(v) for k, v in match_counts.items()},
          'limitations': 'Profiled prefill scopes, not independent requests; same scope does not establish tensor identity, dependency, exposed critical path, representative serving throughput, or causal speedup. Distinct stream tasks may overlap; no task times summed.'}
for name, rs in rows.items():
    def summ(group):
        s = [Decimal(x['scatter_us']) for x in group]
        a = [Decimal(x['other_us']) for x in group]
        return {'count': len(group), 'other_longer_count': sum(y > x for x, y in zip(s, a)),
                'same_stream_count': sum(x['scatter_stream'] == x['other_stream'] for x in group),
                'scatter_median_us': str(median(s)), 'other_median_us': str(median(a))}
    result['groups'][name] = {'all': summ(rs), 'by_rows': {str(n): summ([r for r in rs if r['rows'] == n]) for n in sorted({r['rows'] for r in rs})}, 'records': rs}
output = root / 'agents/diagnosis/f08-attention-scope-comparison.json'
output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({name: g['all'] for name, g in result['groups'].items()}, ensure_ascii=False))
print('output_sha256', hashlib.sha256(output.read_bytes()).hexdigest())
