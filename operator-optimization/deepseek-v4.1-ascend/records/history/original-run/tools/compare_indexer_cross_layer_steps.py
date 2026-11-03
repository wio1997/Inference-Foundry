#!/usr/bin/env python3
"""Read the saved stage177 Indexer records; compare adjacent layers within rank/run."""
import hashlib, json, statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / 'agents/diagnosis/indexer-layer-window-occupancy.json'
out = root / 'agents/diagnosis/indexer-cross-layer-steps.json'
raw = source.read_bytes()
expected = 'ca1d1dabed7cede2a95bc1ac117d24ad68f8fd2368f32c3aa225583cb3e9afc8'
assert hashlib.sha256(raw).hexdigest() == expected
rows = json.loads(raw)['rows']
assert len(rows) == 384
by = {}
for r in rows:
    key = (r['rank'], r['layer'], r['run'])
    assert key not in by
    assert abs((r['indexer']['end_ns'] - r['indexer']['start_ns']) / 1000 - r['indexer_duration_us']) < 1e-6
    by[key] = r['indexer_duration_us']
ranks = sorted({key[0] for key in by})
assert len(ranks) == 16
assert set(by) == {(rank, layer, run) for rank in ranks for layer in [2, 8, 14, 20, 24, 28, 32, 36] for run in [1, 2, 3]}
pairs = [(2, 8), (8, 14), (14, 20), (20, 24), (24, 28), (28, 32), (32, 36)]
results = []
for left, right in pairs:
    for run in [1, 2, 3]:
        items = [{'rank': rank, 'left_us': by[rank, left, run], 'right_us': by[rank, right, run], 'ratio': by[rank, right, run] / by[rank, left, run]} for rank in ranks]
        ratios = [x['ratio'] for x in items]
        results.append({'left_layer':left,'right_layer':right,'run':run,'count':len(items),'greater_count':sum(x>1 for x in ratios),'minimum_ratio':min(ratios),'median_ratio':statistics.median(ratios),'maximum_ratio':max(ratios),'items':items})
obj = {'source':str(source.relative_to(root)), 'source_sha256': expected, 'results':results, 'limits':['same rank and ordinal is not proof of same request or equal workload','different layers and runs cannot establish causal attribution','task duration is not end-to-end critical path','no hardware work']}
out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print('source sha256:', expected)
print('output sha256:', hashlib.sha256(out.read_bytes()).hexdigest())
for x in results:
    if (x['left_layer'], x['right_layer']) in [(14,20), (20,24)]:
        print(x['left_layer'], x['right_layer'], 'run', x['run'], 'greater', x['greater_count'], '/', x['count'], 'min/median/max', *[round(x[k],4) for k in ['minimum_ratio','median_ratio','maximum_ratio']])
