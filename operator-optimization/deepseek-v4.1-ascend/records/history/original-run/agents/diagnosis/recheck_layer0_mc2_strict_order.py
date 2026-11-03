#!/usr/bin/env python3
"""Recheck existing attributed MC2 samples only; performs no device operations."""
import contextlib
import hashlib
import io
import json
import runpy
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

by_rank_run = defaultdict(dict)
for row in data['records']:
    by_rank_run[(row['rank'], row['run'])][row['layer']] = row['duration_us']
assert len(by_rank_run) == 48 and all(set(v) == set(range(40)) for v in by_rank_run.values())
results = []
for run in (1, 2, 3):
    groups = {}
    for dp in ('dp0', 'dp1'):
        rows = [(rank, layers) for (rank, r), layers in by_rank_run.items()
                if r == run and rank.split('_')[0] == dp]
        assert len(rows) == 8
        values = [layers[0] for _, layers in rows]
        ranks = [1 + sum(layers[i] > layers[0] for i in range(1, 40)) for _, layers in rows]
        groups[dp] = {'count': len(rows), 'layer0_min_us': min(values),
                      'layer0_max_us': max(values),
                      'layer0_strictly_longest_count': sum(layers[0] > max(layers[i] for i in range(1, 40)) for _, layers in rows),
                      'layer0_duration_rank_min': min(ranks), 'layer0_duration_rank_max': max(ranks)}
    results.append({'run': run, 'groups': groups,
                    'dp1_all_greater_than_dp0': groups['dp1']['layer0_min_us'] > groups['dp0']['layer0_max_us']})
print(json.dumps({'inputs_sha256': {str(p.name): v for p, v in EXPECTED.items()},
                  'method': 'within-rank durations, previously validated layer attribution; no cross-device timestamps',
                  'results': results,
                  'limits': ['sampled windows are not independent experiments',
                             'no same request, work or collective asserted',
                             'no cause, removable time, overall device utilization, or general performance bound inferred'],
                  'no_device_work': True}, ensure_ascii=False, indent=2))
