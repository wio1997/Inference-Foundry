#!/usr/bin/env python3
"""Recheck selected quant-gap/scatter co-occurrence in existing scope records."""
from collections import Counter
from decimal import Decimal as D
from hashlib import sha256
from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parents[2]
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
scatter = Path(sys.argv[2]) if len(sys.argv) > 2 else root / 'agents/fusion/f08-scatter-variation.json'
scatter4760 = Path(sys.argv[3]) if len(sys.argv) > 3 else root / 'agents/fusion/f08-4760-rank-overlap.json'
source_sha = '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
scatter_sha = '1290969851878c17b1eac8638dfeb1ae71ba045bfce4fda7cbb4e22f0d091dc5'
scatter4760_sha = 'ea892e7faf87ad87e010b3c40c5a4f5a5ecbc711ec06b7d2e9107b3e9cbcab5f'
assert sha256(source.read_bytes()).hexdigest() == source_sha
raw = scatter.read_bytes()
assert sha256(raw).hexdigest() == scatter_sha
data = json.loads(raw)
assert data['source_sha256'] == source_sha and data['n'] == 640
idx = {}
for row in data['all_samples_sorted']:
    key = (row['digest'], row['ep'], row['layer'], row['scope'])
    assert key not in idx
    idx[key] = row
assert len(idx) == 640
raw4760 = scatter4760.read_bytes()
assert sha256(raw4760).hexdigest() == scatter4760_sha
data4760 = json.loads(raw4760)
assert data4760['input_sha256']['diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'] == source_sha
idx4760 = {}
for row in data4760['records']:
    key = (row['digest'], row['ep'], row['layer'], row['scope_id'])
    assert key not in idx4760
    idx4760[key] = row
assert len(idx4760) == 320
assert [(r['ep'], r['layer']) for r in idx4760.values() if r['ep'] != 3 and r['class'] == 'short'] == [(2, 0)]

layers = {2, 8, 14, 20, 24, 28, 32, 36}
counts, seen, ep3 = Counter(), set(), []
with source.open() as f:
    for line in f:
        record = json.loads(line)
        if record.get('kind') != 'rank':
            continue
        for scope in record['data']['layers']:
            digest = scope['scheduler_digest'][:8]
            if digest not in ('db852bff', '7ddf7ae9') or scope['layer'] not in layers:
                continue
            key = (scope['scheduler_digest'], scope['ep'], scope['layer'], scope['layer_scope_id'])
            assert key not in seen and scope['ep'] == record['data']['ep'] and scope['ep'] in range(8)
            seen.add(key)
            qgroups, mgroups = [], []
            for g in scope['operator_groups']:
                sig = g['physical_shape_dtype_format']
                inp = sig['Input Shapes'].strip('"')
                m = inp.split(',')[0]
                if m not in ('595', '1019', '1023'):
                    continue
                if (g['operator_name'] == 'aclnnDynamicQuantV2_DynamicQuant_DynamicQuant'
                    and inp == f'{m},1280' and sig['Output Shapes'].strip('"') == f'{m},1280;{m}'
                    and len(g['tasks']) == 2):
                    qgroups.append((m, g))
                if (g['operator_name'] == 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3'
                    and inp == f'{m},1280;128,80,16,32;4096;{m}'
                    and sig['Output Shapes'].strip('"') == f'{m},4096' and len(g['tasks']) == 1):
                    mgroups.append((m, g))
            assert len(qgroups) == len(mgroups) == 1 and qgroups[0][0] == mgroups[0][0]
            m, qgroup = qgroups[0]
            assert (digest == '7ddf7ae9' and m == '595') or (
                digest == 'db852bff' and m == ('1019' if scope['ep'] == 7 else '1023')
            )
            q = sorted(qgroup['tasks'], key=lambda t: D(t['device_start_us']))[1]
            mat = mgroups[0][1]['tasks'][0]
            ident = lambda t: (t['device_track_pid'], t['device_track_tid'], t['csv']['Stream ID'])
            assert ident(q) == ident(mat)
            start = D(q['device_start_us']) + D(q['trace_duration_us'])
            end = D(mat['device_start_us'])
            assert end >= start
            gap = end - start
            if digest == 'db852bff':
                row = idx[key]
                candidates = [t for g in scope['operator_groups'] for t in g['tasks']
                              if t['csv']['Task ID'] == row['task']
                              and t['csv']['Stream ID'] == row['stream']
                              and t['csv']['Type'] == 'ScatterNdUpdateSk'
                              and t['device_start_us'] == row['start_us']]
            else:
                row = idx4760[key]
                candidates = [t for g in scope['operator_groups'] for t in g['tasks']
                              if t['csv']['Task ID'] == str(row['tasks'][2])
                              and t['flow_id'] == row['flow_ids'][2]
                              and t['csv']['Type'] == 'ScatterNdUpdateSk']
            assert len(candidates) == 1
            st = candidates[0]
            duration = D(st['trace_duration_us'])
            assert abs(duration - D(str(row['duration_us']))) <= D('0.001')
            assert st['device_track_pid'] == q['device_track_pid']
            assert (duration < 50) or (duration > (200 if digest == 'db852bff' else 100))
            assert (';8180,2;8180,512' if digest == 'db852bff' else ';4760,2;4760,512') in st['csv']['Input Shapes']
            counts[(digest, gap > 10, duration < 50)] += 1
            if scope['ep'] == 3:
                scatter_end = D(st['device_start_us']) + duration
                ep3.append({'digest': digest, 'layer': scope['layer'], 'scope': scope['layer_scope_id'], 'physical_rows': int(m),
                            'quant_task': q['csv']['Task ID'], 'matmul_task': mat['csv']['Task ID'],
                            'scatter_task': st['csv']['Task ID'], 'scatter_stream': st['csv']['Stream ID'],
                            'device_gap_us': str(gap), 'scatter_duration_us': str(duration),
                            'scatter_end_minus_gap_start_us': str(scatter_end - start)})

assert len(seen) == 128 and len(ep3) == 16
assert counts == {('db852bff', True, True): 7, ('db852bff', False, False): 57,
                  ('7ddf7ae9', True, True): 8, ('7ddf7ae9', False, False): 56}
print(json.dumps({
    'source_sha256': source_sha, 'scatter_sha256': scatter_sha, 'scatter4760_sha256': scatter4760_sha,
    'scopes': len(seen),
    'posthoc_thresholds_us': {'quant_gap_gt': 10, 'scatter_duration_lt': 50},
    'contingency': {digest: {'gap_long_scatter_short': counts[(digest, True, True)],
                             'gap_long_scatter_not_short': counts[(digest, True, False)],
                             'gap_not_long_scatter_short': counts[(digest, False, True)],
                             'gap_not_long_scatter_not_short': counts[(digest, False, False)]}
                    for digest in ('db852bff', '7ddf7ae9')},
    'ep3_by_layer': sorted(ep3, key=lambda r: (r['digest'], r['layer'])),
    'outside_selected_layers': '7ddf7ae9 EP2 layer 0 has short scatter but no selected quant/matmul pair; do not fill its gap with zero.',
    'limits': 'Same profiled execution and post-hoc thresholds, not independent requests or causality. '
              'Different-stream relative clock error is unverified. The two scatter tables have different input-row domains.',
}, indent=2))
