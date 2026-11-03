#!/usr/bin/env python3
"""Inspect saved host-flow starts and selected other tracks around quant/matmul gaps."""
from collections import Counter, defaultdict
from decimal import Decimal as D
from hashlib import sha256
from pathlib import Path
from statistics import median
import contextlib
import io
import json
import runpy
import sys

root = Path(__file__).resolve().parents[2]
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
source_sha = '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
prior = Path(__file__).with_name('check_second_quant_matmul_gaps.py')
assert sha256(source.read_bytes()).hexdigest() == source_sha
assert sha256(prior.read_bytes()).hexdigest() == '2ca5886bbbe0184a7910e19d3dbfb2d03bdfd817182f5fca2e77d8050f7e1ba7'
with contextlib.redirect_stdout(io.StringIO()):
    checked = runpy.run_path(str(prior), run_name='prior_check')
long_by_scope = {x['scope']: x for x in checked['long_gaps']}
assert len(long_by_scope) == 15
layers = {2, 8, 14, 20, 24, 28, 32, 36}
seen = set()
grouped = defaultdict(list)
other_counts = Counter()
fixed = {}
with source.open() as f:
    for line in f:
        record = json.loads(line)
        if record.get('kind') != 'rank':
            continue
        for scope in record['data']['layers']:
            if scope['layer'] not in layers:
                continue
            digest = scope['scheduler_digest'][:8]
            key = (digest, scope['ep'], scope['layer'])
            assert key not in seen and scope['ep'] == record['data']['ep']
            seen.add(key)
            qs, ms = [], []
            for g in scope['operator_groups']:
                sig = g['physical_shape_dtype_format']
                sh = sig['Input Shapes'].strip('"')
                m = sh.split(',')[0]
                if m not in ('595', '643', '1019', '1023'):
                    continue
                if (g['operator_name'] == 'aclnnDynamicQuantV2_DynamicQuant_DynamicQuant'
                    and sh == f'{m},1280' and sig['Output Shapes'].strip('"') == f'{m},1280;{m}'
                    and len(g['tasks']) == 2):
                    qs.append((m, g['tasks']))
                if (g['operator_name'] == 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3'
                    and sh == f'{m},1280;128,80,16,32;4096;{m}'
                    and sig['Output Shapes'].strip('"') == f'{m},4096' and len(g['tasks']) == 1):
                    ms.append((m, g['tasks'][0]))
            assert len(qs) == len(ms) == 1 and qs[0][0] == ms[0][0], key
            m, qt = qs[0]
            q = sorted(qt, key=lambda t: D(t['device_start_us']))[1]
            mat = ms[0][1]
            ident = lambda t: (t['device_track_pid'], t['device_track_tid'], t['csv']['Stream ID'])
            assert ident(q) == ident(mat)
            assert (q['host_pid'], q['host_tid']) == (mat['host_pid'], mat['host_tid'])
            a = D(q['device_start_us']) + D(q['trace_duration_us'])
            b = D(mat['device_start_us'])
            assert b >= a
            delta = D(mat['host_start_us']) - D(q['host_start_us'])
            bucket = 'long' if b - a > 10 else 'short'
            grouped[(digest, m, bucket)].append(delta)
            scope_id = scope['layer_scope_id']
            if scope_id in ('2128-26', '2128-42'):
                assert scope['ep'] == 3 and digest == 'db852bff'
                fixed[scope_id] = {'layer': scope['layer'], 'host_flow_start_delta_us': str(delta),
                                   'device_gap_us': str(b-a), 'quant_task': q['csv']['Task ID'],
                                   'matmul_task': mat['csv']['Task ID']}
            if bucket == 'long':
                assert scope_id in long_by_scope
                anchor = long_by_scope[scope_id]
                assert str(a) == anchor['gap_start_us'] and str(b) == anchor['gap_end_us']
                assert q['csv']['Task ID'] == anchor['quant_task'] and mat['csv']['Task ID'] == anchor['matmul_task']
                intersections = 0
                for g in scope['operator_groups']:
                    for t in g['tasks']:
                        if t['device_track_pid'] != q['device_track_pid'] or ident(t) == ident(q):
                            continue
                        start = D(t['device_start_us'])
                        end = start + D(t['trace_duration_us'])
                        if start < b and end > a:
                            intersections += 1
                other_counts[intersections] += 1
assert len(seen) == 256 and other_counts == {0: 15}
assert set(fixed) == {'2128-26', '2128-42'}
summary = {}
for (digest, m, bucket), values in sorted(grouped.items()):
    summary[f'{digest}:{m}:{bucket}'] = {
        'scopes': len(values), 'host_flow_start_delta_us_min_median_max': [
            str(min(values)), str(median(values)), str(max(values))],
    }
print(json.dumps({
    'source_sha256': source_sha, 'scopes': len(seen), 'groups': summary,
    'long_gaps_with_other_track_selected_intersections': dict(other_counts),
    'same_ep_same_digest_examples': fixed,
    'limits': 'Host flow start deltas are not API durations or queue waits. Other-track overlap uses '
              'exported coordinates with unverified relative clock error; only selected same-scope '
              'matched tasks are covered. No evidence of physical idleness or cause.',
}, indent=2))
