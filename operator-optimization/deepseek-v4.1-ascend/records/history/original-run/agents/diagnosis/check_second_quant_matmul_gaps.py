#!/usr/bin/env python3
"""Inspect long same-scope selected-task gaps without device activity."""
from collections import Counter
from decimal import Decimal as D
from hashlib import sha256
from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parents[2]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
)
source_sha = '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert sha256(path.read_bytes()).hexdigest() == source_sha
layers = {2, 8, 14, 20, 24, 28, 32, 36}
seen, long_gaps, other_gaps = set(), [], []

with path.open() as f:
    for line in f:
        record = json.loads(line)
        if record.get('kind') != 'rank':
            continue
        for scope in record['data']['layers']:
            if scope['layer'] not in layers:
                continue
            key = (scope['scheduler_digest'], scope['ep'], scope['layer'])
            assert key not in seen and scope['ep'] == record['data']['ep']
            seen.add(key)
            quant, matmul = [], []
            for group in scope['operator_groups']:
                sig = group['physical_shape_dtype_format']
                inp = sig['Input Shapes'].strip('"')
                m = inp.split(',')[0]
                if m not in ('595', '643', '1019', '1023'):
                    continue
                if (group['operator_name'] == 'aclnnDynamicQuantV2_DynamicQuant_DynamicQuant'
                    and inp == f'{m},1280' and sig['Output Shapes'].strip('"') == f'{m},1280;{m}'
                    and len(group['tasks']) == 2):
                    quant.append((m, group))
                if (group['operator_name'] == 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3'
                    and inp == f'{m},1280;128,80,16,32;4096;{m}'
                    and sig['Output Shapes'].strip('"') == f'{m},4096' and len(group['tasks']) == 1):
                    matmul.append((m, group))
            assert len(quant) == len(matmul) == 1 and quant[0][0] == matmul[0][0], key
            m, qgroup = quant[0]
            first, q = sorted(qgroup['tasks'], key=lambda t: D(t['device_start_us']))
            mat = matmul[0][1]['tasks'][0]
            ident = lambda t: (t['device_track_pid'], t['device_track_tid'], t['csv']['Stream ID'])
            assert ident(first) == ident(q) == ident(mat)
            assert D(first['device_start_us']) + D(first['trace_duration_us']) <= D(q['device_start_us'])
            a = D(q['device_start_us']) + D(q['trace_duration_us'])
            b = D(mat['device_start_us'])
            assert b >= a
            gap = b - a
            if gap <= 10:
                other_gaps.append(gap)
                continue
            matches = []
            for group in scope['operator_groups']:
                for task in group['tasks']:
                    if ident(task) != ident(q):
                        continue
                    start = D(task['device_start_us'])
                    end = start + D(task['trace_duration_us'])
                    if start < b and end > a:
                        matches.append({'type': task['csv']['Type'], 'task_id': task['csv']['Task ID']})
            long_gaps.append({
                'ep': scope['ep'], 'layer': scope['layer'], 'scope': scope['layer_scope_id'],
                'digest_prefix': scope['scheduler_digest'][:8], 'physical_rows': int(m),
                'track_pid': q['device_track_pid'], 'track_tid': q['device_track_tid'],
                'stream': q['csv']['Stream ID'], 'quant_task': q['csv']['Task ID'],
                'matmul_task': mat['csv']['Task ID'], 'gap_start_us': str(a), 'gap_end_us': str(b),
                'gap_us': str(gap), 'selected_same_track_matches': matches,
            })

assert len(seen) == 256 and len(long_gaps) == 15 and len(other_gaps) == 241
assert all(r['ep'] == 3 and not r['selected_same_track_matches'] for r in long_gaps)
assert Counter(r['digest_prefix'] for r in long_gaps) == {'7ddf7ae9': 8, 'db852bff': 7}
print(json.dumps({
    'source_sha256': source_sha, 'scopes': len(seen), 'threshold_us': '10',
    'threshold_note': 'Exploratory threshold selected after observing the gaps.',
    'long_gap_count': len(long_gaps), 'other_gap_max_us': str(max(other_gaps)),
    'long_gaps': sorted(long_gaps, key=lambda r: (r['digest_prefix'], r['layer'])),
    'limits': 'Only previously matched tasks within each selected scope and the same device track/stream. '
              'No match does not establish physical idleness, cause, or complete recording.',
}, indent=2))
