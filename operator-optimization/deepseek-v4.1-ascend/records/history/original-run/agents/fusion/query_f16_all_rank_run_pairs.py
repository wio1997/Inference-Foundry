#!/usr/bin/env python3
"""Read-only fixed EP0-15 layer24 run1-3 BF16 projection-shaped CSV rows."""
import argparse, csv, hashlib, json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--task-root', required=True)
p.add_argument('--trace-root', required=True)
a = p.parse_args()
t, trace = Path(a.task_root), Path(a.trace_root)
wp = t / 'captured-layer-attribution/module-call-metrics.json'
mp = t / 'evidence/TRACE-EXPORT-SHA256.json'
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
assert digest(wp) == 'd98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a'
assert digest(mp) == '45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'
expected = {e['rank']: e['kernel_csv_sha256'] for e in json.loads(mp.read_text())['exports']}
by_rank = defaultdict(list)
for w in json.loads(wp.read_text()):
    if w['run_in_model'] in (1, 2, 3) and w['layer'] == 24:
        by_rank[w['rank']].append(w)
assert set(by_rank) == set(expected) and len(by_rank) == 16
rows, hashes = [], {}
for rank, windows in sorted(by_rank.items()):
    assert len(windows) == 3 and {w['run_in_model'] for w in windows} == {1,2,3}
    windows.sort(key=lambda w: w['window_start_ns'])
    assert all(x['window_end_ns'] <= y['window_start_ns'] for x,y in zip(windows,windows[1:]))
    assert all(str(w['model_id']) == '49' for w in windows)
    path = trace / rank / 'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
    assert path.stat().st_size <= 50 * 1024 * 1024
    hashes[rank] = digest(path)
    assert hashes[rank] == expected[rank]
    hits = {w['run_in_model']: {'first': [], 'second': []} for w in windows}
    with path.open(newline='') as f:
        for z in csv.DictReader(f):
            typ = z['Type']
            if typ not in ('TransposeBatchMatMul', 'MatMulV2'):
                continue
            start = int(Decimal(z['Start Time(us)'].strip()) * 1000)
            end = start + int(Decimal(z['Duration(us)'].strip()) * 1000)
            if end <= start:
                continue
            for w in windows:
                if not (w['window_start_ns'] <= start < end <= w['window_end_ns'] and z['Model ID'].strip() == '49' and z['Stream ID'].strip() == str(w['main_stream_id'])):
                    continue
                shape = z['Input Shapes'].strip().strip('"')
                output_shape = z['Output Shapes'].strip().strip('"')
                dtypes = z['Input Data Types'].strip()
                output_dtype = z['Output Data Types'].strip()
                key = None
                if typ == 'TransposeBatchMatMul' and shape == '24,1,4096;1,4096,1024' and output_shape == '24,1,1024':
                    key = 'first'
                elif typ == 'MatMulV2' and shape == '24,1024;5120,1024' and output_shape == '24,5120':
                    key = 'second'
                if key is not None and dtypes == 'DT_BF16;DT_BF16' and output_dtype == 'DT_BF16':
                    hits[w['run_in_model']][key].append({'task_id': z['Task ID'], 'type': typ,
                        'start_rel_ns': start - w['window_start_ns'], 'end_rel_ns': end - w['window_start_ns'],
                        'duration_ns': end - start, 'input_shapes': shape, 'output_shapes': output_shape,
                        'input_dtypes': dtypes, 'output_dtype': output_dtype})
    for w in windows:
        h = hits[w['run_in_model']]
        assert len(h['first']) <= 16 and len(h['second']) <= 16, 'excess candidate rows'
        paired = len(h['first']) == len(h['second']) == 1 and h['first'][0]['end_rel_ns'] <= h['second'][0]['start_rel_ns']
        rows.append({'rank': rank, 'run': w['run_in_model'], 'layer': 24, 'model': '49',
                     'main_stream': str(w['main_stream_id']), 'first_count': len(h['first']),
                     'second_count': len(h['second']), 'first': h['first'], 'second': h['second'],
                     'unique_ordered_candidates': paired})
assert len(rows) == 48
counts = Counter((r['run'],r['first_count'],r['second_count'],r['unique_ordered_candidates']) for r in rows)
print(json.dumps({'scope': '48 fixed rank-local target layer24 windows; output matching exact F16 type, physical shapes and BF16, model49 and main stream only',
                  'inputs': {'window_sha256': digest(wp), 'manifest_sha256': digest(mp), 'kernel_csv_sha256': hashes},
                  'rows': rows, 'counts': [{'run': k[0], 'first_count': k[1], 'second_count': k[2], 'ordered': k[3], 'windows': n} for k,n in sorted(counts.items())],
                  'logical_mac_if_standard_dense': {'first': 24*4096*1024, 'second': 24*1024*5120},
                  'limits': ['Unique temporal candidates are not tensor, weight, request, source, runtime instance or dependency identities.',
                             'Physical shapes and dense-work assumptions do not prove actual valid token count, instruction count, tiling, or device path.',
                             'Same rank run is not matched request/work across runs; no cross-rank timestamp arithmetic.',
                             'No NPU work; absence of selected rows is not evidence of idle or absence of fused variants.']}, indent=2))
