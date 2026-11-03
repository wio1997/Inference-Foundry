#!/usr/bin/env python3
"""Summarize selected same-scope dynamic quant and quantized matmul times."""
from collections import Counter, defaultdict
from decimal import Decimal as D
from hashlib import sha256
from pathlib import Path
from statistics import median
import json
import sys

root = Path(__file__).resolve().parents[2]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
)
sha = '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert sha256(path.read_bytes()).hexdigest() == sha
layers = {2, 8, 14, 20, 24, 28, 32, 36}
expected = {('1b826883', 1019): 8, ('1b826883', 1023): 56,
            ('7ddf7ae9', 595): 64, ('8c18c293', 643): 64,
            ('db852bff', 1019): 8, ('db852bff', 1023): 56}
groups = defaultdict(list)
seen = set()
with path.open() as f:
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
                    quant.append((int(m), group))
                if (group['operator_name'] == 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3'
                    and inp == f'{m},1280;128,80,16,32;4096;{m}'
                    and sig['Output Shapes'].strip('"') == f'{m},4096' and len(group['tasks']) == 1):
                    matmul.append((int(m), group))
            assert len(quant) == len(matmul) == 1 and quant[0][0] == matmul[0][0], key
            m, qgroup = quant[0]
            first, second = sorted(qgroup['tasks'], key=lambda t: D(t['device_start_us']))
            mat = matmul[0][1]['tasks'][0]
            for q in (first, second):
                c = q['csv']
                assert (c['Type'], c['Input Data Types'], c['Output Data Types'], c['Input Formats'],
                        c['Output Formats'], c['Accelerator Core'], c['Block Num'], c['Mix Block Num']) == (
                    'DynamicQuant', 'DT_BF16', 'INT8;FLOAT', 'ND', 'ND;ND', 'AI_VECTOR_CORE', '48', '0')
            c = mat['csv']
            assert (c['Type'], c['Input Data Types'], c['Output Data Types'], c['Input Formats'],
                    c['Output Formats'], c['Accelerator Core'], c['Block Num'], c['Mix Block Num']) == (
                'QuantBatchMatmulV3', 'INT8;INT8;DT_BF16;FLOAT', 'DT_BF16',
                'ND;FRACTAL_NZ;ND;ND', 'ND', 'MIX_AIC', '24', '24')
            ident = lambda t: (t['device_track_pid'], t['device_track_tid'], t['csv']['Stream ID'])
            assert ident(first) == ident(second) == ident(mat)
            assert D(first['device_start_us']) + D(first['trace_duration_us']) <= D(second['device_start_us'])
            qdur, mdur = D(second['trace_duration_us']), D(mat['trace_duration_us'])
            assert min(qdur, mdur) > 0
            gap = D(mat['device_start_us']) - D(second['device_start_us']) - qdur
            assert gap >= 0
            groups[(digest, m)].append((qdur, mdur, qdur / (qdur + mdur), gap))

assert len(seen) == 256 and {k: len(v) for k, v in groups.items()} == expected
result = {}
for (digest, m), rows in sorted(groups.items()):
    ratio = [r[2] for r in rows]
    result[f'{digest}:{m}'] = {
        'scopes': len(rows), 'second_quant_median_us': str(median(r[0] for r in rows)),
        'matmul_median_us': str(median(r[1] for r in rows)),
        'quant_fraction_min_median_max': [str(min(ratio)), str(median(ratio)), str(max(ratio))],
        'selected_gap_min_median_max_us': [str(min(r[3] for r in rows)),
                                           str(median(r[3] for r in rows)), str(max(r[3] for r in rows))],
    }
print(json.dumps({'source_sha256': sha, 'groups': result,
                  'limits': 'Selected trace durations only; no tensor identity, critical-path or savings proof.'}, indent=2))
