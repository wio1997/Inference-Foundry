#!/usr/bin/env python3
"""Read only: compare selected F22 projection durations within each scope."""
import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / 'agents/diagnosis/ffn-crossshape-chain-pairs.json'
data = path.read_bytes()
assert hashlib.sha256(data).hexdigest() == '036f74db28a55a1329ab371bb3c65b6c2c2bebb90a97db76e2be93dd64e6b9ce'
rows = json.loads(data)['pairs']
assert len(rows) == 32
assert {(r['ep'], r['layer']) for r in rows} == {(e, l) for e in range(8) for l in (0, 12, 24, 36)}
for shape in ('595', '1023'):
    ratios = []
    for row in rows:
        first, second = row[shape]['projection_1'], row[shape]['projection_2']
        assert first['stream'] == second['stream']
        assert first['type'] == second['type'] == 'QuantBatchMatmulV3'
        assert first['signature']['Input Shapes'].startswith('"' + shape + ',5120;')
        assert first['signature']['Output Shapes'] == '"' + shape + ',4608"'
        assert second['signature']['Input Shapes'].startswith('"' + shape + ',2304;')
        assert second['signature']['Output Shapes'] == '"' + shape + ',5120"'
        a, b = Decimal(first['duration_us']), Decimal(second['duration_us'])
        assert a > 0 and b > 0
        assert Decimal(first['start_us']) + a <= Decimal(second['start_us'])
        ratios.append(a / b)
    print(shape, 'count', len(ratios), 'min/median/max', min(ratios), statistics.median(ratios), max(ratios), 'below 2', sum(x < 2 for x in ratios))
