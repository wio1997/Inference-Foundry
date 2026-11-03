#!/usr/bin/env python3
"""Recompute only existing draft bracket shape sequences; no device access."""
import collections
import hashlib
import json
from pathlib import Path
import re

path = Path(__file__).with_name('16rank-draft-alltoall-distribution.json')
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '633011028e6d5953f03f1493c197085bdd1a810c095cc646e4f5b0e115f61ec6'
data = json.loads(raw)
assert not data['issues']
groups = collections.defaultdict(list)
counts = collections.Counter()
for row in data['brackets']:
    rank = int(re.search(r'_ep(\d+)_', row['rank']).group(1))
    shape = row['mc2_input_shape'].strip('"').split(';')
    m, k = map(int, shape[0].split(','))
    token_rows, topk = map(int, shape[3].split(','))
    assert k == 5120 and m == token_rows and topk == 3
    groups[(rank, row['gap_ordinal'])].append((row['bracket_start_ns'], m))
    counts[(row['gap_ordinal'], m, row['status'])] += 1
assert set(groups) == {(rank, gap) for rank in range(16) for gap in (1, 2)}
for gap in (1, 2):
    seq = collections.Counter(tuple(m for _, m in sorted(groups[rank, gap])) for rank in range(16))
    print(f'gap={gap}: {dict(seq)}')
for key, n in sorted(counts.items()):
    print(*key, n)
