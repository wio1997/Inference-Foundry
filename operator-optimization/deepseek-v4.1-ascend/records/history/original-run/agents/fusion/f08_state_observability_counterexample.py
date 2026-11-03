#!/usr/bin/env python3
"""CPU-only counterexample for the saved F08 scatter's structural metadata.

This checks hypothetical states, not the observed tensor values or profiler timing.
"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / 'agents/fusion/f08-ep8-tensor-provenance.json'
raw = source.read_bytes()
data = json.loads(raw)
rows = [r for r in data['task_rows'] if r['task']['taskId'] == 13276]
assert len(rows) == 1
r = rows[0]
assert r['task']['deviceId'] == 8 and r['task']['streamId'] == 35
assert r['decoded']['opType'] == 'ScatterNdUpdateSk'
assert r['decoded']['inputShapes'] == '"17314,128,512;8180,2;8180,512"'
assert r['decoded']['inputDataTypes'] == 'DT_BF16;INT32;DT_BF16'
pages, slots_per_page, n, width = 17314, 128, 8180, 512
slots = [(i // slots_per_page, i % slots_per_page) for i in range(n)]
assert len(set(slots)) == n
assert all(0 <= p < pages and 0 <= s < slots_per_page for p, s in slots)
# Positive finite BF16 bits 0x0101, zero BF16 target. Both bytes differ.
target = bytes(2 * width * n)
initial_same = target
initial_different = bytes([1, 1]) * (width * n)
changed_same = sum(a != b for a, b in zip(initial_same, target))
changed_different = sum(a != b for a, b in zip(initial_different, target))
assert changed_same == 0 and changed_different == 8376320
print(json.dumps({'source_sha256': hashlib.sha256(raw).hexdigest(),
 'task': 13276, 'n_distinct_valid_slots': len(slots),
 'pages_used': len({p for p, _ in slots}),
 'target_bytes': len(target), 'changed_bytes_same': changed_same,
 'changed_bytes_different': changed_different,
 'scope': 'hypothetical interface states, not observed historical values or timing'},
 sort_keys=True, indent=2))
