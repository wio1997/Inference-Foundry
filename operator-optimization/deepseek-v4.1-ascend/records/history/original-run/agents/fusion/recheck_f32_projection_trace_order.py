#!/usr/bin/env python3
"""Read only: selected EP-local device-export times, not dependency inference."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[2]
inputs = {
    'rows': ('agents/fusion/f32-ep0-7-projection-rows.json', '87fa9a16d2acad858fe1da3f216915c1497bd1964a28671d39883386d400e831'),
    'flows': ('agents/fusion/f32-ep3-layer36-device-flows.json', 'd8bae5c11c49f16d2c305fc6b23a6667766a1356131b7dcafd01b8a488774021'),
    'scope': ('agents/fusion/f32-ep3-layer36-scope.json', '210a74e86892d4a11c0b127eb16149fbb1c2256c90e7272c1a15d63c88be2b84'),
}
data = {}
for key, (name, expected) in inputs.items():
    b = (root / name).read_bytes()
    assert hashlib.sha256(b).hexdigest() == expected
    data[key] = json.loads(b)
rows = data['rows']['rows']
assert len(rows) == 8 and {r['ep'] for r in rows} == set(range(8))
gaps = []
for row in sorted(rows, key=lambda r: r['ep']):
    q, kv = row['tasks']['q_shape'], row['tasks']['kv_shape']
    assert row['trace_scope_count'] == 1 and row['sidecar_thread_matches_trace']
    m = 1019 if row['ep'] == 7 else 1023
    assert q['input_shape'] == [m, 5120] and q['output_shape'] == [m, 1280]
    assert kv['input_shape'] == [8180, 5120] and kv['output_shape'] == [8180, 512]
    assert q['stream'] != kv['stream']
    gap = Decimal(kv['device_start_us']) - Decimal(q['device_start_us']) - Decimal(q['duration_us'])
    gaps.append(gap)
    print('ep', row['ep'], 'selected_export_gap_us', gap, 'streams', q['stream'], kv['stream'])
    if row['ep'] == 3:
        for item, conn, gid in ((q, 102408, 3162), (kv, 102416, 3163)):
            matches = [t for t in data['flows']['tasks'] if t['connectionId'] == conn and t['globalTaskId'] == gid]
            assert len(matches) == 1
            task = matches[0]
            assert Decimal(item['device_start_us']) * 1000 == task['startNs']
            assert Decimal(item['duration_us']) * 1000 == task['endNs'] - task['startNs']
        q_db = next(t for t in data['flows']['tasks'] if t['connectionId'] == 102408 and t['globalTaskId'] == 3162)
        op = [t for t in data['scope']['communication_ops'] if t['connectionId'] == 102373 and t['opId'] == 246]
        assert len(op) == 1 and op[0]['deviceId'] == 3
        print('ep3_selected_op_end_to_q_start_us', Decimal(q_db['startNs'] - op[0]['endNs']) / 1000)
assert all(gap > 0 for gap in gaps)
print('gaps_us_min_median_max', min(gaps), statistics.median(gaps), max(gaps))
