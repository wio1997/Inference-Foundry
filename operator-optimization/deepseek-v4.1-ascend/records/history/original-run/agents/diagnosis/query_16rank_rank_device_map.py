#!/usr/bin/env python3
"""Read only rankId/deviceId from the 16 hash-frozen profiler databases."""
import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--task-root', required=True)
parser.add_argument('--trace-root', required=True)
args = parser.parse_args()
task = Path(args.task_root).resolve(strict=True)
trace = Path(args.trace_root).resolve(strict=True)
manifest = task / 'evidence/used-raw-source-sha256.json'
expected_manifest = 'cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

if sha(manifest) != expected_manifest:
    raise RuntimeError('frozen manifest SHA mismatch')
records = json.loads(manifest.read_text())['records']
selected = [r for r in records if '/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_' in r['relative_path'] and r['relative_path'].endswith('.db')]
if len(selected) != 16:
    raise RuntimeError('expected exactly 16 frozen DB entries')
result = []
ep_values = set()
for entry in sorted(selected, key=lambda r: r['relative_path']):
    relative = Path(entry['relative_path'])
    if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 3:
        raise RuntimeError('unexpected manifest path')
    match = re.fullmatch(r'dp\d+_pp\d+_tp\d+_dcp\d+_ep(\d+)_rank\d+_\d+_\d+_ascend_pt', relative.parts[0])
    if not match or relative.parts[1] != 'ASCEND_PROFILER_OUTPUT' or not re.fullmatch(r'ascend_pytorch_profiler_\d+\.db', relative.name):
        raise RuntimeError('unexpected DB path structure')
    ep = int(match.group(1))
    if ep in ep_values:
        raise RuntimeError('duplicate EP label in manifest')
    ep_values.add(ep)
    db = (trace / relative).resolve(strict=True)
    if not db.is_relative_to(trace) or not db.is_file():
        raise RuntimeError('database outside fixed trace root or not a file')
    actual = sha(db)
    if actual != entry['sha256']:
        raise RuntimeError('database SHA mismatch')
    connection = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        rows = connection.execute('SELECT rankId, deviceId FROM RANK_DEVICE_MAP LIMIT 33').fetchall()
    finally:
        connection.close()
    if len(rows) > 32:
        raise RuntimeError('unexpected rank-map row count')
    if any(type(rank) is not int or type(device) is not int for rank, device in rows):
        raise RuntimeError('noninteger rank map field')
    result.append({'profile': relative.parts[0], 'ep_label': ep,
                   'db_relative_path': str(relative), 'db_sha256': actual,
                   'rank_device_map': [{'rankId': rank, 'deviceId': device} for rank, device in rows]})
if ep_values != set(range(16)):
    raise RuntimeError('missing or unexpected EP labels')
print(json.dumps({'scope': 'exactly 16 frozen profiler DBs; SELECT rankId,deviceId only',
    'manifest_sha256': expected_manifest, 'samples': result,
    'limits': ['EP labels and rank IDs are not assumed interchangeable',
      'deviceId is not assumed to equal npu-smi Chip Logic ID or physical slot',
      'rank maps do not establish group membership, task identity, communication routes, or simultaneous execution',
      'empty or -1 fields mean unresolved; no NPU operations']}, ensure_ascii=False, indent=2))
