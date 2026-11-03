#!/usr/bin/env python3
"""Read only the allowed device visibility value in two hash-frozen profile DBs."""
import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--task-root', required=True)
parser.add_argument('--trace-root', required=True)
a = parser.parse_args()
task = Path(a.task_root).resolve(strict=True)
trace = Path(a.trace_root).resolve(strict=True)
manifest = task / 'evidence/used-raw-source-sha256.json'
manifest_hash = 'cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a'
selector = ('dp0_pp0_tp0_dcp0_ep0_rank0_728_20261028020238584_ascend_pt',
            'dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt')
target = 'ASCEND_RT_VISIBLE_DEVICES'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

if sha(manifest) != manifest_hash:
    raise RuntimeError('manifest hash mismatch')
entries = json.loads(manifest.read_text())['records']
out = []
for profile in selector:
    found = [x for x in entries if x['relative_path'].startswith(profile+'/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_') and x['relative_path'].endswith('.db')]
    if len(found) != 1:
        raise RuntimeError('ambiguous DB manifest')
    rel = Path(found[0]['relative_path'])
    if rel.is_absolute() or '..' in rel.parts:
        raise RuntimeError('invalid DB path')
    db = (trace / rel).resolve(strict=True)
    if not db.is_relative_to(trace) or not db.is_file() or sha(db) != found[0]['sha256']:
        raise RuntimeError('invalid DB location or hash')
    connection = sqlite3.connect(db.as_uri()+'?mode=ro', uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        rows = connection.execute('SELECT name,value FROM META_DATA LIMIT 101').fetchall()
    finally:
        connection.close()
    if len(rows) > 100:
        raise RuntimeError('metadata row bound exceeded')
    vals = []
    other_env_detected = False
    inspected_nodes = 0
    depth_truncated = False
    def record(v):
        # Only a short, unambiguous comma-separated list of decimal IDs is output.
        if isinstance(v, str) and len(v) <= 255 and re.fullmatch(r'[0-9]+(?:,[0-9]+)*', v):
            ids = [int(x) for x in v.split(',')]
            if 1 <= len(ids) <= 32 and all(0 <= x <= 65535 for x in ids):
                vals.append({'status': 'VALIDATED_NUMERIC_LIST', 'ids': ids})
                return
        vals.append({'status': 'PRESENT_UNSUPPORTED_OR_INVALID'})
    def visit(obj, depth=0):
        global inspected_nodes, depth_truncated, other_env_detected
        inspected_nodes += 1
        if inspected_nodes > 10000:
            raise RuntimeError('metadata node bound exceeded')
        if depth > 10:
            depth_truncated = True
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == target:
                    record(value)
                elif isinstance(key, str) and (key.startswith('ASCEND_') or key.startswith('NPU_')):
                    other_env_detected = True
                if isinstance(value, (dict, list)):
                    visit(value, depth+1)
        elif isinstance(obj, list):
            if len(obj) > 128:
                raise RuntimeError('metadata array bound exceeded')
            for value in obj:
                if isinstance(value, (dict, list)):
                    visit(value, depth+1)
    for name, value in rows:
        if not isinstance(name, str) or not isinstance(value, str) or len(name) > 256 or len(value) > 1048576:
            raise RuntimeError('unexpected metadata type or size')
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if name == target:
            record(parsed if isinstance(parsed, str) else value)
        elif name.startswith('ASCEND_') or name.startswith('NPU_'):
            other_env_detected = True
        if isinstance(parsed, (dict, list)):
            visit(parsed)
    if len(vals) > 20:
        raise RuntimeError('too many matches')
    out.append({'profile': profile, 'db_sha256': found[0]['sha256'],
                'target_matches': vals, 'target_occurrences': len(vals),
                'other_ascend_or_npu_environment_key_present': other_env_detected,
                'json_depth_truncated': depth_truncated})
print(json.dumps({'scope': 'exact allowlisted value from EP0 and EP8 META_DATA only',
    'manifest_sha256': manifest_hash, 'samples': out,
    'limits': ['empty result does not prove variable unset',
      'metadata capture and producer version/timing are not established',
      'deviceId versus npu-smi Chip Logic ID is not assumed',
      'other environment names and values are not emitted', 'no NPU operations']}, ensure_ascii=False, indent=2))
