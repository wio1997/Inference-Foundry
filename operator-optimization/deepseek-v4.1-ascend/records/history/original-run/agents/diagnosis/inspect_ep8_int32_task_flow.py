#!/usr/bin/env python3
"""Inspect exact timestamp only in one previously saved trace; no device access."""
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

p = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json')
target_ns = 1793152960209278701
h = hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
        h.update(chunk)
assert h.hexdigest() == '3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99'
before = p.stat()
decoder = json.JSONDecoder(parse_float=Decimal)
items = []
count = 0
with p.open('r', encoding='utf-8') as f:
    buf, pos, started, eof = f.read(1048576), 0, False, False
    while True:
        while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ','):
            pos += 1
        if not started:
            assert pos < len(buf) and buf[pos] == '['
            pos += 1
            started = True
            continue
        if pos >= len(buf):
            if eof:
                raise RuntimeError('unexpected end of trace')
            buf, pos = f.read(1048576), 0
            eof = not buf
            continue
        if buf[pos] == ']':
            break
        try:
            obj, end = decoder.raw_decode(buf, pos)
        except json.JSONDecodeError:
            remaining, more = buf[pos:], f.read(1048576)
            if not more or len(remaining) + len(more) > 4194304:
                raise RuntimeError('truncated or oversized trace item')
            buf, pos = remaining + more, 0
            continue
        pos = end
        count += 1
        if not isinstance(obj, dict) or obj.get('ph') not in ('s', 't', 'f', 'X'):
            continue
        try:
            if Decimal(str(obj.get('ts'))) * 1000 != target_ns:
                continue
        except (InvalidOperation, ValueError, TypeError):
            continue
        args = obj.get('args') if isinstance(obj.get('args'), dict) else {}
        name = obj.get('name') if isinstance(obj.get('name'), str) else ''
        cat = obj.get('cat') if isinstance(obj.get('cat'), str) else ''
        items.append({'phase': obj.get('ph'), 'pid': obj.get('pid'), 'tid': obj.get('tid'),
                      'flow_id': str(obj.get('id')) if isinstance(obj.get('id'), (int, str)) else None,
                      'is_torch_to_npu': 'torch_to_npu' in (name + cat).lower(),
                      'name_is_quant_matmul': 'QuantBatchMatmulV3' in name,
                      'args': {k: args[k] for k in ('Model Id', 'Task Id', 'Physic Stream Id', 'connection_id') if k in args}})
        if len(items) > 100:
            raise RuntimeError('unexpectedly many exact-time records')
after = p.stat()
assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
print(json.dumps({'trace_sha256': h.hexdigest(), 'target_ns': target_ns, 'scanned_items': count,
                  'matched_items': items,
                  'limits': ['Exact timestamp alone does not bind flow endpoint to a task track.',
                             'Only exact timestamp and selected trace fields are inspected; missing endpoints or other IDs are not excluded.',
                             'Trace hash matches prior recorded value; this is not an independent frozen-trace hash.']}, ensure_ascii=False, indent=2, default=str))
