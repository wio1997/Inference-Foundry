#!/usr/bin/env python3
"""Inspect fixed pre-existing trace only; time/stream proximity is not dependency."""
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import ijson

p = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json')
expected = '53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23'
h = hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda: f.read(1048576), b''):
        h.update(chunk)
assert h.hexdigest() == expected
host_t = Decimal('1793241327169187.740')
dev_t = Decimal('1793241328465293.999')
host = []
device = []
starts = {}
ends = []
flow_counts = {}
with p.open('rb') as f:
    for e in ijson.items(f, 'item'):
        if not isinstance(e, dict) or e.get('ts') is None:
            continue
        ts = Decimal(str(e['ts']))
        pid, tid, ph = str(e.get('pid')), str(e.get('tid')), e.get('ph')
        if ph == 'X' and pid == '737' and tid == '737' and abs(ts-host_t) <= 1500:
            host.append(e)
        if ph == 'X' and pid == '755144' and tid == '47' and abs(ts-dev_t) <= 500:
            device.append(e)
        if e.get('name') == 'torch_to_npu' and ph in ('s', 'f'):
            key = (str(e.get('id')), ph)
            flow_counts[key] = flow_counts.get(key, 0) + 1
        if e.get('name') == 'torch_to_npu' and ph == 's':
            starts[str(e.get('id'))] = e
        if e.get('name') == 'torch_to_npu' and ph == 'f' and pid == '755144' and tid == '47' and abs(ts-dev_t) <= 500:
            ends.append(e)

device.sort(key=lambda x: Decimal(str(x['ts'])))
host.sort(key=lambda x: Decimal(str(x['ts'])))
flows = [{ 'start': starts.get(str(e.get('id'))), 'end': e,
           's_count_global': flow_counts.get((str(e.get('id')), 's'), 0),
           'f_count_global': flow_counts.get((str(e.get('id')), 'f'), 0)} for e in ends]
print(json.dumps({'trace_sha256': expected, 'host_t': str(host_t), 'dev_t': str(dev_t), 'host': host, 'device': device, 'flows': flows}, default=str, indent=2))
