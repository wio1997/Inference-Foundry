#!/usr/bin/env python3
"""Retain actual per-TP final-PUT drain rows for the immediate-handoff test."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--run-id', default='final-put-producer-v1')
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.run_id)
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    test = json.loads((base/'tests/final-put-immediate-control-v1'/('progress.json' if args.partial else 'result.json')).read_text())
    assert args.partial or test['completed']
    root = base/'pd-node32-v12'
    cid = json.loads((root/'launch.json').read_text())['container_id']
    out = base/'consumer-audits'/args.run_id
    out.mkdir(exist_ok=False)
    logs = subprocess.run(['docker', 'logs', '--timestamps', cid], capture_output=True, timeout=45)
    assert logs.returncode == 0
    raw = logs.stdout+b'\nDOCKER_STDERR_STREAM\n'+logs.stderr
    (out/'console.txt.gz').write_bytes(gzip.compress(raw, compresslevel=1))
    drains = []
    errors = []
    for line in raw.decode(errors='replace').splitlines():
        match = re.search(r'Final prompt PUT batch drained before handoff tp_rank=(\d+) requests=([^ ]+) wait_ms=([\d.]+)', line)
        if match:
            drains.append({'tp_rank': int(match[1]), 'requests': match[2].split(','), 'wait_ms': float(match[3]), 'line': line})
        if 'ERROR' in line or 'Traceback' in line:
            errors.append(line)
    joined = []
    for wave in test['waves']:
        rows = [row for row in drains if any(request.startswith(wave['producer_id']+'-') for request in row['requests'])]
        joined.append({'wave': wave['wave'], 'producer_id': wave['producer_id'],
                       'tp_ranks_with_final_prompt_drain': sorted({row['tp_rank'] for row in rows}),
                       'wait_ms_per_tp': {str(row['tp_rank']): row['wait_ms'] for row in rows}})
    with urllib.request.urlopen('http://172.16.12.32:29481/metrics', timeout=10) as response:
        metrics = response.read()
    (out/'metrics.txt').write_bytes(metrics)
    report = {'container_id': cid, 'partial_observation': args.partial, 'waves': joined, 'drain_rows': drains, 'native_error_lines': errors,
              'hybm_data_op_types': [int(x) for x in re.findall(rb'Initialize success:[^\n]*bmDataOpType: (\d+)', raw)],
              'raw_console_bytes': len(raw), 'raw_console_sha256': hashlib.sha256(raw).hexdigest(),
              'scope': 'A drain row means existing FIFO batch processing completed. It is not a durable write-success acknowledgement; separate SDK error and consumer-hit audits are required. Host clocks differ across nodes; no cross-node timestamp subtraction.'}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
