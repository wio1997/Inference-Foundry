#!/usr/bin/env python3
"""Retain per-request external-hit evidence and native errors after timing ends."""
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
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.run_id)
    assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    out = base/'consumer-audits'/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    cid = json.loads((base/'pd-node33-v13/launch.json').read_text())['container_id']
    proc = subprocess.run(['docker', 'logs', '--timestamps', cid], capture_output=True, timeout=45)
    raw = proc.stdout+b'\nDOCKER_STDERR_STREAM\n'+proc.stderr
    (out/'console.txt.gz').write_bytes(gzip.compress(raw, compresslevel=1))
    rows = []
    errors = []
    types = []
    for line in raw.decode(errors='replace').splitlines():
        match = re.search(r'Reqid: ([^,]+), Total tokens (\d+), kvpool hit tokens: (\d+), need to load: (\d+)', line)
        if match:
            rows.append({'request_id': match[1], 'input_tokens': int(match[2]),
                         'pool_cached_tokens': int(match[3]), 'need_load_tokens': int(match[4]), 'line': line})
        if 'ERROR' in line or 'Traceback' in line:
            errors.append(line)
        match = re.search(r'Initialize success:.*bmDataOpType: (\d+)', line)
        if match:
            types.append(int(match[1]))
    with urllib.request.urlopen('http://172.16.12.33:29482/metrics', timeout=10) as response:
        metric_raw = response.read()
    (out/'metrics.txt').write_bytes(metric_raw)
    report = {'requests': rows, 'native_error_lines': errors, 'hybm_data_op_types': types,
              'raw_console_bytes': len(raw), 'raw_console_sha256': hashlib.sha256(raw).hexdigest(),
              'compressed_console': str(out/'console.txt.gz'),
              'compressed_sha256': hashlib.sha256((out/'console.txt.gz').read_bytes()).hexdigest(),
              'metrics_sha256': hashlib.sha256(metric_raw).hexdigest(),
              'scope': 'Join consumer_id prefixes to per-request hit/need-load rows after the benchmark; request output matching and SDK get counters provide separate checks.'}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
