#!/usr/bin/env python3
"""Bounded read-only resource sampling for the task-owned PD model container."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--duration', type=int, default=3600)
    parser.add_argument('--interval', type=int, default=5)
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.run_id)
    node = {'B4-33U-Atlas900RCKA3-32': 32, 'B5-5U-Atlas900RCKA3-33': 33}[socket.gethostname()]
    rev = 9 if node == 32 else 8
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    out = base/'resource-samples'/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    cid = json.loads((base/f'pd-node{node}-v{rev}'/'launch.json').read_text())['container_id']
    start = time.monotonic()
    count = 0
    with (out/'samples.jsonl').open('x') as stream:
        while time.monotonic()-start < args.duration and not (out/'STOP').exists():
            before = time.monotonic()
            row = {'node': node, 'host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'monotonic_ns': time.monotonic_ns()}
            for key, command in [('npu_smi', ['npu-smi', 'info']),
                                 ('docker_stats', ['docker', 'stats', '--no-stream', '--format', '{{json .}}', cid])]:
                p = subprocess.run(command, capture_output=True, text=True, timeout=20)
                row[key] = {'rc': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
            row['proc_stat'] = Path('/proc/stat').read_text().splitlines()[0]
            row['memory'] = [s for s in Path('/proc/meminfo').read_text().splitlines()
                             if s.startswith(('MemTotal:', 'MemAvailable:', 'Cached:', 'SwapFree:'))]
            row['diskstats'] = Path('/proc/diskstats').read_text()
            row['netdev'] = Path('/proc/net/dev').read_text()
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            count += 1
            time.sleep(max(.05, args.interval-(time.monotonic()-before)))
    raw = (out/'samples.jsonl').read_bytes()
    report = {'node': node, 'samples': count, 'interval_seconds': args.interval,
              'elapsed_seconds': time.monotonic()-start, 'path': str(out/'samples.jsonl'),
              'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
              'scope': 'CPU-only read/query sampling of the existing model; no compute device context or inference launched. Same sampler settings apply to baseline and mechanism tests.'}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
