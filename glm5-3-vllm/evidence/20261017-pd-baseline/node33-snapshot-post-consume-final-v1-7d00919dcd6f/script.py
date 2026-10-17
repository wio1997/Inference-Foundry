#!/usr/bin/env python3
"""Freeze idle diagnostic controls and console without accessing model tensors."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=int, choices=(32, 33), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--deployment', choices=('payload-control', 'post-consume'), default='payload-control')
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.run_id)
    assert socket.gethostname() == {32: 'B4-33U-Atlas900RCKA3-32', 33: 'B5-5U-Atlas900RCKA3-33'}[args.node]
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    revision = {'payload-control': (11, 10), 'post-consume': (13, 12)}[args.deployment][args.node-32]
    source = base / f'pd-node{args.node}-v{revision}'
    launch = json.loads((source/'launch.json').read_text())
    cid = launch['container_id']
    state = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
    assert state['State']['Running']
    assert state['Image'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    with urllib.request.urlopen(f'http://172.16.12.{args.node}:{29481 if args.node == 32 else 29482}/metrics', timeout=10) as response:
        metrics = response.read()
    for line in metrics.decode().splitlines():
        if re.match(r'vllm:num_requests_(running|waiting)[{ ]', line):
            assert float(line.rsplit(' ', 1)[-1]) == 0, line
    files = sorted((source/'kv-control-audit').glob('*.json*'))
    assert len([p for p in files if p.name.startswith('layout-')]) == 16
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    time.sleep(1)
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out = base/'consumer-audits'/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    destination = out/'kv-control-audit'
    destination.mkdir()
    phases = {}
    for path in files:
        shutil.copyfile(path, destination/path.name)
        assert hashlib.sha256((destination/path.name).read_bytes()).hexdigest() == hashes[path.name]
        if path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                row = json.loads(line)
                phases[row['phase']] = phases.get(row['phase'], 0)+1
    console = subprocess.run(['docker', 'logs', '--timestamps', cid], capture_output=True, timeout=45)
    assert console.returncode == 0
    raw = console.stdout+b'\nDOCKER_STDERR_STREAM\n'+console.stderr
    (out/'console.txt.gz').write_bytes(gzip.compress(raw, compresslevel=1))
    (out/'metrics.txt').write_bytes(metrics)
    shutil.copyfile(source/'launch.json', out/'launch.json')
    errors = [line for line in raw.decode(errors='replace').splitlines() if 'ERROR' in line or 'Traceback' in line]
    report = {'node': args.node, 'source': str(source), 'container_id': cid, 'phases': phases,
              'files_sha256': hashes, 'native_error_lines': errors,
              'hybm_data_op_types': [int(x) for x in re.findall(rb'Initialize success:[^\n]*bmDataOpType: (\d+)', raw)],
              'scope': 'Idle stable JSON controls only; integer moments are not cryptographic KV payload hashes. Full console retained compressed.'}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
