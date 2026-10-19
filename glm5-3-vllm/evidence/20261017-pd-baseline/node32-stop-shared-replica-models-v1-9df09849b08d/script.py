#!/usr/bin/env python3
"""Stop only the completed diagnostic instances and retain their final lifecycle logs."""
import gzip
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def main():
    node = {'B4-33U-Atlas900RCKA3-32': 32, 'B5-5U-Atlas900RCKA3-33': 33}[socket.gethostname()]
    rev = 9 if node == 32 else 8
    root = Path(f'/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/pd-node{node}-v{rev}')
    cid = json.loads((root/'launch.json').read_text())['container_id']
    doc = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
    assert doc['Name'] == f'/glm53-memcache-baseline-{node}-v{rev}-20261017'
    assert doc['Image'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    if doc['State']['Running']:
        subprocess.run(['docker', 'stop', '-t', '90', cid], check=True, capture_output=True, timeout=100)
    state = json.loads(subprocess.check_output(['docker', 'inspect', cid, '--format', '{{json .State}}']))
    logs = subprocess.run(['docker', 'logs', '--timestamps', cid], capture_output=True, timeout=30)
    raw = logs.stdout+b'\nDOCKER_STDERR_STREAM\n'+logs.stderr
    path = root/'pre-rank-preservation-closed-console.txt.gz'
    assert not path.exists()
    path.write_bytes(gzip.compress(raw, compresslevel=1))
    report = {'node': node, 'container_id': cid, 'state': state,
              'reason': 'Completed shared-replica baseline and ABBA diagnostics; restart only these owned models for per-TP KV ownership experiment.',
              'raw_console_bytes': len(raw), 'raw_console_sha256': hashlib.sha256(raw).hexdigest(),
              'compressed_console': str(path), 'compressed_bytes': path.stat().st_size,
              'compressed_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    archive = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/consumer-audits')/f'node{node}-pre-rank-lifecycle'
    archive.mkdir(exist_ok=False)
    (archive/'console.txt.gz').write_bytes(path.read_bytes())
    (archive/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    (root/'pre-rank-preservation-stop.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert not state['Running'] and not state['OOMKilled'] and state['ExitCode'] in (0, 143)


if __name__ == '__main__':
    main()
