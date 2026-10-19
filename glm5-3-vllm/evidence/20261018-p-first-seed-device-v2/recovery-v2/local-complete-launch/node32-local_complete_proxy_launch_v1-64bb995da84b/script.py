#!/usr/bin/env python3
"""Launch one finite CPU-only proxy attached to an exact live owned P."""
import argparse
import ast
import base64
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request
import zlib


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--expected-P-CID', required=True)
    p.add_argument('--source-base64', required=True)
    p.add_argument('--expected-source-sha256', required=True)
    a = p.parse_args()
    base = a.run_dir.resolve()
    assert str(base).startswith('/data/wio/glm5-3-vllm-execution/20261018-state-tail/') and base.name == 'node32'
    assert re.fullmatch('[0-9a-f]{64}', a.expected_P_CID)
    plan = json.loads((base / 'plan.json').read_text())
    cid = json.loads((base / 'launch.json').read_text())['container_id']
    ins = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
    assert ins['Id'] == a.expected_P_CID and ins['State']['Running']
    assert ins['Name'] == '/' + plan['container']['name']
    assert ins['Image'] == plan['image']['id'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    started = datetime.datetime.fromisoformat(ins['State']['StartedAt'][:26] + '+00:00')
    remaining = int(plan['container']['lifetime_seconds'] - (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() - 180)
    assert 900 < remaining < 14400
    source = zlib.decompress(base64.b64decode(a.source_base64, validate=True))
    assert len(source) < 100000 and hashlib.sha256(source).hexdigest() == a.expected_source_sha256
    ast.parse(source.decode())
    name = 'glm53-local-complete-proxy-32-' + base.parent.name
    assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
    with socket.socket() as sock:
        sock.bind(('172.16.12.32', 29882))
    root = base / 'local-complete-proxy-v1'
    root.mkdir(exist_ok=False)
    (root / 'proxy.py').write_bytes(source)
    argv = ['docker', 'run', '-d', '--pull', 'never', '--name', name, '--network', 'host',
            '--read-only', '--tmpfs', '/tmp:rw,size=128m', '--cpus', '4', '--memory', '4g', '--pids-limit', '256',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-e', 'HF_HUB_OFFLINE=1', '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
            '-v', str(root / 'proxy.py') + ':/proxy.py:ro', '--entrypoint', '/usr/bin/timeout', plan['image']['id'],
            '--signal=TERM', '--kill-after=15', str(remaining), 'python3', '/proxy.py',
            '--host', '172.16.12.32', '--port', '29882', '--prefiller-hosts', '172.16.12.32', '--prefiller-ports', '29481',
            '--decoder-hosts', '172.16.12.33', '--decoder-ports', '29482']
    proxy_cid = subprocess.check_output(argv, text=True, timeout=30).strip()
    report = {'CID': proxy_cid, 'argv': argv, 'parent_P_CID': cid, 'source_sha256': a.expected_source_sha256,
              'lifetime_seconds': remaining, 'ready': False, 'compute_devices': [],
              'scope': 'Separate29882; original29880/29881 unchanged. Explicit p-complete header and strict max1 only; native handler fallback. No model reload or new operator.'}
    (root / 'launch.json').write_text(json.dumps(report, indent=2) + '\n')
    for _ in range(45):
        try:
            with urllib.request.urlopen('http://172.16.12.32:29882/healthcheck', timeout=2) as response:
                report['health'] = json.load(response)
            report['ready'] = True
            break
        except Exception:
            time.sleep(1)
    if not report['ready']:
        report['stop_on_failure'] = subprocess.run(['docker', 'stop', '-t', '15', proxy_cid], capture_output=True, text=True).returncode
    (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GLM53_RESULT_JSON_BEGIN\n' + json.dumps(report) + '\nGLM53_RESULT_JSON_END')
    assert report['ready']


if __name__ == '__main__':
    main()
