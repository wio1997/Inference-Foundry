#!/usr/bin/env python3
"""Plan a fresh native run; --execute retains an already healthy service."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/data/tiankuan/wio/910c-deepinfer-172')
PACKAGE = Path(__file__).resolve().parents[1]
HOST = 'bm-jn-zs-zone1-910C-64G-10-172'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--run-id', default='native-' + time.strftime('%Y%m%dT%H%M%S'))
    args = parser.parse_args()
    assert re.fullmatch(r'native-[A-Za-z0-9_-]+', args.run_id), 'invalid run id'
    control = ROOT / 'serving' / args.run_id
    print(json.dumps({'host': HOST, 'control': str(control),
                      'mode': 'execute' if args.execute else 'plan',
                      'workers': 16, 'dp': 2, 'tp': 8, 'ep': 16,
                      'prompt_lookup_tokens': 63}, indent=2))
    if not args.execute:
        return
    assert socket.gethostname() == HOST, 'wrong server'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open('http://127.0.0.1:8172/health', timeout=3) as response:
            health = json.load(response)
    except Exception:
        health = None
    if health == {'status': 'ok', 'backend': 'DeepInfer-native'}:
        print('Existing healthy native service retained; no restart.')
        return
    port = subprocess.check_output(['ss', '-ltnH', 'sport = :8172'], text=True)
    assert not port.strip(), 'port occupied; inspect existing service'
    smi = subprocess.check_output(['npu-smi', 'info'], text=True)
    assert all(f'No running processes found in NPU {i}' in smi for i in range(8)), 'NPU occupied or idle status unknown'
    for name, expected in json.loads((PACKAGE / 'config/container-manifest.json').read_text()).items():
        actual = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
        assert actual['State']['Running'] and actual['Image'] == expected['image_id'], name
        assert actual['HostConfig']['ReadonlyRootfs'] and actual['HostConfig']['Privileged'] == expected['privileged'], name
        assert actual['HostConfig']['Runtime'] == 'ascend', name
        weights = [m for m in actual['Mounts'] if m['Destination'] == '/models/DeepSeek-V4.1-Flash-1017-w8a8']
        assert len(weights) == 1 and not weights[0]['RW'], 'weights must remain read-only'
    manifest = json.loads((PACKAGE / 'config/deployed-source-sha256.json').read_text())
    for relative, expected in manifest.items():
        assert hashlib.sha256((ROOT / 'src/DeepInfer' / relative).read_bytes()).hexdigest() == expected, relative
    assert not control.exists(), 'fresh control directory required'
    template = (PACKAGE / 'controllers/native_service_v8_job.py').read_text()
    # Only filenames/control paths change. Model commands and defaults remain as measured.
    template = template.replace('serving-v8', 'serving-' + args.run_id).replace("'serving/v8'", repr('serving/' + args.run_id))
    template = template.replace("'DEEPINFER_910C_MC2=1',", "'DEEPINFER_910C_MC2=1','-e','DEEPINFER_PROMPT_LOOKUP_TOKENS=63',")
    source_record = ROOT / 'evidence' / ('serving-' + args.run_id + '-source') / 'SHA256.json'
    source_record.parent.mkdir(parents=True, exist_ok=False)
    source_record.write_text(json.dumps(manifest, indent=2))
    launch = ROOT / 'tools' / ('launch-' + args.run_id + '.py')
    assert not launch.exists()
    launch.write_text(template)
    with (ROOT / 'logs' / ('controller-' + args.run_id + '.log')).open('x') as log:
        child = subprocess.Popen([sys.executable, str(launch)], stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'dispatched_host_pid': child.pid,
                      'status': str(ROOT / 'runs' / ('serving-' + args.run_id + '-status.json')),
                      'note': 'Dispatch is not readiness; inspect status and /health.'}, indent=2))


if __name__ == '__main__':
    main()
