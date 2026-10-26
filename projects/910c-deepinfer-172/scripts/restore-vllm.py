#!/usr/bin/env python3
"""Retain baseline identity, stop only this native service, then restore vLLM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time

ROOT = Path('/data/tiankuan/wio/910c-deepinfer-172')
PACKAGE = Path(__file__).resolve().parents[1]


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    identity = json.loads((PACKAGE / 'config/vllm-restore-identity.json').read_text())
    print(json.dumps({'mode': 'execute' if args.execute else 'plan',
                      'original_container': identity['name'], 'id': identity['id'],
                      'steps': ['check occupation and identities', 'back up original logs before redirection',
                                'SIGTERM only owned native torchrun/API parents',
                                'require all devices idle', 'docker start retained original container']}, indent=2))
    if not args.execute:
        return
    assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-172', 'wrong server'
    original = inspect(identity['name'])
    assert original['Id'] == identity['id'] and original['Image'] == identity['image_id'], 'baseline identity changed'
    if original['State']['Running']:
        print('Original vLLM container already running; retained. Inspect health separately.')
        return
    own_pids = set()
    parents = []
    configs = json.loads((PACKAGE / 'config/container-manifest.json').read_text())
    for name, needle in [('wio-deepinfer-s172-npu', 'torch.distributed.run --standalone --nproc-per-node=16 examples/ascend910c_w8a8_worker.py'),
                         ('wio-deepinfer-s172', 'python examples/ascend910c_w8a8_api.py')]:
        native = inspect(name)
        assert native['Image'] == configs[name]['image_id'], 'native identity changed'
        if not native['State']['Running']:
            continue
        rows = subprocess.check_output(['docker', 'top', name, '-eo', 'pid,args'], text=True).splitlines()[1:]
        own_pids.update(int(row.split(None, 1)[0]) for row in rows)
        selected = [row for row in rows if needle in row and str(ROOT / 'serving') + '/' in row]
        assert len(selected) <= 1, 'multiple native services: inspect manually'
        parents += [int(row.split(None, 1)[0]) for row in selected]
    smi = subprocess.check_output(['npu-smi', 'info'], text=True)
    device_pids = {int(m.group(1)) for m in re.finditer(r'^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|', smi, re.M)}
    assert 'Process id' in smi, 'unknown NPU process table'
    assert device_pids <= own_pids, 'foreign NPU task; refuse to stop or launch'
    assert not device_pids or len(parents) == 2, 'owned workers without a unique native service: inspect manually'
    backup = ROOT / 'baseline' / ('before-restore-' + time.strftime('%Y%m%dT%H%M%S'))
    backup.mkdir(parents=True, exist_ok=False)
    (backup / 'container-command.json').write_text(json.dumps({key: original['Config'][key] for key in ['Entrypoint', 'Cmd', 'WorkingDir']}, indent=2))
    baseline_root = Path(identity['baseline_root'])
    # The retained container redirects to the original log with > on every start.
    log_paths = set()
    command = ' '.join(original['Config'].get('Cmd') or [])
    for match in re.finditer(r">\s*['\"]?(/data/tiankuan/wio/[^\s'\"]+)", command):
        log_paths.add(Path(match.group(1)))
    log_paths.update([baseline_root / 'logs/current.log'])
    marker = baseline_root / 'runs/log-path.txt'
    if marker.exists():
        log_paths.add(Path(marker.read_text().strip()))
    hashes = {}
    for path in log_paths:
        if path.exists():
            resolved = path.resolve()
            relative = resolved.relative_to(baseline_root)
            target = backup / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, target)
            hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    assert hashes, 'no original log backed up; refuse destructive redirection'
    (backup / 'SHA256.json').write_text(json.dumps(hashes, indent=2))
    for pid in parents:
        os.kill(pid, signal.SIGTERM)
    for attempt in range(45):
        smi = subprocess.check_output(['npu-smi', 'info'], text=True)
        port = subprocess.check_output(['ss', '-ltnH', 'sport = :8172'], text=True)
        if all(f'No running processes found in NPU {i}' in smi for i in range(8)) and not port.strip():
            break
        time.sleep(1)
    else:
        raise RuntimeError('devices/port not idle after native stop; no forced kill or vLLM launch')
    assert not subprocess.check_output(['ss', '-ltnH', 'sport = :8172'], text=True).strip(), 'port still occupied'
    subprocess.run(['docker', 'start', identity['id']], check=True)
    print('Original vLLM container started; this is not a health or generation verdict. Backup:', backup)


if __name__ == '__main__':
    main()
