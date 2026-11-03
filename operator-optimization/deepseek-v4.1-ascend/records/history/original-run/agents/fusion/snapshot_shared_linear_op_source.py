#!/usr/bin/env python3
"""Snapshot the exact stage-container linear_op.py source; never import NPU packages."""
import base64
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

T = Path(__file__).resolve().parents[2]
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINER = 'wio-vllm-v41-repro169-stage-20261028'
IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
ROOT = '/vllm-workspace/vllm-ascend/vllm_ascend/ops'
FILES = ['linear_op.py']
LIMIT = 200000
CODE = r'''import base64, hashlib, json
from pathlib import Path
root = Path(%r)
files = %r
limit = %d
out = []
for name in files:
    path = root / name
    if path.resolve(strict=True) != path or not path.is_file():
        raise SystemExit('symlink or invalid file: ' + str(path))
    size = path.stat().st_size
    if size > limit:
        raise SystemExit('oversize file: ' + str(path))
    data = path.read_bytes()
    if len(data) != size:
        raise SystemExit('file changed while reading: ' + str(path))
    out.append({'path': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'data': base64.b64encode(data).decode('ascii')})
print(json.dumps(out))
''' % (ROOT, FILES, LIMIT)

def remote(args):
    result = subprocess.run([SSH, shlex.join(args)], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError('read-only query failed, exit ' + str(result.returncode))
    return result.stdout

if remote(['docker', 'inspect', '--format', '{{.Image}}', CONTAINER]).strip() != IMAGE:
    raise RuntimeError('unexpected image')
mounts = json.loads(remote(['docker', 'inspect', '--format', '{{json .Mounts}}', CONTAINER])) or []
for mount in mounts:
    dest = mount.get('Destination', '').rstrip('/') or '/'
    for name in FILES:
        source = ROOT + '/' + name
        if dest == '/' or source == dest or source.startswith(dest + '/'):
            raise RuntimeError('source path overmounted: ' + source)
entries = json.loads(remote(['docker', 'exec', CONTAINER, 'python3', '-c', CODE]))
if [entry['path'] for entry in entries] != FILES:
    raise RuntimeError('unexpected file list')
records = []
for entry in entries:
    content = base64.b64decode(entry['data'], validate=True)
    digest = hashlib.sha256(content).hexdigest()
    if digest != entry['sha256'] or len(content) != entry['size'] or len(content) > LIMIT:
        raise RuntimeError('transport hash or size mismatch')
    target = T / 'agents/fusion/frozen-source/vllm_ascend/ops' / entry['path']
    if target.is_symlink():
        raise RuntimeError('snapshot is symlink')
    if target.exists() and target.read_bytes() != content:
        raise RuntimeError('snapshot differs; no overwrite')
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(content)
    records.append({'source': ROOT + '/' + entry['path'], 'snapshot': str(target.relative_to(T)), 'size': len(content), 'sha256': digest})
print(json.dumps({'status': 'SOURCE_SNAPSHOTS_VERIFIED', 'container': CONTAINER, 'image_id': IMAGE, 'source_paths_overmounted': False, 'files': records, 'no_device_work': True, 'historical_binary_binding': 'UNKNOWN'}, indent=2))
