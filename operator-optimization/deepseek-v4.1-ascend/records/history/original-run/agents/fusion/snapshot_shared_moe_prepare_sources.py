#!/usr/bin/env python3
"""For review: read only two fixed files from the fixed 169 stage container.

Never imports model/NPU packages or issues device operations.  Container data
is emitted only into this task's frozen-source folder after all checks pass.
"""
import base64
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINER = 'wio-vllm-v41-repro169-stage-20261028'
IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
ROOT = '/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe'
NAMES = ('shared_experts.py', 'prepare_finalize.py')
MAX_SIZE = 200_000

REMOTE_CODE = r'''import base64, hashlib, json, os, stat
from pathlib import Path
root = Path(%r)
names = %r
limit = %d
records = []
for name in names:
    path = root / name
    if path.resolve(strict=True) != path:
        raise SystemExit('source contains symlink: ' + str(path))
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > limit:
        raise SystemExit('source type or size invalid: ' + str(path))
    data = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(data) != before.st_size:
        raise SystemExit('source changed during read: ' + str(path))
    records.append({'name': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'data': base64.b64encode(data).decode('ascii')})
print(json.dumps(records))
''' % (ROOT, NAMES, MAX_SIZE)


def remote(args):
    result = subprocess.run([SSH, shlex.join(args)], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError('read-only query failed, exit ' + str(result.returncode))
    return result.stdout


def inspect(target):
    raw = remote(['docker', 'inspect', '--format', '{{.Id}}|{{.Image}}|{{json .Mounts}}', target]).strip()
    container_id, image_id, mount_json = raw.split('|', 2)
    if len(container_id) != 64 or any(c not in '0123456789abcdef' for c in container_id):
        raise RuntimeError('invalid container id')
    if image_id != IMAGE:
        raise RuntimeError('unexpected image')
    mounts = json.loads(mount_json) or []
    for mount in mounts:
        dest = mount.get('Destination', '').rstrip('/') or '/'
        for name in NAMES:
            source = ROOT + '/' + name
            if dest == '/' or source == dest or source.startswith(dest + '/'):
                raise RuntimeError('source path overmounted: ' + source)
    return container_id, image_id, mounts


before = inspect(CONTAINER)
entries = json.loads(remote(['docker', 'exec', before[0], 'python3', '-c', REMOTE_CODE]))
after = inspect(before[0])
if before != after:
    raise RuntimeError('container image or mounts changed during snapshot')
if [item.get('name') for item in entries] != list(NAMES):
    raise RuntimeError('unexpected file list')

verified = []
for item in entries:
    content = base64.b64decode(item['data'], validate=True)
    digest = hashlib.sha256(content).hexdigest()
    if not content or len(content) > MAX_SIZE or len(content) != item['size'] or digest != item['sha256']:
        raise RuntimeError('transport hash or size mismatch')
    verified.append((item['name'], content, digest))

target_dir = TASK / 'agents/fusion/frozen-source/vllm_ascend/ops/fused_moe'
current = TASK
for component in target_dir.relative_to(TASK).parts:
    current = current / component
    if current.is_symlink():
        raise RuntimeError('snapshot directory contains symlink')
    current.mkdir(exist_ok=True)
if target_dir.resolve() != target_dir:
    raise RuntimeError('snapshot directory path changed')
for name, content, digest in verified:
    target = target_dir / name
    if target.is_symlink() or (target.exists() and target.read_bytes() != content):
        raise RuntimeError('snapshot target differs or is symlink; no overwrite')
for name, content, digest in verified:
    target = target_dir / name
    if not target.exists():
        with target.open('xb') as handle:
            handle.write(content)

print(json.dumps({
    'status': 'SOURCE_SNAPSHOTS_VERIFIED',
    'container': CONTAINER,
    'container_id': before[0],
    'image_id': before[1],
    'source_paths_overmounted': False,
    'files': [{'source': ROOT + '/' + n, 'snapshot': str((target_dir / n).relative_to(TASK)), 'size': len(c), 'sha256': h} for n, c, h in verified],
    'no_device_work': True,
    'historical_binary_binding': 'UNKNOWN',
}, indent=2))
