#!/usr/bin/env python3
"""For review: classify only diagnostic log line 24; never emit raw log text."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
STAGE = 'wio-vllm-v41-repro169-stage-20261028'
STAGE_ID = '42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602'
DIAG_ID = '238c1fae630aff68ef3a7c77d4a4fdd26e55019b35cd2f8e2ae3a3c776d182f3'
IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
HOST_ROOT = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028'
HOST_MOUNT = '/data/tiankuan/wio/main'
STAGE_MOUNT = '/work'
WORK = '/work/vllm-ascend-reproduce-172-on-169-20261028'
APP = '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
DIAG_RUNTIME = HOST_ROOT + '/analysis/operator-ceiling-20261028/diagnostic/runtime'
LOG = WORK + '/analysis/operator-ceiling-20261028/diagnostic/runtime/logs/service.log'
OUT = TASK / 'agents/diagnosis/diagnostic-line24-context.json'

CODE = r'''import hashlib, json, re, stat
from pathlib import Path
p = Path(%r)
if p.resolve(strict=True) != p:
    raise SystemExit('log contains symlink')
before = p.stat()
if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > 256 * 1024 * 1024:
    raise SystemExit('log type or size outside limit')
expected_sha = '51375755909e205512f1a7781627d79cca628d5970b198a51ab9de7247ee295d'
h = hashlib.sha256()
line24 = None
with p.open('rb') as f:
    for line_no, line in enumerate(f, 1):
        if len(line) > 262144:
            raise SystemExit('log line exceeds limit')
        h.update(line)
        if line_no == 24:
            line24 = line.decode('utf-8', errors='replace')
after = p.stat()
if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
    raise SystemExit('log changed during read')
if h.hexdigest() != expected_sha or line24 is None:
    raise SystemExit('diagnostic log hash mismatch or line missing')
head = line24[:512]
prefix = re.match(r'^\((APIServer|EngineCore(?:_DP[0-9]+)?|Worker(?:_[A-Za-z0-9]+)*) pid=[0-9]+\)\s+', head)
if prefix is None:
    process_category = 'unknown'
elif prefix.group(1) == 'APIServer':
    process_category = 'api_server'
elif prefix.group(1).startswith('EngineCore'):
    process_category = 'engine_core'
else:
    process_category = 'worker'
has_pid = prefix is not None
has_level = bool(re.search(r'(?:^|\s)(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+', head))
has_timestamp = bool(re.search(r'\b[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\b', head))
has_source_location = bool(re.search(r'\[[A-Za-z0-9_.-]+\.py:[0-9]+\]', head))
non_default = bool(re.search(r'\[[A-Za-z0-9_.-]+\.py:[0-9]+\]\s+non-default args:', head))
print(json.dumps({'line_number':24, 'log_size':before.st_size, 'log_sha256':h.hexdigest(),
                  'process_category':process_category,
                  'logger_category':'non_default_args' if non_default else 'other_or_unknown',
                  'has_pid':has_pid, 'has_level':has_level, 'has_timestamp':has_timestamp,
                  'has_source_location':has_source_location}))
''' % LOG


def remote(args):
    r = subprocess.run([SSH, shlex.join(args)], capture_output=True, text=True, timeout=90)
    if r.returncode:
        raise RuntimeError('bounded read or container inspection failed')
    return r.stdout


def inspect(target, expected_id=None):
    raw = remote(['docker', 'inspect', '--format', '{{.Id}}|{{.Image}}|{{json .Mounts}}', target]).strip()
    cid, image_id, mounts = raw.split('|', 2)
    if len(cid) != 64 or any(c not in '0123456789abcdef' for c in cid) or image_id != IMAGE:
        raise RuntimeError('unexpected container or image')
    if expected_id and cid != expected_id:
        raise RuntimeError('unexpected diagnostic container id')
    mounts_list = json.loads(mounts) or []
    if not isinstance(mounts_list, list) or any(not isinstance(m, dict) for m in mounts_list):
        raise RuntimeError('invalid mount metadata')
    # Compare every original mount field, canonically ordered even when
    # Destination/Source/Type are identical for more than one mount.
    mounts_list.sort(key=lambda m: json.dumps(m, sort_keys=True, separators=(',', ':'), ensure_ascii=True))
    return cid, image_id, mounts_list


stage = inspect(STAGE, STAGE_ID)
diag = inspect(DIAG_ID, DIAG_ID)
for item, dest, expected in ((stage, STAGE_MOUNT, HOST_MOUNT), (diag, APP, DIAG_RUNTIME)):
    matching = [m for m in item[2] if m.get('Destination') == dest]
    if len(matching) != 1 or matching[0].get('Type') != 'bind' or matching[0].get('Source') != expected:
        raise RuntimeError('unexpected diagnostic or staging mount')
for m in stage[2]:
    dest = m.get('Destination', '').rstrip('/') or '/'
    if dest not in (STAGE_MOUNT, '/') and (LOG == dest or LOG.startswith(dest + '/')):
        raise RuntimeError('diagnostic log path unexpectedly overmounted')
    if dest == '/':
        raise RuntimeError('unexpected root mount')
diag_log = APP + '/logs/service.log'
for m in diag[2]:
    dest = m.get('Destination', '').rstrip('/') or '/'
    if dest not in (APP, '/') and (diag_log == dest or diag_log.startswith(dest + '/')):
        raise RuntimeError('diagnostic container log path unexpectedly overmounted')
    if dest == '/':
        raise RuntimeError('unexpected diagnostic root mount')

data = json.loads(remote(['docker', 'exec', stage[0], 'python3', '-c', CODE]))
if inspect(stage[0]) != stage or inspect(DIAG_ID, DIAG_ID) != diag:
    raise RuntimeError('container metadata changed during read')
data.update({'diagnostic_container_id': DIAG_ID, 'staging_container_id': stage[0], 'image_id': IMAGE,
             'source': LOG, 'category_does_not_establish_worker_resolved_config': True, 'no_device_work': True})
payload = (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode()
if OUT.is_symlink() or (OUT.exists() and OUT.read_bytes() != payload):
    raise RuntimeError('output exists or is symlink; no overwrite')
if not OUT.exists():
    with OUT.open('xb') as f:
        f.write(payload)
print(json.dumps({'status': 'LINE24_CATEGORY_QUERY_COMPLETE', 'output': str(OUT.relative_to(TASK)),
                  'log_sha256': data['log_sha256']}))
