#!/usr/bin/env python3
"""For review: read bounded diagnostic-instance log, emit only whitelisted fields."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
STAGE = 'wio-vllm-v41-repro169-stage-20261028'
DIAG_ID = '238c1fae630aff68ef3a7c77d4a4fdd26e55019b35cd2f8e2ae3a3c776d182f3'
IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
HOST_ROOT = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028'
HOST_MOUNT = '/data/tiankuan/wio/main'
STAGE_MOUNT = '/work'
WORK = '/work/vllm-ascend-reproduce-172-on-169-20261028'
APP = '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
DIAG_RUNTIME = HOST_ROOT + '/analysis/operator-ceiling-20261028/diagnostic/runtime'
LOG = WORK + '/analysis/operator-ceiling-20261028/diagnostic/runtime/logs/service.log'
OUT = TASK / 'agents/diagnosis/diagnostic-runtime-config-evidence.json'

CODE = r'''import hashlib, json, re, stat
from pathlib import Path
p = Path(%r)
if p.resolve(strict=True) != p:
    raise SystemExit('log contains symlink')
before = p.stat()
if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > 256 * 1024 * 1024:
    raise SystemExit('log type or size outside limit')
fields = ('use_sequence_parallel_moe','enable_shared_expert_dp','multistream_overlap_shared_expert','enable_fused_mc2','enable_flashcomm1','all2all_backend')
pat = re.compile(r'(?<![A-Za-z0-9_])[' + "'\"" + r']?(' + '|'.join(fields) + r')[' + "'\"" + r']?\s*[:=]\s*[' + "'\"" + r']?([A-Za-z0-9_]+)')
backend_allowed = {'allgather_reducescatter','deepep_high_throughput','deepep_low_latency','deepep_v2','flashinfer_nvlink_one_sided','mori_high_throughput','mori_low_latency','nixl_ep','flashinfer_all2allv','pplx','naive'}
events = []
markers = {
  'flashcomm_sp_unsupported': 'FlashComm1 is enabled, but the current config does not support sp MoE. Disabling',
  'shared_multistream_forced_off': 'enable_fused_mc2 and multistream_overlap_shared_expert cannot be enabled at the same time. Setting multistream_overlap_shared_expert to False.',
  'flashcomm_disabled': 'FlashComm1 is disabled. Using flashinfer_all2allv as the all2all backend.',
}
h = hashlib.sha256()
with p.open('rb') as f:
    for line_no, line in enumerate(f, 1):
        if len(line) > 262144:
            raise SystemExit('log line exceeds limit')
        h.update(line)
        text = line.decode('utf-8', errors='replace')
        for m in pat.finditer(text):
            key, val = m.group(1), m.group(2)
            if key == 'all2all_backend':
                if val not in backend_allowed:
                    continue
                parsed = val
            elif key == 'enable_fused_mc2':
                if val not in ('0','1','2'):
                    continue
                parsed = int(val)
            else:
                if val.lower() not in ('true','false'):
                    continue
                parsed = val.lower() == 'true'
            events.append({'line': line_no, 'key': key, 'value': parsed, 'classification': 'unverified_log_field'})
        for marker, exact in markers.items():
            if exact in text:
                events.append({'line': line_no, 'marker': marker, 'classification': 'exact_message_unbound_to_rank'})
        if len(events) > 200:
            raise SystemExit('too many matching fields')
after = p.stat()
if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
    raise SystemExit('log changed during read')
print(json.dumps({'log_size':before.st_size,'log_sha256':h.hexdigest(),'events':events}))
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
    return cid, image_id, json.loads(mounts) or []


stage = inspect(STAGE)
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
             'source': LOG, 'fields_are_not_automatically_resolved': True, 'no_device_work': True})
payload = (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode()
if OUT.is_symlink() or (OUT.exists() and OUT.read_bytes() != payload):
    raise RuntimeError('output exists or is symlink; no overwrite')
if not OUT.exists():
    with OUT.open('xb') as f:
        f.write(payload)
print(json.dumps({'status': 'BOUNDED_LOG_QUERY_COMPLETE', 'output': str(OUT.relative_to(TASK)),
                  'events': len(data['events']), 'log_sha256': data['log_sha256']}))
