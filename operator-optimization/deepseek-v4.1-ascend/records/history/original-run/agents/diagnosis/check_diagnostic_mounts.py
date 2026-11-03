#!/usr/bin/env python3
"""For review: inspect only fixed container identities and relevant mount metadata.

No docker exec, service-log reads, device commands, or container state changes.
"""
import hashlib
import json
import shlex
import subprocess

SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
TARGETS = {
    'stage': ('42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602',
              '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/logs/service.log'),
    'diagnostic': ('238c1fae630aff68ef3a7c77d4a4fdd26e55019b35cd2f8e2ae3a3c776d182f3',
                   '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/logs/service.log'),
}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def inspect(cid, watched_path):
    args = ['docker', 'inspect', '--format', '{{.Id}}|{{.Image}}|{{json .Mounts}}', cid]
    result = subprocess.run([SSH, shlex.join(args)], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('docker inspect failed for fixed container')
    actual_id, image_id, mounts_json = result.stdout.strip().split('|', 2)
    if actual_id != cid:
        raise RuntimeError('container id does not match fixed identity')
    mounts = json.loads(mounts_json) or []
    relevant = []
    for m in mounts:
        dest = m.get('Destination', '').rstrip('/') or '/'
        if dest == '/' or watched_path == dest or watched_path.startswith(dest + '/'):
            relevant.append({k: m.get(k) for k in ('Source', 'Destination', 'Type')})
    canonical = sorted(mounts, key=lambda m: (str(m.get('Destination')), str(m.get('Source')), str(m.get('Type'))))
    return {
        'container_id': actual_id,
        'image_id': image_id,
        'relevant_mounts': relevant,
        'mount_count': len(mounts),
        'ordered_mount_metadata_sha256': digest(mounts),
        'sorted_mount_metadata_sha256': digest(canonical),
    }


first = {name: inspect(*info) for name, info in TARGETS.items()}
second = {name: inspect(*info) for name, info in TARGETS.items()}
summary = {}
for name in TARGETS:
    a, b = first[name], second[name]
    summary[name] = {
        'first': a,
        'second': b,
        'identity_changed': a['container_id'] != b['container_id'],
        'image_changed': a['image_id'] != b['image_id'],
        'relevant_mounts_changed': a['relevant_mounts'] != b['relevant_mounts'],
        'ordered_mount_metadata_changed': a['ordered_mount_metadata_sha256'] != b['ordered_mount_metadata_sha256'],
        'sorted_mount_metadata_changed': a['sorted_mount_metadata_sha256'] != b['sorted_mount_metadata_sha256'],
    }
print(json.dumps({'status': 'MOUNT_METADATA_ONLY', 'containers': summary, 'service_log_read': False, 'no_device_work': True}, indent=2))
